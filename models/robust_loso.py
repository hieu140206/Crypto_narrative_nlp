"""LOSO robustness: drop one main source, rebuild daily text features, rerun A/B/C.

Does not overwrite default parquet/CSV. Panels → data/processed/loso/<id>/
Results → outputs/tables/robust_loso_*. Registry family='robust_loso'.
Seed 42. train+valid only; dates ≤ 2023-12-31. No ALLOW_TEST.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_daily_panel import (  # noqa: E402
    MODELS,
    abs_path,
    assert_docs_have_both_models,
    build_daily_panel,
    load_config,
    prepare_docs,
    project_root,
)
from build_model_table import build_model_table_A, build_model_table_B  # noqa: E402
from build_narratives import (  # noqa: E402
    CORE_NARRATIVES,
    assign_doc_narratives,
    build_daily_narratives,
    filter_exclude_sources,
    load_narrative_config,
)
from modeltable_io import HARD_DATE_CAP, assert_no_hard_cap_dates, load_model_table  # noqa: E402
from models.har_text import TEXT_MODELS, append_test_registry, run_all as run_har_text  # noqa: E402
from models.rq1_rq2_sentiment import (  # noqa: E402
    BLOCK_LEN,
    N_BOOT,
    SEED,
    load_train_valid,
    prepare_asset_frame,
    run_rq1_vs_market,
)
from models.rq3_narratives import holm_adjust  # noqa: E402

FAMILY = "robust_loso"
LOSO_SOURCES = ("coindesk", "cointelegraph", "decrypt", "blockworks")
EST_END = "2023-12-31"
TOL = 1e-9
SKIP_COMPARE = {
    "date", "asset", "split", "complete_day", "own_low_n",
}


def _seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.pop("ALLOW_TEST", None)


def loso_groups() -> list[tuple[str, list[str]]]:
    return [("full", [])] + [(s, [s]) for s in LOSO_SOURCES]


def loso_dir(root: Path, label: str) -> Path:
    return root / "data" / "processed" / "loso" / label


def feature_outcome_cols(df: pd.DataFrame) -> list[str]:
    skip = set(SKIP_COMPARE)
    out = []
    for c in df.columns:
        if c in skip:
            continue
        if str(c).startswith("embargo_drop_") or str(c).startswith("target_end_date_"):
            continue
        out.append(c)
    return out


def _naive(s: pd.Series) -> pd.Series:
    d = pd.to_datetime(s)
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_convert("UTC").dt.tz_localize(None)
    return d.dt.normalize()


def compare_model_table_tv(
    rebuilt: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    splits: tuple[str, ...] = ("train", "valid"),
    tol: float = TOL,
) -> dict:
    """Compare feature+outcome cols on train+valid. |Δ|<tol."""
    a = rebuilt.loc[rebuilt["split"].isin(splits)].copy()
    b = reference.loc[reference["split"].isin(splits)].copy()
    a["date"] = _naive(a["date"])
    b["date"] = _naive(b["date"])
    keys = ["date", "asset"]
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    cols = [c for c in feature_outcome_cols(a) if c in b.columns]
    missing_in_ref = [c for c in feature_outcome_cols(a) if c not in b.columns]
    missing_in_reb = [c for c in feature_outcome_cols(b) if c not in a.columns]
    max_abs = 0.0
    n_mismatch = 0
    worst = None
    if len(a) != len(b):
        return {
            "ok": False,
            "n_rebuilt": int(len(a)),
            "n_reference": int(len(b)),
            "n_cols": len(cols),
            "max_abs_delta": float("inf"),
            "n_mismatch_cells": -1,
            "worst_col": "nrow",
            "missing_in_ref": missing_in_ref,
            "missing_in_reb": missing_in_reb,
            "tol": tol,
        }
    key_ok = (a["date"].to_numpy() == b["date"].to_numpy()).all() and (
        a["asset"].astype(str).to_numpy() == b["asset"].astype(str).to_numpy()
    ).all()
    if not key_ok:
        return {
            "ok": False,
            "n_rebuilt": int(len(a)),
            "n_reference": int(len(b)),
            "n_cols": len(cols),
            "max_abs_delta": float("inf"),
            "n_mismatch_cells": -1,
            "worst_col": "keys",
            "missing_in_ref": missing_in_ref,
            "missing_in_reb": missing_in_reb,
            "tol": tol,
        }
    for c in cols:
        xa = pd.to_numeric(a[c], errors="coerce")
        xb = pd.to_numeric(b[c], errors="coerce")
        both_nan = xa.isna() & xb.isna()
        delta = (xa - xb).abs()
        delta = delta.where(~both_nan, 0.0)
        # one-sided NaN counts as mismatch
        one_nan = xa.isna() ^ xb.isna()
        delta = delta.where(~one_nan, np.inf)
        col_max = float(delta.max()) if len(delta) else 0.0
        n_bad = int((delta > tol).sum())
        if col_max > max_abs:
            max_abs = col_max
            worst = c
        n_mismatch += n_bad
    ok = bool(np.isfinite(max_abs) and max_abs < tol and not missing_in_ref and not missing_in_reb)
    return {
        "ok": ok,
        "n_rebuilt": int(len(a)),
        "n_reference": int(len(b)),
        "n_cols": len(cols),
        "max_abs_delta": max_abs if np.isfinite(max_abs) else float("inf"),
        "n_mismatch_cells": n_mismatch,
        "worst_col": worst,
        "missing_in_ref": missing_in_ref,
        "missing_in_reb": missing_in_reb,
        "tol": tol,
    }


def _assert_est_dates(df: pd.DataFrame, *, name: str) -> None:
    if df is None or df.empty:
        return
    assert_no_hard_cap_dates(df, cap=HARD_DATE_CAP)
    d = _naive(df["date"])
    if (d > pd.Timestamp(EST_END)).any():
        raise AssertionError("%s có date > %s" % (name, EST_END))
    if "split" in df.columns and (df["split"].astype(str) == "test").any():
        raise AssertionError("%s chứa split=test" % name)


def build_loso_artifacts(
    root: Path,
    label: str,
    exclude_sources: list[str],
    *,
    params: dict,
    paths: dict,
) -> dict[str, Path]:
    """Rebuild panel + narratives + model tables under data/processed/loso/<label>/."""
    out = loso_dir(root, label)
    out.mkdir(parents=True, exist_ok=True)
    panel_path = out / "daily_panel.parquet"
    narr_path = out / "daily_narratives.parquet"
    doc_path = out / "narrative_doc.parquet"
    path_a = out / "model_table_A.parquet"
    path_b = out / "model_table_B.parquet"

    text = pd.read_parquet(
        abs_path(root, paths["text_clean_path"]),
        columns=["doc_id", "news_date", "source", "currencies", "in_main_sample", "title"],
    )
    docs = prepare_docs(text, exclude_sources=exclude_sources)
    sent_path = abs_path(root, paths["sentiment_doc_path"])
    sent = pd.read_parquet(sent_path)
    if "text_variant" in sent.columns:
        sent = sent[sent["text_variant"] == "title"]
    sent = sent[sent["model"].isin(MODELS)]
    assert_docs_have_both_models(docs, sent, models=MODELS)

    start = params["sample"]["start"]
    end = params["sample"]["t_end"]
    abn_window = int(params.get("abn_attention_window", 30))
    panel = build_daily_panel(docs, sent, start=start, end=end, abn_window=abn_window)
    panel.to_parquet(panel_path, index=False)

    cfg = load_narrative_config(root / "config" / "narratives.yaml")
    narr_meta = cfg["narratives"]
    names = list(narr_meta.keys())
    narr_for_assign = {
        n: {"patterns": narr_meta[n]["patterns"], "excludes": narr_meta[n]["excludes"]}
        for n in names
    }
    default_doc = abs_path(root, paths.get("narrative_doc_path", "data/interim/narrative_doc.parquet"))
    main_df = text.loc[text["in_main_sample"]].copy()
    main_df = filter_exclude_sources(main_df, exclude_sources)
    if default_doc.exists():
        nd = pd.read_parquet(default_doc)
        docs_daily = main_df.merge(nd, on="doc_id", how="inner")
        if len(docs_daily) != main_df["doc_id"].nunique():
            assigned = assign_doc_narratives(main_df["title"], narr_for_assign, flags=cfg["flags"])
            docs_daily = pd.concat(
                [main_df[["doc_id", "news_date", "currencies", "title", "source"]].reset_index(drop=True),
                 assigned.reset_index(drop=True)],
                axis=1,
            )
            pd.concat([main_df[["doc_id"]].reset_index(drop=True), assigned.reset_index(drop=True)], axis=1).to_parquet(
                doc_path, index=False
            )
        else:
            docs_daily[["doc_id", *[c for c in nd.columns if c != "doc_id"]]].drop_duplicates("doc_id").to_parquet(
                doc_path, index=False
            )
    else:
        assigned = assign_doc_narratives(main_df["title"], narr_for_assign, flags=cfg["flags"])
        docs_daily = pd.concat(
            [main_df[["doc_id", "news_date", "currencies", "title", "source"]].reset_index(drop=True),
             assigned.reset_index(drop=True)],
            axis=1,
        )
        pd.concat([main_df[["doc_id"]].reset_index(drop=True), assigned.reset_index(drop=True)], axis=1).to_parquet(
            doc_path, index=False
        )

    daily_n = build_daily_narratives(
        docs_daily, start, end, names, abn_window=abn_window,
        flag_names=["price_desc"], exclude_sources=exclude_sources,
    )
    daily_n.to_parquet(narr_path, index=False)

    market = pd.read_parquet(abs_path(root, paths["market_daily_path"]))
    fred = pd.read_parquet(abs_path(root, paths["fred_path"]))
    table_a = build_model_table_A(panel, daily_n, market, fred, params)
    table_b = build_model_table_B(panel, daily_n, market, fred, params)
    table_a.to_parquet(path_a, index=False)
    table_b.to_parquet(path_b, index=False)

    return {
        "dir": out,
        "panel": panel_path,
        "narr": narr_path,
        "doc": doc_path,
        "table_a": path_a,
        "table_b": path_b,
        "n_docs": int(docs["doc_id"].nunique()),
    }


def coverage_row(
    panel: pd.DataFrame,
    *,
    label: str,
    exclude_sources: list[str],
    n_docs: int,
    params: dict,
) -> dict:
    g = panel.loc[panel["scope"] == "all"].copy()
    g["date"] = _naive(g["date"])
    vol = g["vader_NewsVolume_t"].astype(float)
    train_a, train_b = (pd.Timestamp(x) for x in params["split"]["train"])
    val_a, val_b = (pd.Timestamp(x) for x in params["split"]["validation"])
    tv = g[(g["date"] >= train_a) & (g["date"] <= val_b)]
    vol_tv = tv["vader_NewsVolume_t"].astype(float)
    return {
        "loso_id": label,
        "exclude_source": ",".join(exclude_sources) if exclude_sources else "",
        "n_docs": n_docs,
        "n_calendar_days": int(len(g)),
        "median_articles_per_day": float(vol.median()) if len(vol) else float("nan"),
        "pct_days_lt3": float((vol < 3).mean()) if len(vol) else float("nan"),
        "n_days_lt3": int((vol < 3).sum()) if len(vol) else 0,
        "train_valid_start": str(train_a.date()),
        "train_valid_end": str(val_b.date()),
        "n_days_train_valid": int(len(tv)),
        "median_articles_per_day_tv": float(vol_tv.median()) if len(vol_tv) else float("nan"),
        "pct_days_lt3_tv": float((vol_tv < 3).mean()) if len(vol_tv) else float("nan"),
        "n_days_lt3_tv": int((vol_tv < 3).sum()) if len(vol_tv) else 0,
        "core_narratives": ",".join(CORE_NARRATIVES),
    }


def _registry_rows_a(df: pd.DataFrame, ts: str, label: str) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "timestamp": ts,
            "script": "models/robust_loso.py",
            "model": f"{r['model']}|{r['y']}|{label}",
            "asset": r["asset"],
            "h": "",
            "split": "train+valid",
            "n_oos": r.get("n", ""),
            "p_raw": r.get("p_boot", float("nan")),
            "baseline": "market",
            "family": FAMILY,
            "scale": "1sd",
            "note": "loso:%s" % label,
        })
    return rows


def run_part_a(
    table_a: Path,
    label: str,
    exclude_sources: list[str],
    *,
    root: Path,
    n_boot: int,
    seed: int,
    ts: str,
) -> tuple[pd.DataFrame, list[dict]]:
    raw = load_train_valid(
        "A", root=root, panel="title", table_path=table_a,
        exclude_sources=exclude_sources,
    )
    _assert_est_dates(raw, name="A/%s" % label)
    parts = []
    for asset in ("BTC", "ETH"):
        df = prepare_asset_frame(raw, asset)
        _assert_est_dates(df, name="A/%s/%s" % (label, asset))
        part = run_rq1_vs_market(
            df, asset, family=FAMILY, label=label,
            n_boot=n_boot, seed=seed, holm=True,
        )
        part["loso_id"] = label
        part["exclude_source"] = ",".join(exclude_sources) if exclude_sources else ""
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    return out, _registry_rows_a(out, ts, label)


def run_part_b(
    table_b: Path,
    label: str,
    exclude_sources: list[str],
    *,
    root: Path,
    seed: int,
    registry_path: Path,
) -> pd.DataFrame:
    har_dir = loso_dir(root, label) / "har"
    summary = run_har_text(
        root=root,
        out_dir=har_dir,
        panel="title",
        seed=seed,
        models=dict(TEXT_MODELS),
        include_deseason_robust=False,
        register=True,
        stage="valid",
        registry_path=registry_path,
        table_b_path=table_b,
        registry_family=FAMILY,
        registry_note="loso:%s" % label,
        exclude_sources=exclude_sources,
    )
    # assert valid dates
    for asset in ("BTC", "ETH"):
        for h in (1, 3, 7):
            p = har_dir / f"har_text_{asset}_h{h}.csv"
            if p.exists():
                oos = pd.read_csv(p)
                if "date" in oos.columns:
                    _assert_est_dates(oos, name="B/%s/%s/h%s" % (label, asset, h))
    out = summary.copy()
    out["loso_id"] = label
    out["exclude_source"] = ",".join(exclude_sources) if exclude_sources else ""
    out["family"] = FAMILY
    return out


def run_reproduce_check(root: Path, paths: dict, rebuilt_b: Path) -> dict:
    ref_path = abs_path(root, paths["model_table_B_path"])
    if not ref_path.exists():
        return {"ok": False, "error": "missing default model_table_B.parquet"}
    rebuilt = pd.read_parquet(rebuilt_b)
    reference = pd.read_parquet(ref_path)
    chk = compare_model_table_tv(rebuilt, reference)
    chk["rebuilt_path"] = str(rebuilt_b)
    chk["reference_path"] = str(ref_path)
    return chk


def run_all(
    root: Path | None = None,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    skip_if_check_fail: bool = True,
) -> dict:
    root = root or project_root()
    _seed(seed)
    if os.environ.get("ALLOW_TEST") == "1":
        raise RuntimeError("LOSO cấm ALLOW_TEST")
    params, paths = load_config(root)
    seed = int(params.get("seed", seed))
    _seed(seed)

    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    reg_path = out_dir / "test_registry.csv"
    n_reg_before = int(len(pd.read_csv(reg_path))) if reg_path.exists() else 0
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    groups = loso_groups()
    check: dict | None = None
    rows_a: list[pd.DataFrame] = []
    rows_b: list[pd.DataFrame] = []
    rows_c: list[dict] = []
    reg_a: list[dict] = []

    for i, (label, excl) in enumerate(groups):
        print("=== LOSO %s exclude=%r ===" % (label, excl))
        arts = build_loso_artifacts(root, label, excl, params=params, paths=paths)
        panel = pd.read_parquet(arts["panel"])
        rows_c.append(coverage_row(
            panel, label=label, exclude_sources=excl,
            n_docs=arts["n_docs"], params=params,
        ))

        if label == "full":
            check = run_reproduce_check(root, paths, arts["table_b"])
            (loso_dir(root, label) / "reproduce_check.json").write_text(
                json.dumps(check, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            print("REPRODUCE CHECK:", json.dumps({k: check[k] for k in (
                "ok", "n_rebuilt", "n_reference", "n_cols", "max_abs_delta",
                "n_mismatch_cells", "worst_col",
            ) if k in check}, default=str))
            if skip_if_check_fail and not check.get("ok"):
                (out_dir / "robust_loso_check.json").write_text(
                    json.dumps(check, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                raise SystemExit(
                    "STOP: exclude_sources=[] không khớp model_table_B train+valid "
                    "(max|Δ|=%s col=%s)"
                    % (check.get("max_abs_delta"), check.get("worst_col"))
                )

        a_df, a_reg = run_part_a(
            arts["table_a"], label, excl, root=root,
            n_boot=n_boot, seed=seed, ts=ts,
        )
        rows_a.append(a_df)
        reg_a.extend(a_reg)

        b_df = run_part_b(
            arts["table_b"], label, excl, root=root, seed=seed,
            registry_path=reg_path,
        )
        rows_b.append(b_df)

    table_a = pd.concat(rows_a, ignore_index=True)
    table_b = pd.concat(rows_b, ignore_index=True)
    table_c = pd.DataFrame(rows_c)

    # Holm already per (loso_id, asset) inside run_rq1_vs_market (6) and har_text (9)
    path_a = out_dir / "robust_loso_A.csv"
    path_b = out_dir / "robust_loso_B.csv"
    path_c = out_dir / "robust_loso_C.csv"
    table_a.to_csv(path_a, index=False)
    table_b.to_csv(path_b, index=False)
    table_c.to_csv(path_c, index=False)

    append_test_registry(reg_a, reg_path)
    n_reg_after = int(len(pd.read_csv(reg_path)))
    n_added = n_reg_after - n_reg_before

    check_path = out_dir / "robust_loso_check.json"
    check_path.write_text(
        json.dumps(check or {}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    meta = {
        "timestamp": ts,
        "seed": seed,
        "n_boot": n_boot,
        "block_len": BLOCK_LEN,
        "family": FAMILY,
        "est_end": EST_END,
        "hard_cap": HARD_DATE_CAP,
        "groups": [{"loso_id": a, "exclude_sources": b} for a, b in groups],
        "check": check,
        "n_registry_before": n_reg_before,
        "n_registry_after": n_reg_after,
        "n_registry_added": n_added,
        "holm_A": "per loso_id × asset (6 tests: 2 models × 3 y)",
        "holm_B": "per loso_id × asset (9 tests: 3 models × 3 h) via har_text",
        "paths": {"A": str(path_a), "B": str(path_b), "C": str(path_c)},
    }
    (out_dir / "robust_loso.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("A ->", path_a)
    print("B ->", path_b)
    print("C ->", path_c)
    print("Registry +%d → %d" % (n_added, n_reg_after))
    return {
        "A": table_a, "B": table_b, "C": table_c,
        "check": check, "meta": meta,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="LOSO robustness A/B/C (train+valid)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    run_all(n_boot=args.n_boot)


if __name__ == "__main__":
    main()

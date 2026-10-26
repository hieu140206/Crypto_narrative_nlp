"""build_narratives.py - Narrative đa nhãn từ title (RQ3, bản chính) — dict v3.

Doc-level: narrative_doc.parquet = doc_id + bool narratives + other + n_narratives
  + price_desc + is_template_price (template → chỉ other, xoá mọi narrative).

Daily: Att_k / LogRatio_k / AbnAtt_k + Att_price_desc; VIF trên AbnAtt_k nhóm chính.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)

SCOPES = ("all", "BTC", "ETH")
VIF_TRAIN = ("2021-03-31", "2022-12-31")
CORE_NARRATIVES = (
    "regulation", "etf", "institutional_adoption", "hacks_security",
    "defi", "stablecoins", "monetary_policy",
)
# Tiêu đề bắt đầu bằng các cụm này → template giá
_TEMPLATE_PREFIX = re.compile(
    r"^\s*price\s+(analysis|predictions?)\b", re.IGNORECASE
)
# ≥4 ticker viết hoa phân tách bằng dấu phẩy: SPX, DXY, BTC, ETH
_TICKER_LIST = re.compile(
    r"\b[A-Z]{2,6}\b(?:\s*,\s*\b[A-Z]{2,6}\b){3,}"
)


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_config(root: Path) -> tuple[dict, dict]:
    import yaml
    params = yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}
    paths = yaml.safe_load((root / "config" / "paths.yaml").read_text(encoding="utf-8")) or {}
    return params, paths


def abs_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def load_narrative_config(path: Path) -> dict:
    """Trả narratives (patterns+excludes+exploratory) và flags."""
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    narr_raw = data.get("narratives") or {}
    narratives: dict[str, dict] = {}
    for name, spec in narr_raw.items():
        if isinstance(spec, list):
            patterns, excludes = spec, []
            exploratory = name not in CORE_NARRATIVES
        else:
            patterns = spec.get("patterns") or []
            excludes = spec.get("excludes") or []
            exploratory = bool(spec.get("exploratory", False))
        narratives[name] = {
            "exploratory": exploratory,
            "patterns": [re.compile(p, re.IGNORECASE) for p in patterns],
            "excludes": [re.compile(p, re.IGNORECASE) for p in excludes],
        }
    flags: dict[str, list[re.Pattern]] = {}
    for name, spec in (data.get("flags") or {}).items():
        pats = spec if isinstance(spec, list) else (spec.get("patterns") or [])
        flags[name] = [re.compile(p, re.IGNORECASE) for p in pats]
    return {
        "version": data.get("version", 1),
        "narratives": narratives,
        "flags": flags,
    }


def load_narratives(path: Path) -> dict[str, list[re.Pattern]]:
    """API cũ: name → list patterns (cho tests)."""
    cfg = load_narrative_config(path)
    return {n: v["patterns"] for n, v in cfg["narratives"].items()}


def match_title(title: str, patterns: list[re.Pattern]) -> bool:
    text = title or ""
    return any(p.search(text) for p in patterns)


def is_template_price(title: str) -> bool:
    """Price analysis/prediction… hoặc ≥4 ticker phân tách bằng dấu phẩy."""
    t = title or ""
    if _TEMPLATE_PREFIX.search(t):
        return True
    return bool(_TICKER_LIST.search(t))


def match_narrative(
    title: str,
    patterns: list[re.Pattern],
    excludes: list[re.Pattern] | None = None,
) -> bool:
    text = title or ""
    if excludes and any(e.search(text) for e in excludes):
        return False
    return match_title(text, patterns)


def _norm_narr_meta(narratives: dict) -> dict[str, dict]:
    """Chuẩn hoá {name: patterns} hoặc {name: {patterns, excludes}} → meta."""
    meta: dict[str, dict] = {}
    for name, spec in narratives.items():
        if isinstance(spec, list):
            meta[name] = {"patterns": spec, "excludes": []}
        elif isinstance(spec, dict) and "patterns" in spec:
            meta[name] = {
                "patterns": spec["patterns"],
                "excludes": list(spec.get("excludes") or []),
            }
        else:
            raise TypeError(f"narrative {name!r}: expected list or dict with patterns")
    return meta


def assign_doc_narratives(
    titles: pd.Series,
    narratives: dict,
    flags: dict[str, list[re.Pattern]] | None = None,
) -> pd.DataFrame:
    """Thuần: title → narratives + other + n_narratives + price_desc + is_template_price.

    is_template_price=True → xoá mọi narrative, other=1, price_desc=1.
    """
    meta = _norm_narr_meta(narratives)
    names = list(meta.keys())

    data = {
        name: titles.map(
            lambda t, p=meta[name]["patterns"], e=meta[name]["excludes"]: match_narrative(t, p, e)
        )
        for name in names
    }
    out = pd.DataFrame(data)

    templates = titles.map(is_template_price)
    out["is_template_price"] = templates.values
    if bool(templates.any()):
        out.loc[templates.values, names] = False

    out["n_narratives"] = out[names].sum(axis=1).astype(int)
    out["other"] = out["n_narratives"] == 0

    price_pats = (flags or {}).get("price_desc", [])
    price = titles.map(lambda t, ps=price_pats: match_title(t, ps)) if price_pats else pd.Series(
        False, index=titles.index
    )
    out["price_desc"] = np.asarray(price.values, dtype=bool) | np.asarray(templates.values, dtype=bool)
    return out


def overlap_matrix(doc: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Tỷ lệ bài thuộc cả hai: M[i,j] = |i∩j| / |i| (theo hàng i)."""
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        na = int(doc[a].sum())
        for b in names:
            mat.loc[a, b] = np.nan if na == 0 else float((doc[a] & doc[b]).sum()) / na
    return mat


def abnatt_corr(daily: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Tương quan Pearson AbnAtt_k trên scope=all."""
    g = daily[daily["scope"] == "all"]
    return g[[f"AbnAtt_{n}" for n in names]].corr()
def currency_has(currencies, tag: str) -> bool:
    if currencies is None or (isinstance(currencies, float) and np.isnan(currencies)):
        return False
    try:
        if isinstance(currencies, str):
            return tag in currencies.upper()
        return tag in {str(x).upper() for x in list(currencies)}
    except TypeError:
        return False


def filter_scope(docs: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return docs
    return docs.loc[docs["currencies"].map(lambda c: currency_has(c, scope))]


def abn_attention(n_k: pd.Series, window: int = 30) -> pd.Series:
    logn = np.log1p(n_k.astype(float))
    past = logn.shift(1).rolling(window, min_periods=window).mean()
    return logn - past


def build_scope_narratives(
    docs: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    scope: str,
    narrative_names: list[str],
    abn_window: int = 30,
    flag_names: list[str] | None = None,
) -> pd.DataFrame:
    scoped = filter_scope(docs, scope).copy()
    scoped["news_date"] = pd.to_datetime(scoped["news_date"]).dt.normalize()
    base = pd.DataFrame({"date": calendar})
    base["scope"] = scope
    flag_names = flag_names or []

    if scoped.empty:
        vol = pd.Series(0, index=calendar)
    else:
        vol = scoped.groupby("news_date").size().reindex(calendar, fill_value=0)

    base["NewsVolume_t"] = vol.values.astype(int)
    empty = base["NewsVolume_t"] == 0

    if scoped.empty:
        for name in narrative_names:
            base[f"n_{name}"] = 0
        base["n_other"] = 0
        for f in flag_names:
            base[f"n_{f}"] = 0
    else:
        for name in narrative_names:
            cnt = scoped.loc[scoped[name]].groupby("news_date").size()
            base[f"n_{name}"] = cnt.reindex(calendar, fill_value=0).values.astype(int)
        cnt_o = scoped.loc[scoped["other"]].groupby("news_date").size()
        base["n_other"] = cnt_o.reindex(calendar, fill_value=0).values.astype(int)
        for f in flag_names:
            if f in scoped.columns:
                cnt = scoped.loc[scoped[f]].groupby("news_date").size()
                base[f"n_{f}"] = cnt.reindex(calendar, fill_value=0).values.astype(int)
            else:
                base[f"n_{f}"] = 0

    N = base["NewsVolume_t"].astype(float)
    for name in narrative_names:
        nk = base[f"n_{name}"].astype(float)
        att = nk / N.replace(0, np.nan)
        base[f"Att_{name}"] = att
        base.loc[empty, f"Att_{name}"] = np.nan
        base[f"LogRatio_{name}"] = np.log((nk + 0.5) / (base["n_other"].astype(float) + 0.5))
        base[f"AbnAtt_{name}"] = abn_attention(base[f"n_{name}"], window=abn_window).values

    for f in flag_names:
        nk = base[f"n_{f}"].astype(float)
        base[f"Att_{f}"] = nk / N.replace(0, np.nan)
        base.loc[empty, f"Att_{f}"] = np.nan

    base["date"] = pd.to_datetime(base["date"]).dt.tz_localize(None)
    return base


def filter_exclude_sources(
    docs: pd.DataFrame,
    exclude_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Bỏ nguồn (không phân biệt hoa/thường). None = không lọc; [] = giữ hết."""
    if not exclude_sources:
        return docs.reset_index(drop=True) if hasattr(docs, "reset_index") else docs
    if "source" not in docs.columns:
        raise KeyError("exclude_sources yêu cầu cột source")
    excl = {s.lower() for s in exclude_sources}
    out = docs.loc[~docs["source"].astype(str).str.lower().isin(excl)]
    return out.reset_index(drop=True)


def build_daily_narratives(
    docs: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    narrative_names: list[str],
    abn_window: int = 30,
    flag_names: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()
    calendar = pd.date_range(start, end, freq="D")
    docs = filter_exclude_sources(docs, exclude_sources)
    docs = docs.copy()
    docs["news_date"] = pd.to_datetime(docs["news_date"]).dt.tz_localize(None).dt.normalize()
    docs = docs[(docs["news_date"] >= start) & (docs["news_date"] <= end)]
    parts = [
        build_scope_narratives(
            docs, calendar, scope, narrative_names, abn_window, flag_names=flag_names
        )
        for scope in SCOPES
    ]
    return pd.concat(parts, ignore_index=True).sort_values(["scope", "date"]).reset_index(drop=True)


def vif_series(
    daily: pd.DataFrame,
    narrative_names: list[str],
    prefix: str = "AbnAtt_",
    train_start: str = VIF_TRAIN[0],
    train_end: str = VIF_TRAIN[1],
) -> dict[str, float]:
    """VIF trên biến prefix+name (mặc định AbnAtt_k — đầu vào mô hình)."""
    g = daily[daily["scope"] == "all"].copy()
    g["date"] = pd.to_datetime(g["date"])
    mask = (g["date"] >= pd.Timestamp(train_start)) & (g["date"] <= pd.Timestamp(train_end))
    cols = [f"{prefix}{n}" for n in narrative_names]
    X = g.loc[mask, cols].dropna()
    if len(X) < len(cols) + 2:
        return {n: float("nan") for n in narrative_names}
    Xv = X.values.astype(float)
    vifs = {}
    for i, name in enumerate(narrative_names):
        y = Xv[:, i]
        Z = np.delete(Xv, i, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        try:
            beta, _, _, _ = np.linalg.lstsq(Z, y, rcond=None)
            yhat = Z @ beta
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2 = min(max(r2, 0.0), 0.999999)
            vifs[name] = float(1.0 / (1.0 - r2))
        except np.linalg.LinAlgError:
            vifs[name] = float("nan")
    return vifs


def vif_att(
    daily: pd.DataFrame,
    narrative_names: list[str],
    train_start: str = VIF_TRAIN[0],
    train_end: str = VIF_TRAIN[1],
) -> dict[str, float]:
    """Tương thích cũ: VIF trên AbnAtt_k (không còn Att_k)."""
    return vif_series(
        daily, narrative_names, prefix="AbnAtt_",
        train_start=train_start, train_end=train_end,
    )


def write_samples(
    docs: pd.DataFrame,
    narrative_names: list[str],
    out_path: Path,
    seed: int = 42,
    n: int = 30,
    n_other: int = 30,
    exploratory: dict[str, bool] | None = None,
    version: int = 3,
) -> None:
    rng = np.random.default_rng(seed)
    exploratory = exploratory or {}
    lines = [
        f"# Narrative title samples v{version} (kiểm tay)",
        "",
        f"Mỗi narrative: {n} tiêu đề ngẫu nhiên (seed {seed}); other: {n_other} tiêu đề.",
        "",
    ]
    for name in narrative_names:
        hit = docs.loc[docs[name], "title"].dropna().unique()
        tag = "exploratory" if exploratory.get(name) else "core"
        if len(hit) == 0:
            lines.append(f"## {name} [{tag}]\n\n_(không có bài khớp)_\n")
            continue
        pick = rng.choice(hit, size=min(n, len(hit)), replace=False)
        lines.append(f"## {name} [{tag}] (n={len(hit)})\n")
        for t in pick:
            lines.append(f"- {t}")
        lines.append("")
    other = docs.loc[docs["other"], "title"].dropna().unique()
    pick = rng.choice(other, size=min(n_other, len(other)), replace=False) if len(other) else []
    lines.append(f"## other (n={len(other)})\n")
    for t in pick:
        lines.append(f"- {t}")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build narrative features from titles (v4)")
    parser.add_argument("--skip-samples", action="store_true")
    parser.add_argument(
        "--exclude-sources", nargs="*", default=None,
        help="Bỏ nguồn khỏi daily narratives (LOSO). Cần --output-daily.",
    )
    parser.add_argument("--output-daily", default=None, help="Override daily_narratives.parquet")
    parser.add_argument("--output-doc", default=None, help="Override narrative_doc.parquet")
    args = parser.parse_args()

    root = project_root()
    params, paths = load_config(root)
    seed = int(params.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    cfg = load_narrative_config(root / "config" / "narratives.yaml")
    narr_meta = cfg["narratives"]
    names = list(narr_meta.keys())
    exploratory_map = {n: narr_meta[n]["exploratory"] for n in names}
    core = [n for n in names if not exploratory_map[n]]
    explor = [n for n in names if exploratory_map[n]]
    # full meta (patterns + excludes) cho assign
    narr_for_assign = {
        n: {"patterns": narr_meta[n]["patterns"], "excludes": narr_meta[n]["excludes"]}
        for n in names
    }
    flags = cfg["flags"]
    flag_names = list(flags.keys())  # price_desc; is_template_price tính trong assign

    text = pd.read_parquet(
        abs_path(root, paths["text_clean_path"]),
        columns=["doc_id", "title", "news_date", "currencies", "in_main_sample", "source"],
    )
    main_df = text.loc[text["in_main_sample"]].copy().reset_index(drop=True)
    main_df = filter_exclude_sources(main_df, args.exclude_sources)
    assigned = assign_doc_narratives(main_df["title"], narr_for_assign, flags=flags)
    doc = pd.concat([main_df[["doc_id"]], assigned], axis=1)

    default_doc = abs_path(root, paths.get("narrative_doc_path", "data/interim/narrative_doc.parquet"))
    default_daily = abs_path(root, paths.get("daily_narratives_path", "data/processed/daily_narratives.parquet"))
    if args.exclude_sources is not None:
        if not args.output_daily:
            sys.exit("--exclude-sources cần --output-daily (không ghi đè daily_narratives.parquet)")
        if not args.output_doc:
            sys.exit("--exclude-sources cần --output-doc (không ghi đè narrative_doc.parquet)")
    doc_out = abs_path(root, args.output_doc) if args.output_doc else default_doc
    daily_out = abs_path(root, args.output_daily) if args.output_daily else default_daily
    if args.exclude_sources is not None and (
        doc_out.resolve() == default_doc.resolve() or daily_out.resolve() == default_daily.resolve()
    ):
        sys.exit("exclude_sources: từ chối ghi đè narrative mặc định")
    doc_out.parent.mkdir(parents=True, exist_ok=True)
    doc.to_parquet(doc_out, index=False)

    docs_daily = pd.concat([
        main_df[["doc_id", "news_date", "currencies", "title", "source"]],
        assigned,
    ], axis=1)

    start = params["sample"]["start"]
    end = params["sample"]["t_end"]
    abn_window = int(params.get("abn_attention_window", 30))
    # Att_price_desc từ cờ price_desc (đã gồm template)
    daily = build_daily_narratives(
        docs_daily, start, end, names, abn_window=abn_window, flag_names=["price_desc"],
        exclude_sources=args.exclude_sources,
    )

    daily_out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(daily_out, index=False)

    rates = {n: float(doc[n].mean()) for n in names}
    pct_other = float(doc["other"].mean())
    n_template = int(doc["is_template_price"].sum())
    flag_rates = {
        "price_desc": float(doc["price_desc"].mean()),
        "is_template_price": float(doc["is_template_price"].mean()),
    }

    zero_days: dict[str, dict[str, int]] = {}
    for scope in SCOPES:
        g = daily[daily["scope"] == scope]
        zero_days[scope] = {n: int((g[f"n_{n}"] == 0).sum()) for n in names}

    vif_names = [n for n in CORE_NARRATIVES if n in names]
    vifs = vif_series(daily, vif_names, prefix="AbnAtt_")
    vif_warn = [n for n, v in vifs.items() if v == v and v > 10]

    corr = abnatt_corr(daily, vif_names)
    overlap = overlap_matrix(doc, vif_names)

    log = {
        "version": cfg.get("version", 4),
        "n_docs": len(doc),
        "n_is_template_price": n_template,
        "narrative_names": names,
        "exploratory": exploratory_map,
        "core_narratives": core,
        "exploratory_narratives": explor,
        "flag_names": ["price_desc", "is_template_price"],
        "share_per_narrative": rates,
        "share_flags": flag_rates,
        "pct_other": pct_other,
        "pct_matched": 1 - pct_other,
        "zero_days_by_scope": zero_days,
        "days_per_scope": daily.groupby("scope")["date"].nunique().to_dict(),
        "vif_variables": [f"AbnAtt_{n}" for n in vif_names],
        "vif_train_window": list(VIF_TRAIN),
        "vif_abnatt_core_scope_all": vifs,
        "vif_warnings_gt10": vif_warn,
        "abnatt_corr_core_scope_all": corr.round(4).to_dict(),
        "overlap_core_row_norm": overlap.round(4).to_dict(),
        "doc_path": str(doc_out),
        "daily_path": str(daily_out),
        "exclude_sources": args.exclude_sources,
    }
    log_path = daily_out.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if not args.skip_samples:
        write_samples(
            docs_daily, names,
            root / "audit" / "narrative_samples_v4.md",
            seed=seed, n=30, n_other=30, exploratory=exploratory_map, version=4,
        )

    print("Doc %d -> %s" % (len(doc), doc_out))
    print("Daily %d -> %s" % (len(daily), daily_out))
    print("is_template_price: %d (%.2f%%)" % (n_template, 100 * flag_rates["is_template_price"]))
    print("pct_other=%.1f%% matched=%.1f%%" % (100 * pct_other, 100 * (1 - pct_other)))
    print("core share:", {k: round(rates[k], 3) for k in core})
    print("exploratory share:", {k: round(rates[k], 3) for k in explor})
    print("flags:", {k: round(v, 3) for k, v in flag_rates.items()})
    print("VIF vars:", [f"AbnAtt_{n}" for n in vif_names])
    print("VIF(core AbnAtt_k):", {k: round(v, 2) if v == v else None for k, v in vifs.items()})
    if vif_warn:
        print("WARNING VIF>10:", vif_warn)
    print("AbnAtt corr (core):\n", corr.round(3).to_string())
    print("Overlap |i∩j|/|i| (core):\n", overlap.round(3).to_string())
    print("Log:", log_path)


if __name__ == "__main__":
    main()

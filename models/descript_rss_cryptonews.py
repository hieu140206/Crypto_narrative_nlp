"""Descriptive RSS holdout + cryptonews cross-source (text features only).

Part 1: clean_rss → score title (FinBERT/VADER) → narratives v4; Wilson CI vs
main-sample 2025-01-01→2025-06-30. No hypothesis tests.

Part 2: cryptonews-only daily series vs four-source main panel; Pearson/Spearman
with moving-block bootstrap CI (block=10, B=5000, seed=42). No test_registry.
No ALLOW_TEST. Seed 42.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clean_rss  # noqa: E402
import score_sentiment as ss  # noqa: E402
from build_daily_panel import (  # noqa: E402
    abs_path,
    build_daily_panel,
    load_config,
    project_root,
)
from build_narratives import (  # noqa: E402
    CORE_NARRATIVES,
    assign_doc_narratives,
    load_narrative_config,
)
from models.calendar_features import dow_dummy_frame  # noqa: E402
from models.rq1_rq2_sentiment import moving_block_indices  # noqa: E402

SEED = 42
BLOCK_LEN = 10
N_BOOT = 5000
FROZEN_NARR_LF = "78b5a05b559666b32599e246f153858abe355720c8f148b86941707302cd99d9"
MAIN_COMPARE_START = "2025-01-01"
MAIN_COMPARE_END = "2025-06-30"
CN_START = "2023-06-27"
CN_END = "2025-06-30"
CN_MIN_N = 3
PERIODS = (
    ("p1_2023h2", "2023-06-27", "2023-12-31"),
    ("p2_2024_2025h1", "2024-01-01", "2025-06-30"),
)
ASCII_MIN_RATIO = 0.90  # same threshold as clean_merge (descriptive only)
VARS = ("AbnAttention", "Sent_FinBERT", "Sent_VADER")
VERSIONS = ("raw", "dow_resid", "noprice")


def _seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval for proportion k/n. Returns (p, lo, hi)."""
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    half = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo = (centre - half) / denom
    hi = (centre + half) / denom
    return float(p), float(max(0.0, lo)), float(min(1.0, hi))


def narratives_yaml_hashes(root: Path) -> dict:
    path = root / "config" / "narratives.yaml"
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    # Normalize to CRLF: LF-only → CRLF; keep existing CRLF as-is
    crlf = lf.replace(b"\n", b"\r\n")
    out = {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256_raw": hashlib.sha256(raw).hexdigest(),
        "sha256_lf": hashlib.sha256(lf).hexdigest(),
        "sha256_crlf": hashlib.sha256(crlf).hexdigest(),
        "frozen_lf_prereg": FROZEN_NARR_LF,
        "lf_matches_frozen": hashlib.sha256(lf).hexdigest() == FROZEN_NARR_LF,
        "nbytes_raw": len(raw),
        "has_crlf_in_raw": b"\r\n" in raw,
    }
    return out


def ascii_ratio(text: str) -> float:
    return sum(1 for c in text if ord(c) < 128) / len(text) if text else 1.0


def make_rss_doc_id(link: str) -> str:
    h = hashlib.sha256((link or "").encode("utf-8")).hexdigest()[:20]
    return f"rss_{h}"


def residualize_dow(values: pd.Series, dates: pd.Series) -> pd.Series:
    """OLS residual of values on 6 DOW dummies (+ intercept). NaN stays NaN."""
    import statsmodels.api as sm

    y = values.astype(float).to_numpy()
    ok = np.isfinite(y)
    out = np.full(len(y), np.nan, dtype=float)
    if ok.sum() < 8:
        return pd.Series(out, index=values.index, name=values.name)
    dum = dow_dummy_frame(pd.to_datetime(dates))
    X = sm.add_constant(dum.loc[ok].to_numpy(dtype=float), has_constant="add")
    res = sm.OLS(y[ok], X).fit()
    out[ok] = res.resid
    return pd.Series(out, index=values.index, name=values.name)


def corr_with_block_boot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    n_boot: int = N_BOOT,
    block_len: int = BLOCK_LEN,
    seed: int = SEED,
) -> dict:
    """Pearson or Spearman + percentile 95% CI via circular moving-block bootstrap."""
    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = int(len(x))
    if n < 3:
        return {
            "method": method, "n": n, "corr": np.nan,
            "ci_lo": np.nan, "ci_hi": np.nan,
        }
    if method == "pearson":
        corr = float(pearsonr(x, y)[0])
    elif method == "spearman":
        corr = float(spearmanr(x, y)[0])
    else:
        raise ValueError(method)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        xb, yb = x[idx], y[idx]
        if method == "pearson":
            # constant series → nan
            if np.nanstd(xb) < 1e-15 or np.nanstd(yb) < 1e-15:
                boots[b] = np.nan
            else:
                boots[b] = float(pearsonr(xb, yb)[0])
        else:
            if np.nanstd(xb) < 1e-15 or np.nanstd(yb) < 1e-15:
                boots[b] = np.nan
            else:
                boots[b] = float(spearmanr(xb, yb)[0])
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        lo = hi = np.nan
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "method": method,
        "n": n,
        "corr": corr,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_boot_finite": int(len(finite)),
    }


def label_rates(labels: pd.Series) -> dict[str, tuple[int, int]]:
    """Return {pos/neg/neu: (count, n)}."""
    n = int(len(labels))
    return {
        "pos": (int((labels == "positive").sum()), n),
        "neg": (int((labels == "negative").sum()), n),
        "neu": (int((labels == "neutral").sum()), n),
    }


def rate_row(metric: str, k: int, n: int, main_rate: float) -> dict:
    p, lo, hi = wilson_ci(k, n)
    return {
        "metric": metric,
        "rss_k": k,
        "rss_n": n,
        "rss_rate": p,
        "rss_wilson_lo": lo,
        "rss_wilson_hi": hi,
        "main_rate": float(main_rate) if pd.notna(main_rate) else np.nan,
    }


# ---------------------------------------------------------------------------
# Part 1 — RSS holdout
# ---------------------------------------------------------------------------

def run_part1(root: Path, params: dict, paths: dict, out_dir: Path) -> dict:
    _seed(params.get("seed", SEED))
    hash_info = narratives_yaml_hashes(root)
    pd.DataFrame([hash_info]).to_csv(out_dir / "descript_rss_holdout_narratives_hash.csv", index=False)

    raw_dir = abs_path(root, paths.get("rss_holdout_dir", "data/raw/rss_holdout"))
    clean_path = abs_path(root, paths.get("rss_clean_path", "data/interim/rss_holdout.parquet"))

    # --- a/b pipeline ---
    errors: list[str] = []
    try:
        rss_df, clean_log = clean_rss.clean(raw_dir)
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        rss_df.to_parquet(clean_path, index=False)
        clean_path.with_suffix(".log.json").write_text(
            json.dumps(clean_log, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"clean_rss: {exc}")
        raise

    # time fields / resolution
    time_cols = [
        c for c in (
            "published_at", "collected_at_utc", "available_at",
            "news_date", "use_date", "lag_minutes", "published_missing",
        )
        if c in rss_df.columns
    ]
    time_resolution = {}
    for c in time_cols:
        s = rss_df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            # infer if date-only (normalize equal) or has time component
            if getattr(s.dt, "tz", None) is not None or (
                hasattr(s.dt, "hour") and ((s.dt.hour != 0) | (s.dt.minute != 0) | (s.dt.second != 0)).any()
            ):
                # news_date/use_date are date-normalized naive
                if c in ("news_date", "use_date"):
                    time_resolution[c] = "date_UTC"
                else:
                    time_resolution[c] = "datetime_UTC"
            else:
                time_resolution[c] = "date_UTC"
        else:
            time_resolution[c] = str(s.dtype)

    rss_df = rss_df.copy()
    rss_df["doc_id"] = rss_df["link"].map(make_rss_doc_id)
    assert rss_df["doc_id"].is_unique

    missing_title = (rss_df["title"].fillna("").astype(str).str.strip() == "").mean()
    # duplicates among cleaned: by url / title
    n_dup_url = int(rss_df["link"].map(clean_rss.normalize_url).duplicated().sum())
    n_dup_title = int(rss_df["title"].map(clean_rss.normalize_title).duplicated().sum())
    non_en = rss_df["nlp_text"].map(lambda t: ascii_ratio(t) < ASCII_MIN_RATIO)
    non_en_rate = float(non_en.mean())

    # score title
    cache_dir = abs_path(root, "data/interim") / ss.CACHE_DIR_NAME
    docs_score = rss_df[["doc_id", "title"]].copy()
    docs_score["title"] = docs_score["title"].fillna("").astype(str)
    slim_path = abs_path(root, "data/interim/rss_holdout_slim.parquet")
    docs_score.to_parquet(slim_path, index=False)

    sent_parts = []
    score_meta = {}
    for name in ("vader", "finbert"):
        part, meta = ss.run_model(
            name, docs_score, cache_dir, batch_size=32, device=None, text_col="title",
        )
        part = part[part["doc_id"].astype(str).isin(set(docs_score["doc_id"].astype(str)))]
        sent_parts.append(part)
        score_meta[name] = {k: v for k, v in meta.items() if k != "env"}
        if "env" in meta:
            score_meta[name]["env"] = meta["env"]
    sent = pd.concat(sent_parts, ignore_index=True)
    out_cols = ["doc_id", "model", "label", "p_pos", "p_neg", "p_neu", "text_variant"]
    sent_export = sent[out_cols].copy()
    z = sent_export["p_pos"] + sent_export["p_neg"] + sent_export["p_neu"]
    sent_export["p_pos"] = sent_export["p_pos"] / z
    sent_export["p_neg"] = sent_export["p_neg"] / z
    sent_export["p_neu"] = sent_export["p_neu"] / z
    sent_path = abs_path(root, "data/interim/rss_holdout_sentiment_doc.parquet")
    sent_export.to_parquet(sent_path, index=False)

    # narratives v4
    narr_cfg = load_narrative_config(root / "config" / "narratives.yaml")
    narr_flags = assign_doc_narratives(
        rss_df["title"], narr_cfg["narratives"], flags=narr_cfg["flags"],
    )
    narr_doc = pd.concat([rss_df[["doc_id"]].reset_index(drop=True), narr_flags.reset_index(drop=True)], axis=1)
    narr_path = abs_path(root, "data/interim/rss_holdout_narrative_doc.parquet")
    narr_doc.to_parquet(narr_path, index=False)

    # funnel table 1b
    funnel_rows = [
        {"step": "raw_jsonl_rows", "n_in": clean_log.get("rows_loaded"), "n_out": clean_log.get("rows_loaded"),
         "n_dropped": 0, "note": f"files={clean_log.get('files')}"},
        {"step": "drop_backfill", "n_in": clean_log.get("rows_loaded"),
         "n_out": clean_log.get("rows_loaded", 0) - clean_log.get("dropped_backfill", 0),
         "n_dropped": clean_log.get("dropped_backfill"), "note": ""},
        {"step": "off_topic_drop_decrypt", "n_in": None,
         "n_out": None, "n_dropped": clean_log.get("off_topic_dropped"),
         "note": f"flagged={clean_log.get('off_topic_flagged')}"},
        {"step": "dedup_url", "n_in": None, "n_out": None,
         "n_dropped": clean_log.get("dropped_dup_url"), "note": ""},
        {"step": "dedup_title", "n_in": None, "n_out": None,
         "n_dropped": clean_log.get("dropped_dup_title"), "note": ""},
        {"step": "clean_final", "n_in": None, "n_out": clean_log.get("rows_final"),
         "n_dropped": None, "note": json.dumps(clean_log.get("rows_by_source", {}))},
        {"step": "score_title", "n_in": len(docs_score),
         "n_out": int(sent_export["doc_id"].nunique()),
         "n_dropped": len(docs_score) - int(sent_export["doc_id"].nunique()),
         "note": json.dumps({m: score_meta[m].get("todo") for m in score_meta})},
        {"step": "match_narratives_v4", "n_in": len(narr_doc), "n_out": len(narr_doc),
         "n_dropped": 0, "note": f"version={narr_cfg.get('version')}"},
    ]
    # fill sequential n_in/n_out where missing using clean_log arithmetic
    n_after_backfill = clean_log["rows_loaded"] - clean_log["dropped_backfill"]
    n_after_off = n_after_backfill - clean_log["off_topic_dropped"]
    n_after_url = n_after_off - clean_log["dropped_dup_url"]
    n_after_title = n_after_url - clean_log["dropped_dup_title"]
    funnel_rows[2]["n_in"] = n_after_backfill
    funnel_rows[2]["n_out"] = n_after_off
    funnel_rows[3]["n_in"] = n_after_off
    funnel_rows[3]["n_out"] = n_after_url
    funnel_rows[4]["n_in"] = n_after_url
    funnel_rows[4]["n_out"] = n_after_title
    funnel_rows[5]["n_in"] = n_after_title

    sources = rss_df["source"].value_counts().to_dict()
    meta_rows = [
        {"key": "sources", "value": json.dumps(sources, ensure_ascii=False)},
        {"key": "time_fields", "value": json.dumps(time_cols)},
        {"key": "time_resolution", "value": json.dumps(time_resolution)},
        {"key": "pct_missing_title", "value": f"{100.0 * missing_title:.6f}"},
        {"key": "n_dup_url_remaining", "value": str(n_dup_url)},
        {"key": "n_dup_title_norm_remaining", "value": str(n_dup_title)},
        {"key": "non_en_rate_ascii", "value": f"{non_en_rate:.6f}"},
        {"key": "non_en_n", "value": str(int(non_en.sum()))},
        {"key": "errors", "value": json.dumps(errors)},
        {"key": "limit_note", "value": "4 days RSS holdout; insufficient to conclude on distribution"},
        {"key": "sha256_lf", "value": hash_info["sha256_lf"]},
        {"key": "sha256_crlf", "value": hash_info["sha256_crlf"]},
        {"key": "sha256_raw", "value": hash_info["sha256_raw"]},
        {"key": "lf_matches_frozen", "value": str(hash_info["lf_matches_frozen"])},
    ]
    funnel = pd.DataFrame(funnel_rows)
    funnel.to_csv(out_dir / "descript_rss_holdout_1b_funnel.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(out_dir / "descript_rss_holdout_1b_meta.csv", index=False)

    # --- c distribution vs main 2025H1 ---
    text = pd.read_parquet(
        abs_path(root, paths["text_clean_path"]),
        columns=["doc_id", "source", "news_date", "in_main_sample", "title", "non_en"],
    )
    main = text.loc[text["in_main_sample"]].copy()
    main["news_date"] = pd.to_datetime(main["news_date"]).dt.normalize()
    main_win = main[
        (main["news_date"] >= pd.Timestamp(MAIN_COMPARE_START))
        & (main["news_date"] <= pd.Timestamp(MAIN_COMPARE_END))
    ].copy()

    sent_main = pd.read_parquet(abs_path(root, paths["sentiment_doc_path"]))
    sent_main = sent_main[sent_main["doc_id"].isin(main_win["doc_id"])]
    narr_main = pd.read_parquet(abs_path(root, paths["narrative_doc_path"]))
    narr_main = narr_main[narr_main["doc_id"].isin(main_win["doc_id"])]

    dist_rows: list[dict] = []
    for model in ("finbert", "vader"):
        rss_lab = sent_export.loc[sent_export["model"] == model, "label"]
        main_lab = sent_main.loc[sent_main["model"] == model, "label"]
        rss_counts = label_rates(rss_lab)
        main_counts = label_rates(main_lab)
        for lab_key, lab_name in (("pos", "pos"), ("neg", "neg"), ("neu", "neu")):
            k, n = rss_counts[lab_key]
            mk, mn = main_counts[lab_key]
            dist_rows.append(rate_row(
                f"{model}_{lab_name}_rate", k, n, mk / mn if mn else np.nan,
            ))

    for name in CORE_NARRATIVES:
        k = int(narr_doc[name].sum())
        n = len(narr_doc)
        mk = int(narr_main[name].sum())
        mn = len(narr_main)
        dist_rows.append(rate_row(f"narr_{name}_match_rate", k, n, mk / mn if mn else np.nan))

    k = int(narr_doc["price_desc"].sum())
    n = len(narr_doc)
    mk = int(narr_main["price_desc"].sum())
    mn = len(narr_main)
    dist_rows.append(rate_row("price_desc_rate", k, n, mk / mn if mn else np.nan))

    # title length (mean chars) — not a rate; Wilson left blank
    rss_len = rss_df["title"].fillna("").astype(str).str.len()
    main_len = main_win["title"].fillna("").astype(str).str.len()
    dist_rows.append({
        "metric": "title_length_mean_chars",
        "rss_k": np.nan,
        "rss_n": int(len(rss_len)),
        "rss_rate": float(rss_len.mean()),
        "rss_wilson_lo": np.nan,
        "rss_wilson_hi": np.nan,
        "main_rate": float(main_len.mean()),
    })
    dist_rows.append({
        "metric": "limit_note",
        "rss_k": np.nan,
        "rss_n": np.nan,
        "rss_rate": np.nan,
        "rss_wilson_lo": np.nan,
        "rss_wilson_hi": np.nan,
        "main_rate": np.nan,
        "note": "4 days RSS holdout; insufficient to conclude on distribution",
    })

    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dir / "descript_rss_holdout_1c_vs_main_2025h1.csv", index=False)

    return {
        "hash_info": hash_info,
        "clean_log": clean_log,
        "n_clean": len(rss_df),
        "n_scored": int(sent_export["doc_id"].nunique()),
        "sources": sources,
        "dist_path": str(out_dir / "descript_rss_holdout_1c_vs_main_2025h1.csv"),
    }


# ---------------------------------------------------------------------------
# Part 2 — cryptonews vs main (text only)
# ---------------------------------------------------------------------------

def _ensure_cryptonews_sentiment(
    root: Path,
    cn_docs: pd.DataFrame,
    paths: dict,
) -> pd.DataFrame:
    """Return sentiment for cn+main docs. Score missing cryptonews via cache; do not
    mutate main sentiment_doc.parquet (side file only)."""
    sent_path = abs_path(root, paths["sentiment_doc_path"])
    sent_main = pd.read_parquet(sent_path)
    side = abs_path(root, "data/interim/sentiment_doc_cryptonews.parquet")
    out_cols = ["doc_id", "model", "label", "p_pos", "p_neg", "p_neu", "text_variant"]
    if side.exists():
        sent_cn_existing = pd.read_parquet(side)
        for c in out_cols:
            if c not in sent_cn_existing.columns:
                sent_cn_existing[c] = pd.NA
        sent_cn_existing = sent_cn_existing[out_cols]
    else:
        sent_cn_existing = pd.DataFrame(columns=out_cols)

    parts = [sent_main[out_cols]]
    if len(sent_cn_existing):
        parts.append(sent_cn_existing)
    pooled = pd.concat(parts, ignore_index=True)
    have = (
        pooled.loc[pooled["model"].isin(("vader", "finbert"))]
        .groupby("doc_id")["model"]
        .agg(lambda s: set(s))
    )
    need = set(cn_docs["doc_id"].astype(str))
    missing = [
        d for d in need
        if not {"vader", "finbert"}.issubset(have.get(d, set()))
    ]
    if missing:
        slim = cn_docs.loc[
            cn_docs["doc_id"].astype(str).isin(missing), ["doc_id", "title"]
        ].copy()
        slim["title"] = slim["title"].fillna("").astype(str)
        slim_path = abs_path(root, "data/interim/cryptonews_slim_for_score.parquet")
        slim.to_parquet(slim_path, index=False)

        cache_dir = abs_path(root, "data/interim") / ss.CACHE_DIR_NAME
        parts = []
        for name in ("vader", "finbert"):
            part, _meta = ss.run_model(
                name, slim, cache_dir, batch_size=32, device=None, text_col="title",
            )
            part = part[part["doc_id"].astype(str).isin(set(slim["doc_id"].astype(str)))]
            parts.append(
                part[["doc_id", "model", "label", "p_pos", "p_neg", "p_neu", "text_variant"]]
            )
        new_sent = pd.concat(parts, ignore_index=True)
        z = new_sent["p_pos"] + new_sent["p_neg"] + new_sent["p_neu"]
        new_sent["p_pos"] = new_sent["p_pos"] / z
        new_sent["p_neg"] = new_sent["p_neg"] / z
        new_sent["p_neu"] = new_sent["p_neu"] / z
        if len(sent_cn_existing):
            sent_cn_existing = pd.concat([sent_cn_existing, new_sent], ignore_index=True)
        else:
            sent_cn_existing = new_sent.copy()
        # de-dupe by doc_id×model
        sent_cn_existing = (
            sent_cn_existing
            .drop_duplicates(["doc_id", "model"], keep="last")
            .reset_index(drop=True)
        )
        side.parent.mkdir(parents=True, exist_ok=True)
        sent_cn_existing.to_parquet(side, index=False)
        pooled = pd.concat([sent_main[out_cols], sent_cn_existing[out_cols]], ignore_index=True)

    # Restrict to docs we need (cn ∪ main) to keep merges light
    keep_ids = set(cn_docs["doc_id"].astype(str)) | set(
        # main ids filled by caller via filter; include all main from sent_main
        sent_main["doc_id"].astype(str)
    )
    return pooled.loc[pooled["doc_id"].astype(str).isin(keep_ids)].copy()


def _cn_narratives(root: Path, cn_docs: pd.DataFrame) -> pd.DataFrame:
    narr_cfg = load_narrative_config(root / "config" / "narratives.yaml")
    flags = assign_doc_narratives(cn_docs["title"], narr_cfg["narratives"], flags=narr_cfg["flags"])
    out = pd.concat([cn_docs[["doc_id"]].reset_index(drop=True), flags.reset_index(drop=True)], axis=1)
    path = abs_path(root, "data/interim/cryptonews_narrative_doc.parquet")
    out.to_parquet(path, index=False)
    return out


def _daily_text_series(
    docs: pd.DataFrame,
    sent: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    """scope=all panel → columns date, N_t, AbnAttention, Sent_FinBERT, Sent_VADER."""
    panel = build_daily_panel(
        docs, sent, start=start, end=end, models=("vader", "finbert"), scopes=("all",),
    )
    g = panel[panel["scope"] == "all"].copy()
    g["date"] = pd.to_datetime(g["date"]).dt.normalize()
    return pd.DataFrame({
        "date": g["date"].values,
        "N_t": g["vader_NewsVolume_t"].astype(float).values,
        "AbnAttention": g["vader_AbnAttention_t"].astype(float).values,  # identical across models
        "Sent_FinBERT": g["finbert_Sent_t"].astype(float).values,
        "Sent_VADER": g["vader_Sent_t"].astype(float).values,
    })


def run_part2(root: Path, params: dict, paths: dict, out_dir: Path) -> dict:
    _seed(params.get("seed", SEED))
    text = pd.read_parquet(
        abs_path(root, paths["text_clean_path"]),
        columns=[
            "doc_id", "source", "news_date", "in_main_sample", "title",
            "currencies", "ts_bulk_flag",
        ],
    )
    text["news_date"] = pd.to_datetime(text["news_date"]).dt.normalize()

    cn = text.loc[text["source"].str.lower() == "cryptonews"].copy()
    cn = cn.loc[~cn["ts_bulk_flag"].astype(bool)].copy()
    cn = cn[
        (cn["news_date"] >= pd.Timestamp(CN_START))
        & (cn["news_date"] <= pd.Timestamp(CN_END))
    ].reset_index(drop=True)

    # main four sources (exclude cryptonews)
    main = text.loc[text["in_main_sample"]].copy()
    assert not (main["source"].str.lower() == "cryptonews").any()

    sent = _ensure_cryptonews_sentiment(root, cn, paths)
    narr_cn = _cn_narratives(root, cn)
    narr_main = pd.read_parquet(abs_path(root, paths["narrative_doc_path"]))

    # coverage of articles/day
    daily_n = cn.groupby("news_date")["doc_id"].nunique()
    cal = pd.date_range(CN_START, CN_END, freq="D")
    daily_n = daily_n.reindex(cal, fill_value=0)
    coverage = pd.DataFrame([{
        "n_cryptonews_articles": int(len(cn)),
        "n_calendar_days": int(len(cal)),
        "articles_per_day_median": float(daily_n.median()),
        "articles_per_day_mean": float(daily_n.mean()),
        "pct_days_lt3": float((daily_n < CN_MIN_N).mean()),
        "n_days_ge3": int((daily_n >= CN_MIN_N).sum()),
        "n_days_lt3": int((daily_n < CN_MIN_N).sum()),
    }])
    coverage.to_csv(out_dir / "descript_cryptonews_2f_coverage.csv", index=False)

    # build series versions
    cn_docs_raw = cn[["doc_id", "news_date", "source", "currencies"]].copy()
    main_docs_raw = main[["doc_id", "news_date", "source", "currencies"]].copy()

    cn_with_pd = cn_docs_raw.merge(narr_cn[["doc_id", "price_desc"]], on="doc_id", how="left")
    main_with_pd = main_docs_raw.merge(narr_main[["doc_id", "price_desc"]], on="doc_id", how="left")
    assert cn_with_pd["price_desc"].notna().all()
    assert main_with_pd["price_desc"].notna().all()

    series_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    # raw / base aggregation
    cn_s = _daily_text_series(cn_docs_raw, sent, CN_START, CN_END)
    main_s = _daily_text_series(main_docs_raw, sent, CN_START, CN_END)
    series_cache["raw"] = (cn_s, main_s)

    # noprice: drop price_desc before aggregation
    cn_np = cn_with_pd.loc[~cn_with_pd["price_desc"].astype(bool), ["doc_id", "news_date", "source", "currencies"]]
    main_np = main_with_pd.loc[~main_with_pd["price_desc"].astype(bool), ["doc_id", "news_date", "source", "currencies"]]
    series_cache["noprice"] = (
        _daily_text_series(cn_np, sent, CN_START, CN_END),
        _daily_text_series(main_np, sent, CN_START, CN_END),
    )
    # dow_resid uses raw series; residualize on analysis subsample below
    series_cache["dow_resid"] = series_cache["raw"]

    corr_rows: list[dict] = []
    overlap_rows: list[dict] = []

    for version in VERSIONS:
        cn_ser, main_ser = series_cache[version]
        # For noprice, N_t is noprice volume; still require >=3 of remaining articles.
        cn_ok = cn_ser.loc[cn_ser["N_t"] >= CN_MIN_N].copy()
        merged = cn_ok.merge(main_ser, on="date", suffixes=("_cn", "_main"))

        for period_id, p_start, p_end in PERIODS:
            sub = merged[
                (merged["date"] >= pd.Timestamp(p_start))
                & (merged["date"] <= pd.Timestamp(p_end))
            ].copy()
            if version == "dow_resid":
                for var in VARS:
                    sub[f"{var}_cn"] = residualize_dow(sub[f"{var}_cn"], sub["date"])
                    sub[f"{var}_main"] = residualize_dow(sub[f"{var}_main"], sub["date"])
            n_days_ge3 = int(len(sub))
            overlap_rows.append({
                "version": version,
                "period": period_id,
                "period_start": p_start,
                "period_end": p_end,
                "n_days_cn_ge3": n_days_ge3,
            })
            for var in VARS:
                x = sub[f"{var}_cn"].to_numpy(dtype=float)
                y = sub[f"{var}_main"].to_numpy(dtype=float)
                mask = np.isfinite(x) & np.isfinite(y)
                n_ov = int(mask.sum())
                for method in ("pearson", "spearman"):
                    res = corr_with_block_boot(
                        x[mask], y[mask], method=method,
                        n_boot=N_BOOT, block_len=BLOCK_LEN, seed=SEED,
                    )
                    corr_rows.append({
                        "version": version,
                        "period": period_id,
                        "period_start": p_start,
                        "period_end": p_end,
                        "variable": var,
                        "method": method,
                        "n_overlapping_days": res["n"],
                        "corr": res["corr"],
                        "ci_lo_95": res["ci_lo"],
                        "ci_hi_95": res["ci_hi"],
                        "block_len": BLOCK_LEN,
                        "n_boot": N_BOOT,
                        "seed": SEED,
                    })
                overlap_rows[-1][f"n_overlap_{var}"] = n_ov

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out_dir / "descript_cryptonews_2c_corr.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(out_dir / "descript_cryptonews_2e_overlap_n.csv", index=False)

    # wide-ish summary table 2f
    summary = corr_df.pivot_table(
        index=["version", "period", "variable"],
        columns="method",
        values=["corr", "ci_lo_95", "ci_hi_95", "n_overlapping_days"],
        aggfunc="first",
    )
    # flatten columns
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out_dir / "descript_cryptonews_2f_corr_summary.csv", index=False)

    return {
        "n_cn": len(cn),
        "coverage": coverage.to_dict(orient="records")[0],
        "n_corr_rows": len(corr_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Descriptive RSS holdout + cryptonews")
    parser.add_argument("--part", choices=["1", "2", "all"], default="all")
    args = parser.parse_args()

    root = project_root()
    params, paths = load_config(root)
    _seed(params.get("seed", SEED))
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.part in ("1", "all"):
        print("=== PART 1: RSS holdout ===")
        r1 = run_part1(root, params, paths, out_dir)
        print(json.dumps({k: r1[k] for k in ("hash_info", "n_clean", "n_scored", "sources")}, indent=2, default=str))

    if args.part in ("2", "all"):
        print("=== PART 2: cryptonews ===")
        r2 = run_part2(root, params, paths, out_dir)
        print(json.dumps(r2, indent=2, default=str))

    print("DONE — descriptive only; no test_registry append")


if __name__ == "__main__":
    main()

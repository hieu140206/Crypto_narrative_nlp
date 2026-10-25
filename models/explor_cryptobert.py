"""Exploratory CryptoBERT (PREREG §5) — family='explor_cryptobert'.

ElKulako/cryptobert: Bearish/Neutral/Bullish → negative/neutral/positive.
Doc-level Sent continuum = P(Bullish)−P(Bearish) stored as p_pos−p_neg.
Daily Sent_t via build_daily_panel._agg_model_frame (Pos−Neg labels).

Chỉ train+valid (≤ 2023-12-31). Không ALLOW_TEST / không đọc split test.
Holm riêng trong nhóm explor (không gộp họ xác nhận).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import score_sentiment as ss  # noqa: E402
from build_daily_panel import (  # noqa: E402
    abs_path,
    build_scope_panel,
    load_config,
    project_root,
)
from modeltable_io import load_model_table  # noqa: E402
from models.har_text import holm_adjust, run_asset_model  # noqa: E402
from models.rq1_rq2_sentiment import (  # noqa: E402
    BLOCK_LEN,
    CONTROL_A,
    N_BOOT,
    SEED,
    _ols_r2_adj,
    common_sample,
    moving_block_indices,
    nw_bandwidth,
    sent_coefs_1sd,
)
from models.rq3_narratives import append_registry, prepare_asset_frame  # noqa: E402

CRYPTOBERT_ID = "ElKulako/cryptobert"
# Pin to Hub commit resolved at first successful load (logged again at runtime).
CRYPTOBERT_REVISION = "9e37c910fe87727cb842a9ac55c6388256fe0f15"
CRYPTOBERT_CHECKPOINT = f"{CRYPTOBERT_ID}@{CRYPTOBERT_REVISION[:12]}"
CRYPTOBERT_MAX_LEN = 64  # tweets/headlines; model trained on short crypto text
LABEL_MAP_CBERT = {"Bullish": "positive", "Bearish": "negative", "Neutral": "neutral"}
HUMAN_TO_STD = {"positive": "positive", "negative": "negative", "neutral": "neutral",
                "pos": "positive", "neg": "negative", "neu": "neutral"}
FAMILY = "explor_cryptobert"
SUBSAMPLE_FRAC = 0.25
DATE_START = "2021-03-01"
DATE_END_EST = "2023-12-31"  # absolute estimation ceiling
OUTCOMES_A = ("y_ret", "y_log_rv", "y_abvol")
ASSETS = ("BTC", "ETH")
HORIZONS_B3 = (1, 3, 7)
N_BOOT_ACC = 5000


# ---------------------------------------------------------------------------
# CryptoBERT load / score (cache same pattern as score_sentiment.py)
# ---------------------------------------------------------------------------

def resolve_cryptobert_revision() -> str:
    """Return Hub commit hash; raise on failure (caller must STOP)."""
    try:
        from huggingface_hub import model_info
        info = model_info(CRYPTOBERT_ID, revision=CRYPTOBERT_REVISION)
        return str(info.sha)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cannot resolve CryptoBERT revision for %s@%s: %s"
            % (CRYPTOBERT_ID, CRYPTOBERT_REVISION, exc)
        ) from exc


def load_cryptobert(device: str | None = None) -> tuple[object, object, dict]:
    """Load tokenizer+model; STOP-style raise if download/load fails."""
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": "fp32",
        "model_id": CRYPTOBERT_ID,
        "revision": CRYPTOBERT_REVISION,
    }
    try:
        sha = resolve_cryptobert_revision()
        env["revision_sha"] = sha
        tokenizer = AutoTokenizer.from_pretrained(CRYPTOBERT_ID, revision=CRYPTOBERT_REVISION)
        model = AutoModelForSequenceClassification.from_pretrained(
            CRYPTOBERT_ID, revision=CRYPTOBERT_REVISION, torch_dtype=torch.float32
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "CryptoBERT failed to load (%s@%s): %s"
            % (CRYPTOBERT_ID, CRYPTOBERT_REVISION, exc)
        ) from exc
    model.to(device)
    model.eval()
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    env["id2label"] = id2label
    env["label_map"] = dict(LABEL_MAP_CBERT)
    return tokenizer, model, env


def score_cryptobert(
    texts: list[str],
    doc_ids: list[str],
    text_variant: str = "title",
    batch_size: int = 32,
    device: str | None = None,
    tokenizer=None,
    model=None,
    env: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    import torch

    if tokenizer is None or model is None:
        tokenizer, model, env = load_cryptobert(device=device)
    assert env is not None
    device = env["device"]
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    n_trunc = 0
    rows = []
    for i in range(0, len(texts), batch_size):
        batch_texts = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
        batch_ids = doc_ids[i:i + batch_size]
        enc = tokenizer(
            batch_texts, padding=True, truncation=True,
            max_length=CRYPTOBERT_MAX_LEN, return_tensors="pt",
        )
        lengths = tokenizer(
            batch_texts, truncation=False, padding=False, return_length=True
        )["length"]
        truncated_flags = [L > CRYPTOBERT_MAX_LEN for L in lengths]
        n_trunc += sum(truncated_flags)

        input_ids = enc["input_ids"].to(device)
        attention = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        for j, doc_id in enumerate(batch_ids):
            p = probs[j]
            raw = {id2label[k]: float(p[k]) for k in range(len(p))}
            p_pos = raw.get("Bullish", 0.0)
            p_neg = raw.get("Bearish", 0.0)
            p_neu = raw.get("Neutral", 0.0)
            # hard label from argmax on mapped classes
            label = max(
                [("positive", p_pos), ("negative", p_neg), ("neutral", p_neu)],
                key=lambda x: x[1],
            )[0]
            rows.append({
                "doc_id": doc_id,
                "model": "cryptobert",
                "checkpoint": CRYPTOBERT_CHECKPOINT,
                "text_variant": text_variant,
                "label": label,
                "p_pos": p_pos,
                "p_neg": p_neg,
                "p_neu": p_neu,
                "truncated": bool(truncated_flags[j]),
            })
        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(texts):
            print(f"  CryptoBERT {min(i + batch_size, len(texts))}/{len(texts)}")

    env = dict(env)
    env["truncated_new"] = n_trunc
    return pd.DataFrame(rows), env


def run_cryptobert_cached(
    docs: pd.DataFrame,
    cache_dir: Path,
    text_col: str = "title",
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    text_variant = text_col
    path = ss.cache_path(cache_dir, "cryptobert", CRYPTOBERT_CHECKPOINT, text_variant)
    cached = ss.load_cache(path)
    if len(cached) and "text_variant" in cached.columns:
        cached = cached[cached["text_variant"] == text_variant]
    done = set(cached["doc_id"].astype(str)) if len(cached) else set()
    todo = docs[~docs["doc_id"].astype(str).isin(done)].reset_index(drop=True)
    if len(todo):
        todo = ss.sort_by_text_length(todo, text_col)
    meta: dict = {
        "model": "cryptobert",
        "checkpoint": CRYPTOBERT_CHECKPOINT,
        "text_variant": text_variant,
        "cached": len(done),
        "todo": len(todo),
        "cache_path": str(path),
    }
    print(f"[cryptobert/{text_variant}] cache={len(done)} todo={len(todo)} -> {path.name}")
    if todo.empty:
        meta["env"] = {"revision": CRYPTOBERT_REVISION, "skipped_load": True}
        return cached, meta

    t0 = time.perf_counter()
    tokenizer, model, env = load_cryptobert(device=device)
    scored, env = score_cryptobert(
        todo[text_col].fillna("").astype(str).tolist(),
        todo["doc_id"].astype(str).tolist(),
        text_variant=text_variant,
        batch_size=batch_size,
        device=device,
        tokenizer=tokenizer,
        model=model,
        env=env,
    )
    elapsed = time.perf_counter() - t0
    meta["seconds"] = elapsed
    meta["per_doc_sec"] = elapsed / max(len(todo), 1)
    meta["env"] = env
    print(f"  scored {len(todo)} in {elapsed:.1f}s ({meta['per_doc_sec']*1000:.1f} ms/doc)")
    out = ss.merge_cache(cached, scored)
    ss.save_cache(out, path)
    return out, meta


# ---------------------------------------------------------------------------
# A) Label check
# ---------------------------------------------------------------------------

def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = N_BOOT_ACC,
    seed: int = SEED,
) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(y_true[idx] == y_pred[idx]))
    return acc, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan")
    # confusion
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            cm[idx[a], idx[b]] += 1.0
    po = float(np.trace(cm) / n)
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col) / (n * n))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float("nan")
    return float((po - pe) / (1.0 - pe))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    out = {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    for lab in labels:
        tp = float(np.sum((y_true == lab) & (y_pred == lab)))
        fp = float(np.sum((y_true != lab) & (y_pred == lab)))
        fn = float(np.sum((y_true == lab) & (y_pred != lab)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = float(f1)
    return out


def _normalize_human_label(s) -> str:
    return HUMAN_TO_STD.get(str(s).strip().lower(), str(s).strip().lower())


def eval_model_on_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    *,
    n_boot: int = N_BOOT_ACC,
    seed: int = SEED,
) -> dict:
    labels = ["negative", "neutral", "positive"]
    acc, lo, hi = bootstrap_accuracy_ci(y_true, y_pred, n_boot=n_boot, seed=seed)
    f1s = per_class_f1(y_true, y_pred, labels)
    return {
        "model": model_name,
        "n": int(len(y_true)),
        "accuracy": acc,
        "acc_ci_lo": lo,
        "acc_ci_hi": hi,
        "cohen_kappa": cohen_kappa(y_true, y_pred, labels),
        "f1_negative": f1s["negative"],
        "f1_neutral": f1s["neutral"],
        "f1_positive": f1s["positive"],
        "f1_macro": float(np.mean([f1s[l] for l in labels])),
        "label_map_note": "Bullish→pos, Bearish→neg, Neutral→neu (CryptoBERT); FinBERT/VADER native",
    }


def run_label_check(
    root: Path,
    *,
    batch_size: int = 32,
    device: str | None = None,
    n_boot: int = N_BOOT_ACC,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    random.seed(seed)
    np.random.seed(seed)
    lab_path = root / "labeling" / "label_final_630.csv"
    if not lab_path.exists():
        raise FileNotFoundError("Missing %s" % lab_path)
    lab = pd.read_csv(lab_path)
    need = {"document_id", "headline", "label_final"}
    if not need <= set(lab.columns):
        raise KeyError("label_final_630.csv missing %s" % (need - set(lab.columns)))
    lab = lab.copy()
    lab["doc_id"] = lab["document_id"].astype(str)
    lab["title"] = lab["headline"].fillna("").astype(str)
    lab["y_true"] = lab["label_final"].map(_normalize_human_label)
    assert len(lab) == 630, len(lab)

    cache_dir = abs_path(root, "data/interim") / ss.CACHE_DIR_NAME
    # Recompute FinBERT + VADER on the same 630 (not copy old metrics)
    docs = lab[["doc_id", "title"]].drop_duplicates("doc_id")
    vader = ss.score_vader(
        docs["title"].tolist(), docs["doc_id"].tolist(), text_variant="title"
    )
    fin, n_trunc, fin_env = ss.score_finbert(
        docs["title"].tolist(), docs["doc_id"].tolist(),
        text_variant="title", batch_size=batch_size, device=device,
    )
    cbert_all, cmeta = run_cryptobert_cached(
        docs, cache_dir, text_col="title", batch_size=batch_size, device=device,
    )
    cbert = cbert_all[cbert_all["doc_id"].astype(str).isin(set(docs["doc_id"]))].copy()

    rows = []
    for name, scored in (("vader", vader), ("finbert", fin), ("cryptobert", cbert)):
        m = lab.merge(scored[["doc_id", "label"]], on="doc_id", how="inner")
        assert len(m) == 630, (name, len(m))
        rows.append(eval_model_on_labels(
            m["y_true"].to_numpy(), m["label"].to_numpy(), name, n_boot=n_boot, seed=seed,
        ))
    out = pd.DataFrame(rows)
    meta = {
        "n": 630,
        "seed": seed,
        "n_boot": n_boot,
        "finbert_truncated": n_trunc,
        "finbert_env": fin_env,
        "cryptobert_meta": {k: cmeta.get(k) for k in ("checkpoint", "cached", "todo", "env")},
        "label_mapping": {
            "CryptoBERT": "Bullish→positive, Bearish→negative, Neutral→neutral",
            "human": "positive/negative/neutral (label_final)",
        },
    }
    return out, meta


# ---------------------------------------------------------------------------
# B) Subsample + daily Sent
# ---------------------------------------------------------------------------

def stratified_subsample_ids(
    docs: pd.DataFrame,
    *,
    frac: float = SUBSAMPLE_FRAC,
    seed: int = SEED,
) -> pd.DataFrame:
    """25% of rows, stratified by (source × month), seed 42."""
    d = docs.copy()
    d["news_date"] = pd.to_datetime(d["news_date"]).dt.tz_localize(None).dt.normalize()
    d["month"] = d["news_date"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    parts = []
    for (_, _), g in d.groupby(["source", "month"], sort=True):
        n = len(g)
        n_take = int(round(n * frac))
        if n_take <= 0:
            continue
        n_take = min(n_take, n)
        pick = rng.choice(g.index.to_numpy(), size=n_take, replace=False)
        parts.append(g.loc[pick])
    out = pd.concat(parts, ignore_index=False).sort_values("doc_id")
    return out[["doc_id"]].drop_duplicates().reset_index(drop=True)


def load_main_docs_window(root: Path) -> pd.DataFrame:
    params, paths = load_config(root)
    text_path = abs_path(root, paths["text_clean_path"])
    cols = ["doc_id", "title", "source", "news_date", "currencies", "in_main_sample"]
    df = pd.read_parquet(text_path, columns=cols)
    df = df.loc[df["in_main_sample"]].copy()
    df["news_date"] = pd.to_datetime(df["news_date"]).dt.tz_localize(None).dt.normalize()
    start = pd.Timestamp(params.get("sample", {}).get("start", DATE_START))
    # hard ceiling for explor estimation
    end = pd.Timestamp(DATE_END_EST)
    df = df[(df["news_date"] >= start) & (df["news_date"] <= end)]
    # exclude cryptonews if listed
    excl = set(params.get("exclude_sources_main") or [])
    if excl:
        df = df[~df["source"].isin(excl)]
    return df.reset_index(drop=True)


def build_subsample_sent_long(
    docs_sub: pd.DataFrame,
    sent_parts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Long sentiment for subsample docs; models in sent_parts."""
    frames = []
    ids = set(docs_sub["doc_id"].astype(str))
    for name, s in sent_parts.items():
        sub = s[s["doc_id"].astype(str).isin(ids)].copy()
        sub["model"] = name
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_explor_daily_panel(
    docs_sub: pd.DataFrame,
    sent_long: pd.DataFrame,
    *,
    start: str = DATE_START,
    end: str = DATE_END_EST,
    models: tuple[str, ...] = ("vader", "finbert", "cryptobert"),
) -> pd.DataFrame:
    calendar = pd.date_range(start, end, freq="D")
    scopes = []
    for scope in ("all", "BTC", "ETH"):
        scopes.append(build_scope_panel(
            docs_sub, sent_long, calendar, scope, models=models, abn_window=30,
        ))
    return pd.concat(scopes, ignore_index=True)


def daily_sent_series(panel: pd.DataFrame, scope: str = "all") -> pd.DataFrame:
    sub = panel.loc[panel["scope"] == scope].copy()
    cols = ["date"] + [c for c in sub.columns if c.endswith("_Sent_t")]
    return sub[cols].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# B1 correlations + coverage
# ---------------------------------------------------------------------------

def table_b1_correlations(daily: pd.DataFrame, vol_col: str = "finbert_NewsVolume_t") -> pd.DataFrame:
    cols = {
        "finbert": "finbert_Sent_t",
        "vader": "vader_Sent_t",
        "cryptobert": "cryptobert_Sent_t",
    }
    work = daily[["date", *cols.values()]].copy()
    if vol_col in daily.columns:
        work["_vol"] = daily[vol_col].values
    else:
        # any NewsVolume
        vcands = [c for c in daily.columns if c.endswith("_NewsVolume_t")]
        work["_vol"] = daily[vcands[0]].values if vcands else np.nan

    pairs = [("finbert", "vader"), ("finbert", "cryptobert"), ("vader", "cryptobert")]
    rows = []
    for a, b in pairs:
        ca, cb = cols[a], cols[b]
        m = work[[ca, cb]].dropna()
        if len(m) < 3:
            pear = spear = float("nan")
        else:
            pear = float(m[ca].corr(m[cb], method="pearson"))
            spear = float(m[ca].corr(m[cb], method="spearman"))
        rows.append({
            "pair": f"{a}_vs_{b}",
            "n_days_overlap": int(len(m)),
            "pearson": pear,
            "spearman": spear,
        })
    vol = work["_vol"].astype(float)
    coverage = {
        "pair": "coverage",
        "n_days_overlap": int(len(work)),
        "pearson": float(vol.median()) if vol.notna().any() else float("nan"),
        "spearman": float((vol < 3).mean()) if vol.notna().any() else float("nan"),
        "articles_per_day_median": float(vol.median()) if vol.notna().any() else float("nan"),
        "pct_days_lt3_articles": float((vol < 3).mean()) if vol.notna().any() else float("nan"),
    }
    out = pd.DataFrame(rows)
    # widen coverage into same table as extra row with named cols
    cov_row = {
        "pair": "coverage",
        "n_days_overlap": coverage["n_days_overlap"],
        "pearson": np.nan,
        "spearman": np.nan,
        "articles_per_day_median": coverage["articles_per_day_median"],
        "pct_days_lt3_articles": coverage["pct_days_lt3_articles"],
    }
    out = pd.concat([out, pd.DataFrame([cov_row])], ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# B2 Spec A — ΔR²adj vs market baseline
# ---------------------------------------------------------------------------

def bootstrap_delta_r2_vs_baseline(
    y: np.ndarray,
    X_ctrl: np.ndarray,
    x_sent: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    block_len: int = BLOCK_LEN,
    seed: int = SEED,
) -> dict:
    """ΔR²_adj = R²_adj(ctrl+sent) − R²_adj(ctrl); MBB CI + two-sided p."""
    y = np.asarray(y, dtype=float)
    X_ctrl = np.asarray(X_ctrl, dtype=float)
    x_sent = np.asarray(x_sent, dtype=float).reshape(-1, 1)
    n = len(y)
    r2_full = _ols_r2_adj(y, np.hstack([X_ctrl, x_sent]))
    r2_base = _ols_r2_adj(y, X_ctrl)
    delta = r2_full - r2_base

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        yb = y[idx]
        Cb = X_ctrl[idx]
        boots[b] = (
            _ols_r2_adj(yb, np.hstack([Cb, x_sent[idx]]))
            - _ols_r2_adj(yb, Cb)
        )
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    p_le = float(np.mean(boots <= 0.0))
    p_ge = float(np.mean(boots >= 0.0))
    p_raw0 = float(min(1.0, 2.0 * min(p_le, p_ge)))
    p_floor = 1.0 / (float(n_boot) + 1.0)
    p_boot = float(max(p_raw0, p_floor))
    p_boot_label = ("<%d/%d" % (1, n_boot)) if p_raw0 <= 0.0 else ("%.6g" % p_boot)
    return {
        "r2_adj_full": r2_full,
        "r2_adj_baseline": r2_base,
        "delta_r2_adj": float(delta),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_boot": p_boot,
        "p_boot_raw": p_raw0,
        "p_boot_label": p_boot_label,
        "n_boot": int(n_boot),
        "block_len": int(block_len),
    }


def attach_subsample_sent(
    model_df: pd.DataFrame,
    panel_all: pd.DataFrame,
    *,
    prefix_map: dict[str, str],
) -> pd.DataFrame:
    """Join scope=all Sent_t columns onto model table (by date); rename via prefix_map.

    prefix_map: panel col stem model name → output col name
      e.g. {'finbert': 'sub_finbert_Sent_t', ...}
    """
    daily = panel_all.loc[panel_all["scope"] == "all"].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None).dt.normalize()
    keep = ["date"]
    ren = {}
    for model_name, out_col in prefix_map.items():
        src = f"{model_name}_Sent_t"
        if src not in daily.columns:
            raise KeyError(src)
        keep.append(src)
        ren[src] = out_col
    slim = daily[keep].rename(columns=ren)
    out = model_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    # drop existing target cols if present to avoid _x/_y
    drop_cols = [c for c in ren.values() if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out.merge(slim, on="date", how="left")


def run_b2_spec_a(
    df_a: pd.DataFrame,
    sent_cols: dict[str, str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    family: str = FAMILY,
) -> pd.DataFrame:
    """For each asset × outcome × model: ΔR²adj vs market baseline (CONTROL_A)."""
    rows = []
    for asset in ASSETS:
        df_asset = prepare_asset_frame(df_a, asset)
        # hard date ceiling
        df_asset = df_asset[df_asset["date"] <= pd.Timestamp(DATE_END_EST)]
        for y_col in OUTCOMES_A:
            for model_name, sent_col in sent_cols.items():
                work = common_sample(df_asset, y_col, [sent_col], CONTROL_A)
                if len(work) < len(CONTROL_A) + 5:
                    rows.append({
                        "asset": asset, "y": y_col, "model": model_name,
                        "sent_col": sent_col, "n": int(len(work)),
                        "family": family, "spec": "A",
                        "delta_r2_adj": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                        "p_boot": np.nan, "note": "insufficient_n",
                    })
                    continue
                boot = bootstrap_delta_r2_vs_baseline(
                    work[y_col].to_numpy(),
                    work[list(CONTROL_A)].to_numpy(dtype=float),
                    work[sent_col].to_numpy(dtype=float),
                    n_boot=n_boot, seed=seed,
                )
                bw = nw_bandwidth(None)
                coef, se = sent_coefs_1sd(work, y_col, sent_col, CONTROL_A, bw)
                rows.append({
                    "asset": asset,
                    "y": y_col,
                    "model": model_name,
                    "sent_col": sent_col,
                    "family": family,
                    "spec": "A",
                    "n": int(len(work)),
                    "coef_1sd": coef,
                    "se_nw": se,
                    "nw_bandwidth": bw,
                    **boot,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B3 Family-5-like (valid 2023 only OOS; train≤2022)
# ---------------------------------------------------------------------------

def run_b3_har_text(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    text_col: str = "all_cryptobert_Sent_t",
    model_name: str = "M_cbert",
) -> pd.DataFrame:
    summaries = []
    for h in HORIZONS_B3:
        for asset in ASSETS:
            # date ceiling on both
            tr = train[pd.to_datetime(train["date"]) <= pd.Timestamp(DATE_END_EST)]
            va = valid[pd.to_datetime(valid["date"]) <= pd.Timestamp(DATE_END_EST)]
            # valid window is 2023 only by model_table split
            oos, summary = run_asset_model(tr, va, asset, h, model_name, text_col)
            summaries.append(summary)
            _ = oos  # not persisted as confirmatory
    out = pd.DataFrame(summaries)
    # Holm within explor group of 6 (2 assets × 3 h) — descriptive only
    out = out.sort_values(["asset", "h"]).reset_index(drop=True)
    out["cw_pvalue_holm_explor"] = holm_adjust(out["cw_pvalue_raw"].astype(float).tolist())
    out["family"] = FAMILY
    out["holm_note"] = "Holm within explor 6 tests only; not shared with confirmatory"
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all(
    root: Path | None = None,
    *,
    batch_size: int = 32,
    device: str | None = None,
    n_boot: int = N_BOOT,
    n_boot_acc: int = N_BOOT_ACC,
    seed: int = SEED,
    skip_label: bool = False,
    skip_b3: bool = False,
) -> dict:
    root = root or project_root()
    random.seed(seed)
    np.random.seed(seed)
    out_dir = root / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = abs_path(root, "data/interim") / ss.CACHE_DIR_NAME
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log: dict = {
        "timestamp": ts,
        "seed": seed,
        "family": FAMILY,
        "cryptobert_id": CRYPTOBERT_ID,
        "cryptobert_revision": CRYPTOBERT_REVISION,
        "date_end_est": DATE_END_EST,
    }

    # --- A ---
    if not skip_label:
        print("=== A) Label check (630) ===")
        tab_a, meta_a = run_label_check(
            root, batch_size=batch_size, device=device, n_boot=n_boot_acc, seed=seed,
        )
        tab_a.to_csv(out_dir / "explor_cryptobert_A_label_check.csv", index=False)
        log["A"] = meta_a
        if meta_a.get("cryptobert_meta", {}).get("env"):
            env = meta_a["cryptobert_meta"]["env"]
            if isinstance(env, dict):
                log["cryptobert_revision_sha"] = env.get("revision_sha", CRYPTOBERT_REVISION)
        print(tab_a.to_string(index=False))
    else:
        tab_a = pd.DataFrame()

    # --- B subsample ---
    print("=== B) Stratified subsample ===")
    docs = load_main_docs_window(root)
    ids_df = stratified_subsample_ids(docs, frac=SUBSAMPLE_FRAC, seed=seed)
    ids_path = out_dir / "cryptobert_subsample_ids.csv"
    ids_df.to_csv(ids_path, index=False)
    log["B_n_pool"] = int(len(docs))
    log["B_n_subsample"] = int(len(ids_df))
    print(f"pool={len(docs)} subsample={len(ids_df)} -> {ids_path}")

    docs_sub = docs[docs["doc_id"].astype(str).isin(set(ids_df["doc_id"].astype(str)))].copy()
    # Score CryptoBERT on subsample
    cbert_all, cmeta = run_cryptobert_cached(
        docs_sub[["doc_id", "title"]], cache_dir,
        text_col="title", batch_size=batch_size, device=device,
    )
    log["B_cryptobert_score"] = {
        k: cmeta.get(k) for k in ("cached", "todo", "seconds", "per_doc_sec", "checkpoint", "env")
    }
    if isinstance(cmeta.get("env"), dict):
        log["cryptobert_revision_sha"] = cmeta["env"].get("revision_sha", CRYPTOBERT_REVISION)

    # Existing FinBERT/VADER from sentiment_doc (title), filtered to subsample
    params, paths = load_config(root)
    sent_path = abs_path(root, paths["sentiment_doc_path"])
    sent = pd.read_parquet(sent_path)
    assert (sent["text_variant"] == "title").all()
    id_set = set(ids_df["doc_id"].astype(str))
    fin = sent[(sent["model"] == "finbert") & sent["doc_id"].astype(str).isin(id_set)].copy()
    vad = sent[(sent["model"] == "vader") & sent["doc_id"].astype(str).isin(id_set)].copy()
    cbert = cbert_all[cbert_all["doc_id"].astype(str).isin(id_set)].copy()
    # ensure probs sum
    for df_s in (fin, vad, cbert):
        z = df_s["p_pos"] + df_s["p_neg"] + df_s["p_neu"]
        df_s["p_pos"] = df_s["p_pos"] / z
        df_s["p_neg"] = df_s["p_neg"] / z
        df_s["p_neu"] = df_s["p_neu"] / z

    sent_long = build_subsample_sent_long(docs_sub, {
        "vader": vad, "finbert": fin, "cryptobert": cbert,
    })
    # keep columns expected by build_scope_panel
    need_cols = ["doc_id", "model", "label", "p_pos", "p_neg", "p_neu"]
    sent_long = sent_long[need_cols].copy()

    panel = build_explor_daily_panel(docs_sub, sent_long)
    panel_path = out_dir / "explor_cryptobert_daily_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    log["B_panel_rows"] = int(len(panel))

    # B1
    daily_all = panel.loc[panel["scope"] == "all"].copy()
    tab_b1 = table_b1_correlations(daily_all)
    tab_b1.to_csv(out_dir / "explor_cryptobert_B1_corr.csv", index=False)
    print(tab_b1.to_string(index=False))

    # Attach subsample Sent onto model_table A (train+valid only)
    train_a = load_model_table("A", "train", root=root, panel="title")
    valid_a = load_model_table("A", "valid", root=root, panel="title")
    raw_a = pd.concat([train_a, valid_a], ignore_index=True)
    if (raw_a["split"].astype(str) == "test").any():
        raise RuntimeError("test rows leaked into explor A")
    raw_a = raw_a[pd.to_datetime(raw_a["date"]) <= pd.Timestamp(DATE_END_EST)]

    sent_map = {
        "finbert": "sub_finbert_Sent_t",
        "vader": "sub_vader_Sent_t",
        "cryptobert": "sub_cryptobert_Sent_t",
    }
    raw_a = attach_subsample_sent(raw_a, panel, prefix_map=sent_map)

    # B2: three models on subsample Sent
    print("=== B2) Spec A ΔR²adj vs market baseline ===")
    tab_b2 = run_b2_spec_a(
        raw_a,
        {
            "finbert_sub": "sub_finbert_Sent_t",
            "vader_sub": "sub_vader_Sent_t",
            "cryptobert_sub": "sub_cryptobert_Sent_t",
        },
        n_boot=n_boot, seed=seed,
    )
    # Also FinBERT/VADER full-sample columns already on model_table (power loss contrast)
    # — but restricted to same dates/assets; uses confirmatory all_* Sent on same N days
    tab_b2_full = run_b2_spec_a(
        raw_a,
        {
            "finbert_full": "all_finbert_Sent_t",
            "vader_full": "all_vader_Sent_t",
        },
        n_boot=n_boot, seed=seed,
    )
    tab_b2_full["sample"] = "full_sent_on_trainvalid"
    tab_b2["sample"] = "subsample_25pct"
    tab_b2_all = pd.concat([tab_b2, tab_b2_full], ignore_index=True)
    tab_b2_all.to_csv(out_dir / "explor_cryptobert_B2_specA.csv", index=False)
    print(tab_b2_all.to_string(index=False))

    # B3
    if not skip_b3:
        print("=== B3) HAR+text M_cbert on valid 2023 ===")
        # join cryptobert Sent as all_cryptobert_Sent_t onto B tables in memory
        panel_all = panel.loc[panel["scope"] == "all", ["date", "cryptobert_Sent_t"]].copy()
        panel_all["date"] = pd.to_datetime(panel_all["date"]).dt.tz_localize(None).dt.normalize()
        panel_all = panel_all.rename(columns={"cryptobert_Sent_t": "all_cryptobert_Sent_t"})

        summaries_b3 = []
        for h in HORIZONS_B3:
            tr = load_model_table("B", "train", root=root, panel="title", horizon=h, drop_embargo=True)
            va = load_model_table("B", "valid", root=root, panel="title", horizon=h, drop_embargo=True)
            if (tr["split"].astype(str) == "test").any() or (va["split"].astype(str) == "test").any():
                raise RuntimeError("test leak B3")
            tr = tr[pd.to_datetime(tr["date"]) <= pd.Timestamp(DATE_END_EST)].copy()
            va = va[pd.to_datetime(va["date"]) <= pd.Timestamp(DATE_END_EST)].copy()
            for df_ in (tr, va):
                df_["date"] = pd.to_datetime(df_["date"]).dt.tz_localize(None).dt.normalize()
            if "all_cryptobert_Sent_t" in tr.columns:
                tr = tr.drop(columns=["all_cryptobert_Sent_t"])
            if "all_cryptobert_Sent_t" in va.columns:
                va = va.drop(columns=["all_cryptobert_Sent_t"])
            tr = tr.merge(panel_all, on="date", how="left")
            va = va.merge(panel_all, on="date", how="left")
            for asset in ASSETS:
                oos, summary = run_asset_model(
                    tr, va, asset, h, "M_cbert", "all_cryptobert_Sent_t",
                )
                summaries_b3.append(summary)
                oos.assign(model="M_cbert", asset=asset, h=h).to_csv(
                    out_dir / f"explor_cryptobert_B3_oos_{asset}_h{h}.csv", index=False,
                )
        tab_b3 = pd.DataFrame(summaries_b3).sort_values(["asset", "h"]).reset_index(drop=True)
        tab_b3["cw_pvalue_holm_explor"] = holm_adjust(
            tab_b3["cw_pvalue_raw"].astype(float).tolist()
        )
        tab_b3["family"] = FAMILY
        tab_b3["holm_note"] = "Holm within explor 6 tests only; not shared with confirmatory"
        tab_b3.to_csv(out_dir / "explor_cryptobert_B3_har_text.csv", index=False)
        print(tab_b3.to_string(index=False))
    else:
        tab_b3 = pd.DataFrame()

    # Registry append-only
    reg_path = out_dir / "test_registry.csv"
    n_before = len(pd.read_csv(reg_path)) if reg_path.exists() else 0
    reg_rows: list[dict] = []
    # A: one row per model accuracy p not applicable — skip or note
    for _, r in tab_b2.iterrows():
        reg_rows.append({
            "timestamp": ts,
            "script": "models/explor_cryptobert.py",
            "model": f"B2|{r['model']}|{r['y']}",
            "asset": r["asset"],
            "h": "",
            "split": "train+valid",
            "n_oos": r.get("n", ""),
            "p_raw": r.get("p_boot", float("nan")),
            "baseline": "market_ctrl_A",
            "family": FAMILY,
            "scale": "1sd",
            "note": "explor ΔR²adj vs market; subsample Sent",
        })
    if len(tab_b3):
        for _, r in tab_b3.iterrows():
            reg_rows.append({
                "timestamp": ts,
                "script": "models/explor_cryptobert.py",
                "model": "M_cbert",
                "asset": r["asset"],
                "h": r["h"],
                "split": "valid",
                "n_oos": r.get("n_oos", ""),
                "p_raw": r.get("cw_pvalue_raw", float("nan")),
                "baseline": "har_dow",
                "family": FAMILY,
                "scale": "",
                "note": "explor family5-like; Holm explor-only",
            })
    append_registry(reg_rows, reg_path)
    n_after = len(pd.read_csv(reg_path))
    log["registry_before"] = n_before
    log["registry_added"] = len(reg_rows)
    log["registry_after"] = n_after

    log_path = out_dir / "explor_cryptobert.meta.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Registry +{len(reg_rows)} -> {n_after} total")
    print("Meta ->", log_path)
    return {
        "A": tab_a,
        "B1": tab_b1,
        "B2": tab_b2_all,
        "B3": tab_b3,
        "log": log,
        "registry_added": len(reg_rows),
        "registry_total": n_after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory CryptoBERT (PREREG §5)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-boot-acc", type=int, default=N_BOOT_ACC)
    parser.add_argument("--skip-label", action="store_true")
    parser.add_argument("--skip-b3", action="store_true")
    args = parser.parse_args()
    try:
        resolve_cryptobert_revision()
    except RuntimeError as exc:
        print("STOP:", exc)
        sys.exit(2)
    run_all(
        batch_size=args.batch_size,
        device=args.device,
        n_boot=args.n_boot,
        n_boot_acc=args.n_boot_acc,
        skip_label=args.skip_label,
        skip_b3=args.skip_b3,
    )


if __name__ == "__main__":
    main()

"""RQ1 (họ 1) + RQ2 (họ 2) — PREREG Spec A/B, chỉ train+valid.

Import HAC / Holm / registry / z-score / prepare từ models.rq3_narratives (không sao chép).
Không đọc split='test'; luôn qua load_model_table.

Họ 1 (Spec A): FinBERT Sent vs VADER Sent — ΔR²_adj + moving-block bootstrap.
Họ 2: PosSent+NegSent — Wald H0 β_pos+β_neg=0 (A: ret_t; B: y_ret_h, h=1,3,7).

BTC = xác nhận (Holm riêng từng họ); ETH + biến thể khác = độ bền (Holm riêng ETH).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.rq3_narratives import (  # noqa: E402
    CONTROL_FRED,
    NW_BANDWIDTH,
    SEED,
    append_registry,
    fit_ols_nw,
    holm_adjust,
    prepare_asset_frame,
    project_root,
    zscore_series,
)

# Spec A controls (trễ) + FRED + DOW
CONTROL_A_MARKET: tuple[str, ...] = (
    "ret_lag1", "ret_lag2", "har_d", "har_w", "har_m", "logvol_lag1",
)
CONTROL_A: tuple[str, ...] = CONTROL_A_MARKET + CONTROL_FRED + DOW_DUMMY_COLS

# Spec B controls (cùng ngày ret/logvol + HAR B) + FRED + DOW
CONTROL_B_MARKET: tuple[str, ...] = (
    "ret_t", "logvol_t", "har_d", "har_w", "har_m",
)
CONTROL_B: tuple[str, ...] = CONTROL_B_MARKET + CONTROL_FRED + DOW_DUMMY_COLS

HORIZONS_B: tuple[int, ...] = (1, 3, 7)
BLOCK_LEN = 10
N_BOOT = 5000

COL_FIN_SENT = "all_finbert_Sent_t"
COL_VAD_SENT = "all_vader_Sent_t"
COL_FIN_MEAN = "all_finbert_MeanProb_t"
COL_VAD_MEAN = "all_vader_MeanProb_t"
COL_POS = "all_finbert_PosSent"
COL_NEG = "all_finbert_NegSent"
COL_NVOL = "all_finbert_NewsVolume_t"
OUTCOMES_A: tuple[str, ...] = ("y_ret", "y_log_rv", "y_abvol")
SENT_VS_MARKET: dict[str, str] = {
    "finbert": COL_FIN_SENT,
    "vader": COL_VAD_SENT,
}


def load_params(root: Path | None = None) -> dict:
    import yaml
    root = root or project_root()
    return yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}


def load_train_valid(
    spec: str,
    *,
    root: Path | None = None,
    panel: str = "title",
    horizon: int | None = None,
    drop_embargo: bool = False,
    table_path: Path | str | None = None,
    exclude_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Chỉ train+valid; Spec B có thể drop embargo_drop_h.

    table_path: parquet LOSO / override. exclude_sources chỉ metadata
    (lọc nguồn đã làm ở panel); không đọc split=test.
    """
    del exclude_sources  # lọc ở panel; giữ chữ ký cho LOSO
    root = root or project_root()
    train = load_model_table(
        spec, "train", root=root, panel=panel,
        horizon=horizon, drop_embargo=drop_embargo,
        table_path=table_path,
    )
    valid = load_model_table(
        spec, "valid", root=root, panel=panel,
        horizon=horizon, drop_embargo=drop_embargo,
        table_path=table_path,
    )
    out = pd.concat([train, valid], ignore_index=True)
    if "split" in out.columns and (out["split"].astype(str) == "test").any():
        raise RuntimeError("Mẫu RQ1/RQ2 không được chứa dòng test")
    return out


def nw_bandwidth(h: int | None = None) -> int:
    if h is None:
        return max(NW_BANDWIDTH, 5)
    return max(int(h) + 4, 5)


def _ols_r2_adj(y: np.ndarray, X: np.ndarray) -> float:
    Xc = sm.add_constant(np.asarray(X, dtype=float), has_constant="add")
    res = sm.OLS(np.asarray(y, dtype=float), Xc).fit()
    return float(res.rsquared_adj)


def moving_block_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap indices, length n."""
    if n <= 0:
        return np.array([], dtype=int)
    block_len = max(1, min(int(block_len), n))
    out: list[int] = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        for j in range(block_len):
            out.append((start + j) % n)
            if len(out) >= n:
                break
    return np.asarray(out[:n], dtype=int)


def bootstrap_delta_r2_adj(
    y: np.ndarray,
    X_ctrl: np.ndarray,
    x_f: np.ndarray,
    x_v: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    block_len: int = BLOCK_LEN,
    seed: int = SEED,
) -> dict:
    """ΔR²_adj = R²_adj(F)−R²_adj(V); MBB; CI percentile 95%; p hai phía."""
    y = np.asarray(y, dtype=float)
    X_ctrl = np.asarray(X_ctrl, dtype=float)
    x_f = np.asarray(x_f, dtype=float).reshape(-1, 1)
    x_v = np.asarray(x_v, dtype=float).reshape(-1, 1)
    n = len(y)
    r2_f = _ols_r2_adj(y, np.hstack([X_ctrl, x_f]))
    r2_v = _ols_r2_adj(y, np.hstack([X_ctrl, x_v]))
    delta = r2_f - r2_v

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = moving_block_indices(n, block_len, rng)
        yb = y[idx]
        Cb = X_ctrl[idx]
        boots[b] = (
            _ols_r2_adj(yb, np.hstack([Cb, x_f[idx]]))
            - _ols_r2_adj(yb, np.hstack([Cb, x_v[idx]]))
        )
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))
    # p hai phía: 2 * min(P*(Δ*≤0), P*(Δ*≥0))
    p_le = float(np.mean(boots <= 0.0))
    p_ge = float(np.mean(boots >= 0.0))
    p_raw0 = float(min(1.0, 2.0 * min(p_le, p_ge)))
    # Không báo p=0: sàn 1/(B+1); nhãn "<1/B" khi chạm sàn từ 0
    p_floor = 1.0 / (float(n_boot) + 1.0)
    p_boot = float(max(p_raw0, p_floor))
    p_boot_label = ("<%d/%d" % (1, n_boot)) if p_raw0 <= 0.0 else ("%.6g" % p_boot)
    return {
        "r2_adj_F": r2_f,
        "r2_adj_V": r2_v,
        "delta_r2_adj": float(delta),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_boot": p_boot,
        "p_boot_raw": p_raw0,
        "p_boot_label": p_boot_label,
        "n_boot": int(n_boot),
        "block_len": int(block_len),
        "scale": "1sd",
    }


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
        "scale": "1sd",
    }


def common_sample(
    df: pd.DataFrame,
    y_col: str,
    extra_cols: list[str],
    control_cols: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Bỏ NaN ở mọi biến của CẢ HAI mô hình đang so sánh (+ y + controls)."""
    cols = ["date", y_col, *control_cols, *extra_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError("Thiếu cột: %s" % missing)
    out = df[cols].dropna().sort_values("date").reset_index(drop=True)
    return out


def sent_coefs_1sd(
    work: pd.DataFrame,
    y_col: str,
    sent_col: str,
    control_cols: list[str] | tuple[str, ...],
    bandwidth: int,
) -> tuple[float, float]:
    z, _, _ = zscore_series(work[sent_col])
    X = work[list(control_cols)].copy()
    X["z_sent"] = z.to_numpy()
    res = fit_ols_nw(work[y_col], X, bandwidth=bandwidth)
    return float(res.params["z_sent"]), float(res.bse["z_sent"])


def run_rq1_hypothesis(
    df_asset: pd.DataFrame,
    *,
    asset: str,
    y_col: str,
    hyp: str,
    sent_f: str = COL_FIN_SENT,
    sent_v: str = COL_VAD_SENT,
    control_cols: tuple[str, ...] = CONTROL_A,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    family: str = "rq1_main",
    label: str = "main",
) -> dict:
    work = common_sample(df_asset, y_col, [sent_f, sent_v], control_cols)
    ctrl = list(control_cols)
    boot = bootstrap_delta_r2_adj(
        work[y_col].to_numpy(),
        work[ctrl].to_numpy(dtype=float),
        work[sent_f].to_numpy(dtype=float),
        work[sent_v].to_numpy(dtype=float),
        n_boot=n_boot,
        seed=seed,
    )
    bw = nw_bandwidth(None)
    coef_f, se_f = sent_coefs_1sd(work, y_col, sent_f, ctrl, bw)
    coef_v, se_v = sent_coefs_1sd(work, y_col, sent_v, ctrl, bw)
    return {
        "asset": asset,
        "family": family,
        "label": label,
        "hypothesis": hyp,
        "spec": "A",
        "h": "",
        "y": y_col,
        "n": int(len(work)),
        "sent_F": sent_f,
        "sent_V": sent_v,
        "coef_F_1sd": coef_f,
        "se_F_nw": se_f,
        "coef_V_1sd": coef_v,
        "se_V_nw": se_v,
        "nw_bandwidth": bw,
        **boot,
    }


def run_rq1_for_asset(
    df_a: pd.DataFrame,
    asset: str,
    *,
    family: str = "rq1_main",
    label: str = "main",
    sent_f: str = COL_FIN_SENT,
    sent_v: str = COL_VAD_SENT,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> pd.DataFrame:
    rows = [
        run_rq1_hypothesis(
            df_a, asset=asset, y_col="y_ret", hyp="H1a",
            sent_f=sent_f, sent_v=sent_v, family=family, label=label,
            n_boot=n_boot, seed=seed,
        ),
        run_rq1_hypothesis(
            df_a, asset=asset, y_col="y_log_rv", hyp="H1b",
            sent_f=sent_f, sent_v=sent_v, family=family, label=label,
            n_boot=n_boot, seed=seed,
        ),
    ]
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_boot"].astype(float).tolist())
    out["confirmed"] = out["p_holm"] < 0.05
    return out


def run_rq1_vs_market(
    df_a: pd.DataFrame,
    asset: str,
    *,
    family: str = "robust_loso",
    label: str = "loso",
    sent_cols: dict[str, str] | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    holm: bool = True,
) -> pd.DataFrame:
    """ΔR²_adj (market+Sent vs market) — FinBERT và VADER; y ∈ ret, log_rv, abvol."""
    sent_cols = sent_cols or dict(SENT_VS_MARKET)
    rows = []
    for y_col in OUTCOMES_A:
        for model_name, sent_col in sent_cols.items():
            work = common_sample(df_a, y_col, [sent_col], CONTROL_A)
            boot = bootstrap_delta_r2_vs_baseline(
                work[y_col].to_numpy(),
                work[list(CONTROL_A)].to_numpy(dtype=float),
                work[sent_col].to_numpy(dtype=float),
                n_boot=n_boot,
                seed=seed,
            )
            bw = nw_bandwidth(None)
            coef, se = sent_coefs_1sd(work, y_col, sent_col, CONTROL_A, bw)
            rows.append({
                "asset": asset,
                "family": family,
                "label": label,
                "hypothesis": f"A_{model_name}_{y_col}",
                "spec": "A",
                "h": "",
                "y": y_col,
                "model": model_name,
                "sent_col": sent_col,
                "n": int(len(work)),
                "coef_1sd": coef,
                "se_nw": se,
                "nw_bandwidth": bw,
                **boot,
            })
    out = pd.DataFrame(rows)
    if holm and not out.empty:
        out["p_holm"] = holm_adjust(out["p_boot"].astype(float).tolist())
    else:
        out["p_holm"] = np.nan
    return out


def _wald_sum_two(res, name_a: str, name_b: str) -> tuple[float, float, float, float]:
    """Trả (sum, wald, p, var_sum) cho H0: β_a+β_b=0."""
    s = float(res.params[name_a] + res.params[name_b])
    cov = res.cov_params()
    var = (
        float(cov.loc[name_a, name_a])
        + float(cov.loc[name_b, name_b])
        + 2.0 * float(cov.loc[name_a, name_b])
    )
    if var <= 0 or not np.isfinite(var):
        return s, float("nan"), float("nan"), var
    w = float(s ** 2 / var)
    p = float(1.0 - stats.chi2.cdf(w, 1))
    return s, w, p, var


def wald_pos_neg(
    work: pd.DataFrame,
    y_col: str,
    control_cols: list[str] | tuple[str, ...],
    bandwidth: int,
) -> dict:
    """Bản chính: y ~ controls + PosSent + NegSent (tỷ lệ gốc, chưa z).

    Wald H0: β_pos + β_neg = 0. Bản chuẩn hóa 1SD giữ làm cột phụ (*_1sd).
    """
    ctrl = list(control_cols)
    # --- chính: raw proportions ---
    X_raw = work[ctrl].copy()
    X_raw["pos"] = work[COL_POS].astype(float).to_numpy()
    X_raw["neg"] = work[COL_NEG].astype(float).to_numpy()
    res_raw = fit_ols_nw(work[y_col], X_raw, bandwidth=bandwidth)
    b_pos = float(res_raw.params["pos"])
    b_neg = float(res_raw.params["neg"])
    se_pos = float(res_raw.bse["pos"])
    se_neg = float(res_raw.bse["neg"])
    s, w, p, _ = _wald_sum_two(res_raw, "pos", "neg")

    # --- phụ: 1 SD ---
    z_pos, _, _ = zscore_series(work[COL_POS])
    z_neg, _, _ = zscore_series(work[COL_NEG])
    X_z = work[ctrl].copy()
    X_z["z_pos"] = z_pos.to_numpy()
    X_z["z_neg"] = z_neg.to_numpy()
    res_z = fit_ols_nw(work[y_col], X_z, bandwidth=bandwidth)
    s1, w1, p1, _ = _wald_sum_two(res_z, "z_pos", "z_neg")

    return {
        "scale": "raw",
        "coef_pos": b_pos,
        "se_pos_nw": se_pos,
        "coef_neg": b_neg,
        "se_neg_nw": se_neg,
        "coef_sum": s,
        "wald_stat": w,
        "p_wald": p,
        "neg_stronger": bool(s < 0) if np.isfinite(s) else False,
        # phụ 1SD
        "coef_pos_1sd": float(res_z.params["z_pos"]),
        "se_pos_1sd": float(res_z.bse["z_pos"]),
        "coef_neg_1sd": float(res_z.params["z_neg"]),
        "se_neg_1sd": float(res_z.bse["z_neg"]),
        "coef_sum_1sd": s1,
        "wald_stat_1sd": w1,
        "p_wald_1sd": p1,
        "n": int(len(work)),
        "nw_bandwidth": int(bandwidth),
        "r2_adj": float(res_raw.rsquared_adj),
    }


def run_rq2_for_asset(
    df_a: pd.DataFrame,
    df_b_by_h: dict[int, pd.DataFrame],
    asset: str,
    *,
    family: str = "rq2_main",
    label: str = "main",
) -> pd.DataFrame:
    rows = []
    # Spec A
    work_a = common_sample(df_a, "y_ret", [COL_POS, COL_NEG], CONTROL_A)
    stats_a = wald_pos_neg(work_a, "y_ret", CONTROL_A, nw_bandwidth(None))
    rows.append({
        "asset": asset,
        "family": family,
        "label": label,
        "hypothesis": "H2_A",
        "spec": "A",
        "h": 0,
        "y": "y_ret",
        **stats_a,
    })
    # Spec B
    for h in HORIZONS_B:
        df_b = df_b_by_h[h]
        ycol = f"y_ret_{h}"
        work_b = common_sample(df_b, ycol, [COL_POS, COL_NEG], CONTROL_B)
        stats_b = wald_pos_neg(work_b, ycol, CONTROL_B, nw_bandwidth(h))
        rows.append({
            "asset": asset,
            "family": family,
            "label": label,
            "hypothesis": f"H2_B_h{h}",
            "spec": "B",
            "h": h,
            "y": ycol,
            **stats_b,
        })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_wald"].astype(float).tolist())
    out["confirmed"] = out["p_holm"] < 0.05
    return out


def _drop_low_news_volume(df: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    if COL_NVOL not in df.columns:
        return df
    thr = float(df[COL_NVOL].quantile(q))
    return df.loc[df[COL_NVOL] >= thr].copy()


def _registry_rows(df: pd.DataFrame, ts: str, p_col: str) -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        scale = r.get("scale", "1sd" if "rq1" in str(r.get("family", "")) else "raw")
        rows.append({
            "timestamp": ts,
            "script": "models/rq1_rq2_sentiment.py",
            "model": f"{r.get('hypothesis', '')}|{r.get('label', '')}|{r.get('y', '')}",
            "asset": r["asset"],
            "h": r.get("h", ""),
            "split": "train+valid",
            "n_oos": r.get("n", ""),
            "p_raw": r.get(p_col, float("nan")),
            "baseline": "n/a",
            "family": r["family"],
            "scale": scale,
        })
    return rows


def run_all(
    root: Path | None = None,
    out_dir: Path | None = None,
    seed: int = SEED,
    n_boot: int = N_BOOT,
    include_robust: bool = True,
    table_a_path: Path | str | None = None,
    table_b_path: Path | str | None = None,
    exclude_sources: list[str] | None = None,
    register: bool = True,
    registry_family: str | None = None,
) -> dict[str, pd.DataFrame]:
    root = root or project_root()
    params = load_params(root)
    seed = int(params.get("seed", seed))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (root / "outputs" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    raw_a = load_train_valid(
        "A", root=root, panel="title", table_path=table_a_path,
        exclude_sources=exclude_sources,
    )
    raw_b = {
        h: load_train_valid(
            "B", root=root, panel="title", horizon=h, drop_embargo=True,
            table_path=table_b_path, exclude_sources=exclude_sources,
        )
        for h in HORIZONS_B
    }

    rq1_parts = []
    rq2_parts = []
    reg_rows: list[dict] = []

    for asset in ("BTC", "ETH"):
        fam1 = "rq1_main" if asset == "BTC" else "rq1_robust"
        fam2 = "rq2_main" if asset == "BTC" else "rq2_robust"
        lab = "main" if asset == "BTC" else "robust_ETH"
        df_a = prepare_asset_frame(raw_a, asset)
        b_asset = {h: prepare_asset_frame(raw_b[h], asset) for h in HORIZONS_B}

        r1 = run_rq1_for_asset(
            df_a, asset, family=fam1, label=lab, n_boot=n_boot, seed=seed,
        )
        r2 = run_rq2_for_asset(df_a, b_asset, asset, family=fam2, label=lab)
        r1.to_csv(out_dir / f"rq1_{asset}.csv", index=False)
        r2.to_csv(out_dir / f"rq2_{asset}.csv", index=False)
        rq1_parts.append(r1)
        rq2_parts.append(r2)
        reg_rows.extend(_registry_rows(r1, ts, "p_boot"))
        reg_rows.extend(_registry_rows(r2, ts, "p_wald"))
        print(
            "%s RQ1 confirmed=%d/%d | RQ2 confirmed=%d/%d"
            % (asset, int(r1["confirmed"].sum()), len(r1), int(r2["confirmed"].sum()), len(r2))
        )

    # Độ bền ngoài PREREG — BTC
    if include_robust:
        df_btc = prepare_asset_frame(raw_a, "BTC")
        # (a) MeanProb
        r1_mp = run_rq1_for_asset(
            df_btc, "BTC", family="rq1_robust", label="robust_MeanProb",
            sent_f=COL_FIN_MEAN, sent_v=COL_VAD_MEAN, n_boot=n_boot, seed=seed,
        )
        rq1_parts.append(r1_mp)
        reg_rows.extend(_registry_rows(r1_mp, ts, "p_boot"))
        print("robust MeanProb done")

        # (b) nlp_text / fulltext panel
        try:
            raw_a_ft = load_train_valid(
                "A", root=root, panel="fulltext", exclude_sources=exclude_sources,
            )
            df_ft = prepare_asset_frame(raw_a_ft, "BTC")
            r1_ft = run_rq1_for_asset(
                df_ft, "BTC", family="rq1_robust", label="robust_nlp_text",
                n_boot=n_boot, seed=seed,
            )
            rq1_parts.append(r1_ft)
            reg_rows.extend(_registry_rows(r1_ft, ts, "p_boot"))
            print("robust nlp_text done")
        except FileNotFoundError as e:
            print("skip robust_nlp_text:", e)

        # (c) bỏ NewsVolume thấp nhất 5%
        df_nv = _drop_low_news_volume(df_btc, 0.05)
        r1_nv = run_rq1_for_asset(
            df_nv, "BTC", family="rq1_robust", label="robust_drop_nv_bot5",
            n_boot=n_boot, seed=seed,
        )
        b_btc = {h: prepare_asset_frame(raw_b[h], "BTC") for h in HORIZONS_B}
        b_nv = {h: _drop_low_news_volume(b_btc[h], 0.05) for h in HORIZONS_B}
        r2_nv = run_rq2_for_asset(
            df_nv, b_nv, "BTC", family="rq2_robust", label="robust_drop_nv_bot5",
        )
        rq1_parts.append(r1_nv)
        rq2_parts.append(r2_nv)
        reg_rows.extend(_registry_rows(r1_nv, ts, "p_boot"))
        reg_rows.extend(_registry_rows(r2_nv, ts, "p_wald"))
        print("robust drop NewsVolume bot5 done")

        # (d) panel noprice — ngoài Holm chính
        try:
            raw_a_np = load_train_valid(
                "A", root=root, panel="noprice", exclude_sources=exclude_sources,
            )
            raw_b_np = {
                h: load_train_valid(
                    "B", root=root, panel="noprice", horizon=h, drop_embargo=True,
                    exclude_sources=exclude_sources,
                )
                for h in HORIZONS_B
            }
            df_np = prepare_asset_frame(raw_a_np, "BTC")
            b_np = {h: prepare_asset_frame(raw_b_np[h], "BTC") for h in HORIZONS_B}
            r1_np = run_rq1_for_asset(
                df_np, "BTC", family="rq1_robust_noprice", label="robust_noprice",
                n_boot=n_boot, seed=seed,
            )
            r2_np = run_rq2_for_asset(
                df_np, b_np, "BTC", family="rq2_robust_noprice", label="robust_noprice",
            )
            rq1_parts.append(r1_np)
            rq2_parts.append(r2_np)
            reg_rows.extend(_registry_rows(r1_np, ts, "p_boot"))
            reg_rows.extend(_registry_rows(r2_np, ts, "p_wald"))
            print("robust noprice done")
        except FileNotFoundError as e:
            print("skip robust_noprice:", e)

    rq1_all = pd.concat(rq1_parts, ignore_index=True)
    rq2_all = pd.concat(rq2_parts, ignore_index=True)
    summary = pd.concat([
        rq1_all.assign(table="rq1"),
        rq2_all.assign(table="rq2"),
    ], ignore_index=True)
    sum_path = out_dir / "rq1_rq2_summary.csv"
    summary.to_csv(sum_path, index=False)

    if registry_family:
        for r in reg_rows:
            r["family"] = registry_family
    reg_path = out_dir / "test_registry.csv"
    if register:
        append_registry(reg_rows, reg_path)

    meta = {
        "seed": seed,
        "n_boot": n_boot,
        "block_len": BLOCK_LEN,
        "rq1_holm": "2 tests (H1a,H1b) per asset family",
        "rq2_holm": "4 tests (A + B×3) per asset family",
        "n_registry_appended": len(reg_rows) if register else 0,
        "summary_path": str(sum_path),
        "exclude_sources": exclude_sources,
        "table_a_path": str(table_a_path) if table_a_path else None,
        "table_b_path": str(table_b_path) if table_b_path else None,
        "register": register,
    }
    (out_dir / "rq1_rq2_summary.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Summary ->", sum_path)
    if register:
        print("Registry appended", len(reg_rows), "->", reg_path)
    else:
        print("Registry skipped (register=False)")
    return {"rq1": rq1_all, "rq2": rq2_all, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ1 + RQ2 sentiment (train+valid)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--no-robust", action="store_true")
    args = parser.parse_args()
    run_all(n_boot=args.n_boot, include_robust=not args.no_robust)


if __name__ == "__main__":
    main()

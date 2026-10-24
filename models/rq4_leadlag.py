"""RQ4 / PREREG họ 4 — Granger lead-lag Sent ↔ ret (Spec A).

Bản chính: model_table_A_noprice, BTC train+valid, load_model_table.
Trễ theo NGÀY LỊCH: reindex lịch đầy đủ rồi shift(k); NaN ở t−k → bỏ dòng.
VAR(p) hai biến + 6 DOW; p ∈ {1..7} chọn BIC trên CÙNG mẫu (hợp lệ cho p=7).

Granger: Wald HAC (bw=5) trên TỔNG hệ số trễ của biến kia, từng phương trình
(có thể khác n). Holm 2 phép / asset.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.rq3_narratives import (  # noqa: E402
    NW_BANDWIDTH,
    SEED,
    append_registry,
    fit_ols_nw,
    holm_adjust,
    prepare_asset_frame,
    project_root,
)

COL_RET = "y_ret"
COL_SENT = "all_finbert_Sent_t"
COL_MEAN = "all_finbert_MeanProb_t"
COL_ABN = "all_finbert_AbnAttention_t"
MAX_P = 7
IRF_HORIZON = 7
N_BOOT_IRF = 500
NOTE_CALENDAR = "calendar_lag"


def load_params(root: Path | None = None) -> dict:
    import yaml
    root = root or project_root()
    return yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}


def load_train_valid_A(
    *,
    root: Path | None = None,
    panel: str = "noprice",
) -> pd.DataFrame:
    root = root or project_root()
    train = load_model_table("A", "train", root=root, panel=panel, drop_embargo=False)
    valid = load_model_table("A", "valid", root=root, panel=panel, drop_embargo=False)
    out = pd.concat([train, valid], ignore_index=True)
    if (out["split"].astype(str) == "test").any():
        raise RuntimeError("RQ4 không được chứa dòng test")
    return out


def to_calendar_frame(
    df: pd.DataFrame,
    asset: str,
    sent_col: str = COL_SENT,
) -> pd.DataFrame:
    """Lọc asset, reindex lịch ngày đầy đủ [min,max], gắn DOW theo lịch, giữ NaN."""
    out = prepare_asset_frame(df, asset)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = out.drop_duplicates("date", keep="first").sort_values("date")
    cal = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    keep = ["date", COL_RET, sent_col]
    # giữ cột khác nếu có (MeanProb/Abn đã chọn làm sent_col)
    indexed = out.set_index("date")
    # chỉ lấy cột cần; DOW sẽ tính lại trên lịch đầy đủ
    cols = [c for c in [COL_RET, sent_col] if c in indexed.columns]
    wide = indexed[cols].reindex(cal)
    wide.index.name = "date"
    wide = wide.reset_index()
    wide = add_dow_dummies(wide)
    return wide


def prepare_rq4_frame(
    df: pd.DataFrame,
    asset: str,
    sent_col: str = COL_SENT,
) -> pd.DataFrame:
    """Alias: khung lịch đầy đủ (có thể còn NaN ở ret/Sent)."""
    return to_calendar_frame(df, asset, sent_col=sent_col)


def add_calendar_lags(
    cal: pd.DataFrame,
    y_col: str,
    x_col: str,
    max_p: int = MAX_P,
) -> pd.DataFrame:
    """Thêm y_L1..y_Lp, x_L1..x_Lp bằng shift trên lịch ngày (không bỏ NaN giữa chừng)."""
    out = cal.sort_values("date").reset_index(drop=True).copy()
    for k in range(1, max_p + 1):
        out[f"y_L{k}"] = out[y_col].shift(k)
        out[f"x_L{k}"] = out[x_col].shift(k)
    return out


def valid_mask_for_p(df: pd.DataFrame, p: int, y_col: str) -> pd.Series:
    """Dòng hợp lệ cho phương trình bậc p: y_t + lags 1..p của y và x + DOW."""
    cols = [y_col] + [f"y_L{k}" for k in range(1, p + 1)] + [f"x_L{k}" for k in range(1, p + 1)]
    cols = cols + list(DOW_DUMMY_COLS)
    return df[cols].notna().all(axis=1)


def common_mask_max_p(df: pd.DataFrame, y_col: str, max_p: int = MAX_P) -> pd.Series:
    """Mẫu chung: hợp lệ cho mọi p ≤ max_p (tức hợp lệ cho max_p)."""
    return valid_mask_for_p(df, max_p, y_col)


def _bic_gaussian(n: int, sigma_det: float, n_params: int) -> float:
    """BIC = n*log(det Σ) + k*log(n) (bỏ hằng số)."""
    if n <= 0 or not np.isfinite(sigma_det) or sigma_det <= 0:
        return float("inf")
    return float(n * np.log(sigma_det) + n_params * np.log(n))


def fit_var_eq_on_mask(
    df: pd.DataFrame,
    y_col: str,
    p: int,
    mask: pd.Series,
    bandwidth: int = NW_BANDWIDTH,
):
    """OLS+HAC một phương trình trên mask; trả res, tên cột x-lags."""
    sub = df.loc[mask].copy()
    names_y = [f"y_L{k}" for k in range(1, p + 1)]
    names_x = [f"x_L{k}" for k in range(1, p + 1)]
    X = sub[names_y + names_x + list(DOW_DUMMY_COLS)]
    res = fit_ols_nw(sub[y_col], X, bandwidth=bandwidth)
    return res, names_y, names_x, sub


def select_p_bic_system(
    cal: pd.DataFrame,
    ret_col: str,
    sent_col: str,
    max_p: int = MAX_P,
) -> tuple[int, dict[int, float], int]:
    """Chọn p theo BIC hệ 2 pt trên mẫu chung (hợp lệ p=max_p cả hai chiều)."""
    # Xây lags cho cả hai biến
    work = cal.sort_values("date").reset_index(drop=True).copy()
    for k in range(1, max_p + 1):
        work[f"ret_L{k}"] = work[ret_col].shift(k)
        work[f"sent_L{k}"] = work[sent_col].shift(k)

    need = (
        [ret_col, sent_col]
        + [f"ret_L{k}" for k in range(1, max_p + 1)]
        + [f"sent_L{k}" for k in range(1, max_p + 1)]
        + list(DOW_DUMMY_COLS)
    )
    mask = work[need].notna().all(axis=1)
    n_common = int(mask.sum())
    bic_table: dict[int, float] = {}

    for p in range(1, max_p + 1):
        ret_regs = (
            [f"ret_L{k}" for k in range(1, p + 1)]
            + [f"sent_L{k}" for k in range(1, p + 1)]
            + list(DOW_DUMMY_COLS)
        )
        sent_regs = (
            [f"sent_L{k}" for k in range(1, p + 1)]
            + [f"ret_L{k}" for k in range(1, p + 1)]
            + list(DOW_DUMMY_COLS)
        )
        sub = work.loc[mask]
        # OLS thường (điểm) để BIC
        import statsmodels.api as sm
        Yr = sub[ret_col].to_numpy(dtype=float)
        Ys = sub[sent_col].to_numpy(dtype=float)
        Xr = sm.add_constant(sub[ret_regs].to_numpy(dtype=float), has_constant="add")
        Xs = sm.add_constant(sub[sent_regs].to_numpy(dtype=float), has_constant="add")
        br = np.linalg.lstsq(Xr, Yr, rcond=None)[0]
        bs = np.linalg.lstsq(Xs, Ys, rcond=None)[0]
        er = Yr - Xr @ br
        es = Ys - Xs @ bs
        E = np.column_stack([er, es])
        sigma = (E.T @ E) / n_common
        det = float(np.linalg.det(sigma))
        # params: 2 * (1 + 2p + 6 DOW) = 2*(7+2p)
        n_params = 2 * (1 + 2 * p + len(DOW_DUMMY_COLS))
        bic_table[p] = _bic_gaussian(n_common, det, n_params)

    p_star = int(min(bic_table, key=bic_table.get))
    return p_star, bic_table, n_common


def granger_sum_wald_calendar(
    cal: pd.DataFrame,
    y_col: str,
    x_col: str,
    p: int,
    bandwidth: int = NW_BANDWIDTH,
) -> dict:
    """Phương trình y trên lịch: y_t ~ y_L1..p + x_L1..p + DOW; H0 sum(x lags)=0."""
    work = cal.sort_values("date").reset_index(drop=True).copy()
    for k in range(1, p + 1):
        work[f"y_L{k}"] = work[y_col].shift(k)
        work[f"x_L{k}"] = work[x_col].shift(k)
    mask = valid_mask_for_p(work, p, y_col)
    sub = work.loc[mask]
    names_y = [f"y_L{k}" for k in range(1, p + 1)]
    names_x = [f"x_L{k}" for k in range(1, p + 1)]
    X = sub[names_y + names_x + list(DOW_DUMMY_COLS)]
    res = fit_ols_nw(sub[y_col], X, bandwidth=bandwidth)
    b = np.array([float(res.params[n]) for n in names_x])
    s = float(b.sum())
    cov = res.cov_params().loc[names_x, names_x].to_numpy(dtype=float)
    ones = np.ones(p)
    var = float(ones @ cov @ ones)
    if var <= 0 or not np.isfinite(var):
        w, pval = float("nan"), float("nan")
    else:
        w = float(s ** 2 / var)
        pval = float(1.0 - stats.chi2.cdf(w, 1))
    return {
        "coef_sum": s,
        "se_sum": float(np.sqrt(var)) if var > 0 else float("nan"),
        "wald_stat": w,
        "p_raw": pval,
        "n": int(mask.sum()),
        "p_lags": int(p),
        "nw_bandwidth": int(bandwidth),
        "coefs_lags": b.tolist(),
        "params": {n: float(res.params[n]) for n in names_y + names_x},
        "resid": res.resid.to_numpy(dtype=float),
        "dates": sub["date"].to_numpy(),
    }


def run_granger_pair(
    df: pd.DataFrame,
    sent_col: str,
    p: int | None = None,
    bandwidth: int = NW_BANDWIDTH,
) -> dict:
    """df = khung lịch (có thể NaN). Chọn p bằng BIC mẫu chung nếu p is None."""
    cal = df.sort_values("date").reset_index(drop=True).copy()
    # đảm bảo DOW
    if not set(DOW_DUMMY_COLS).issubset(cal.columns):
        cal = add_dow_dummies(cal)

    bic_table: dict[int, float] = {}
    n_common = 0
    if p is None:
        p, bic_table, n_common = select_p_bic_system(cal, COL_RET, sent_col, max_p=MAX_P)
    else:
        p = int(p)

    g_sr = granger_sum_wald_calendar(cal, COL_RET, sent_col, p, bandwidth=bandwidth)
    g_rs = granger_sum_wald_calendar(cal, sent_col, COL_RET, p, bandwidth=bandwidth)
    return {
        "p": int(p),
        "bic_table": bic_table,
        "n_common_bic": n_common,
        "sent_to_ret": g_sr,
        "ret_to_sent": g_rs,
        "n_obs_calendar": int(len(cal)),
        "note": NOTE_CALENDAR,
    }


# --- tương thích tên cũ (tests cũ có thể gọi) ---
def select_p_bic(endog: pd.DataFrame, exog: pd.DataFrame | None, maxlags: int = MAX_P):
    """Wrapper: endog contiguous không NaN → BIC hệ thống (lịch = index liên tục)."""
    cal = endog.copy()
    if "ret" in cal.columns and "sent" in cal.columns:
        cal = cal.rename(columns={"ret": COL_RET, "sent": COL_SENT})
    if "date" not in cal.columns:
        cal = cal.reset_index(drop=True)
        cal["date"] = pd.date_range("2021-01-01", periods=len(cal), freq="D")
    if exog is not None:
        for c in exog.columns:
            cal[c] = exog[c].to_numpy()
    else:
        cal = add_dow_dummies(cal)
    p, bic, _ = select_p_bic_system(cal, COL_RET, COL_SENT, maxlags)
    return p, bic


def granger_sum_wald(y, x, exog, p, bandwidth=NW_BANDWIDTH):
    """Wrapper array → lịch giả định liên tục (cho test cũ không có NaN giữa)."""
    n = len(y)
    cal = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n, freq="D"),
        "y": np.asarray(y, dtype=float),
        "x": np.asarray(x, dtype=float),
    })
    if exog is not None:
        ex = np.asarray(exog, dtype=float)
        for j, name in enumerate(DOW_DUMMY_COLS):
            cal[name] = ex[:, j] if ex.ndim == 2 else 0.0
    else:
        cal = add_dow_dummies(cal)
    # remap to COL names expected by calendar fn via local rename
    work = cal.rename(columns={"y": COL_RET, "x": COL_SENT})
    # trick: call with y_col/x_col as COL_RET/COL_SENT
    return granger_sum_wald_calendar(work, COL_RET, COL_SENT, p, bandwidth=bandwidth)


def bootstrap_irf_from_eqs(
    g: dict,
    p: int,
    *,
    horizon: int = IRF_HORIZON,
    n_boot: int = N_BOOT_IRF,
    seed: int = SEED,
) -> dict:
    """IRF từ hệ số phương trình (mẫu complete-case lịch); bootstrap phần dư."""
    # Align residuals on intersection of dates
    d_sr = pd.DataFrame({"date": g["sent_to_ret"]["dates"], "e_ret": g["sent_to_ret"]["resid"]})
    d_rs = pd.DataFrame({"date": g["ret_to_sent"]["dates"], "e_sent": g["ret_to_sent"]["resid"]})
    both = d_sr.merge(d_rs, on="date", how="inner")
    if len(both) < 10:
        nan = np.full((horizon + 1, 2, 2), np.nan)
        return {"point": nan, "lo": nan, "hi": nan, "i_ret": 0, "i_sent": 1, "n_boot": 0}

    # Companion from coefs: ret eq and sent eq
    # ret_t = a' ret_lags + b' sent_lags + ...
    # sent_t = c' sent_lags + d' ret_lags + ...
    psr = g["sent_to_ret"]["params"]
    prs = g["ret_to_sent"]["params"]

    def companion(params_ret, params_sent, p_):
        # state = [ret_1..ret_p, sent_1..sent_p]
        k = 2 * p_
        A = np.zeros((k, k))
        # ret equation row 0
        for i in range(p_):
            A[0, i] = params_ret.get(f"y_L{i+1}", 0.0)  # own lags of ret
            A[0, p_ + i] = params_ret.get(f"x_L{i+1}", 0.0)  # sent lags
        # sent equation row p_
        for i in range(p_):
            A[p_, p_ + i] = params_sent.get(f"y_L{i+1}", 0.0)  # own lags of sent
            A[p_, i] = params_sent.get(f"x_L{i+1}", 0.0)  # ret lags
        # shift structure
        for i in range(1, p_):
            A[i, i - 1] = 1.0
            A[p_ + i, p_ + i - 1] = 1.0
        return A

    A0 = companion(psr, prs, p)
    E = both[["e_ret", "e_sent"]].to_numpy(dtype=float)
    sigma = np.cov(E, rowvar=False)

    def irf_path(A, Sig, H):
        # reduced-form shock: identity impact on [ret, sent] contemporaneous
        # state shock: only first of each block
        P = np.linalg.cholesky(Sig + 1e-12 * np.eye(2))
        out = np.zeros((H + 1, 2, 2))
        # out[h, response, impulse]
        for impulse in range(2):
            e0 = P[:, impulse]
            state = np.zeros(2 * p)
            state[0] = e0[0]
            state[p] = e0[1]
            out[0, 0, impulse] = state[0]
            out[0, 1, impulse] = state[p]
            for h in range(1, H + 1):
                state = A @ state
                out[h, 0, impulse] = state[0]
                out[h, 1, impulse] = state[p]
        return out

    point = irf_path(A0, sigma, horizon)
    rng = np.random.default_rng(seed)
    boots = []
    n = len(E)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Sig_b = np.cov(E[idx], rowvar=False)
        # keep A fixed (coefficient uncertainty omitted) — residual scale only
        boots.append(irf_path(A0, Sig_b, horizon))
    stack = np.stack(boots, axis=0)
    return {
        "point": point,
        "lo": np.percentile(stack, 2.5, axis=0),
        "hi": np.percentile(stack, 97.5, axis=0),
        "i_ret": 0,
        "i_sent": 1,
        "n_boot": len(boots),
    }


def plot_irf(irf_res: dict, out_path: Path, title: str) -> None:
    point, lo, hi = irf_res["point"], irf_res["lo"], irf_res["hi"]
    i_ret, i_sent = irf_res["i_ret"], irf_res["i_sent"]
    h = np.arange(point.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharex=True)
    axes[0].plot(h, point[:, i_ret, i_sent], color="C0")
    axes[0].fill_between(h, lo[:, i_ret, i_sent], hi[:, i_ret, i_sent], alpha=0.25, color="C0")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("IRF: Sent → ret")
    axes[0].set_xlabel("h")
    axes[1].plot(h, point[:, i_sent, i_ret], color="C1")
    axes[1].fill_between(h, lo[:, i_sent, i_ret], hi[:, i_sent, i_ret], alpha=0.25, color="C1")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("IRF: ret → Sent")
    axes[1].set_xlabel("h")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _rows_from_granger(
    asset: str,
    panel: str,
    label: str,
    family: str,
    sent_col: str,
    g: dict,
) -> list[dict]:
    rows = []
    for direction, key in (("sent_to_ret", "sent_to_ret"), ("ret_to_sent", "ret_to_sent")):
        st = g[key]
        rows.append({
            "asset": asset,
            "panel": panel,
            "label": label,
            "family": family,
            "direction": direction,
            "sent_col": sent_col,
            "p": g["p"],
            "coef_sum": st["coef_sum"],
            "se_sum": st["se_sum"],
            "wald_stat": st["wald_stat"],
            "p_raw": st["p_raw"],
            "n": st["n"],
            "nw_bandwidth": st["nw_bandwidth"],
            "scale": "raw",
            "note": NOTE_CALENDAR,
            "coefs_lags": st["coefs_lags"],
        })
    return rows


def run_asset_main(
    raw: pd.DataFrame,
    asset: str,
    *,
    panel: str,
    sent_col: str = COL_SENT,
    family: str = "rq4_main",
    label: str = "main_noprice",
    fixed_p: int | None = None,
    do_irf: bool = True,
    fig_dir: Path | None = None,
    n_boot_irf: int = N_BOOT_IRF,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict]:
    cal = to_calendar_frame(raw, asset, sent_col=sent_col)
    g = run_granger_pair(cal, sent_col, p=fixed_p)
    rows = _rows_from_granger(asset, panel, label, family, sent_col, g)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_raw"].astype(float).tolist())
    out["confirmed"] = out["p_holm"] < 0.05
    s2r = out.loc[out["direction"] == "sent_to_ret"].iloc[0]
    out["text_leads_price"] = bool(s2r["confirmed"]) if family == "rq4_main" else False

    meta = {
        "p": g["p"],
        "bic_table": g["bic_table"],
        "n_common_bic": g.get("n_common_bic"),
        "n_sent_to_ret": g["sent_to_ret"]["n"],
        "n_ret_to_sent": g["ret_to_sent"]["n"],
        "note": NOTE_CALENDAR,
    }
    if do_irf and fig_dir is not None:
        irf_res = bootstrap_irf_from_eqs(g, g["p"], n_boot=n_boot_irf, seed=seed)
        fig_path = fig_dir / f"rq4_irf_{asset}_{label}.png"
        plot_irf(irf_res, fig_path, title=f"RQ4 IRF {asset} {label} (p={g['p']}, calendar_lag)")
        meta["irf_path"] = str(fig_path)
        meta["irf_n_boot"] = irf_res.get("n_boot")
    return out, meta


def run_all(
    root: Path | None = None,
    out_dir: Path | None = None,
    fig_dir: Path | None = None,
    seed: int = SEED,
    n_boot_irf: int = N_BOOT_IRF,
) -> pd.DataFrame:
    root = root or project_root()
    params = load_params(root)
    seed = int(params.get("seed", seed))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (root / "outputs" / "tables")
    fig_dir = fig_dir or (root / "outputs" / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    raw_np = load_train_valid_A(root=root, panel="noprice")
    raw_full = load_train_valid_A(root=root, panel="title")

    parts = []
    reg_rows = []
    metas = {}

    for asset in ("BTC", "ETH"):
        fam = "rq4_main" if asset == "BTC" else "rq4_robust"
        lab = "main_noprice" if asset == "BTC" else "robust_ETH_noprice"
        tab, meta = run_asset_main(
            raw_np, asset, panel="noprice", family=fam, label=lab,
            do_irf=True, fig_dir=fig_dir, n_boot_irf=n_boot_irf, seed=seed,
        )
        parts.append(tab)
        metas[f"{asset}_{lab}"] = meta
        print(
            "%s p=%d | n_SR=%d n_RS=%d | Sent→ret p=%.4g Holm=%.4g | "
            "ret→Sent p=%.4g Holm=%.4g | text_leads=%s | BIC=%s"
            % (
                asset, meta["p"], meta["n_sent_to_ret"], meta["n_ret_to_sent"],
                tab.loc[tab.direction == "sent_to_ret", "p_raw"].iloc[0],
                tab.loc[tab.direction == "sent_to_ret", "p_holm"].iloc[0],
                tab.loc[tab.direction == "ret_to_sent", "p_raw"].iloc[0],
                tab.loc[tab.direction == "ret_to_sent", "p_holm"].iloc[0],
                tab["text_leads_price"].iloc[0],
                {k: round(v, 2) for k, v in (meta.get("bic_table") or {}).items()},
            )
        )
        for _, r in tab.iterrows():
            reg_rows.append({
                "timestamp": ts,
                "script": "models/rq4_leadlag.py",
                "model": f"{r['direction']}|{r['label']}|p{r['p']}",
                "asset": asset,
                "h": r["p"],
                "split": "train+valid",
                "n_oos": r["n"],
                "p_raw": r["p_raw"],
                "baseline": "n/a",
                "family": r["family"],
                "scale": "raw",
                "note": NOTE_CALENDAR,
            })

    robust_specs = [
        ("title", COL_SENT, "robust_full_panel", None),
        ("noprice", COL_MEAN, "robust_MeanProb", None),
        ("noprice", COL_ABN, "robust_AbnAttention", None),
        ("noprice", COL_SENT, "robust_p1", 1),
        ("noprice", COL_SENT, "robust_p3", 3),
        ("noprice", COL_SENT, "robust_p7", 7),
    ]
    for panel, scol, lab, fp in robust_specs:
        raw = raw_full if panel == "title" else raw_np
        try:
            tab, meta = run_asset_main(
                raw, "BTC", panel=panel, sent_col=scol,
                family="rq4_robust", label=lab, fixed_p=fp,
                do_irf=False, seed=seed,
            )
        except Exception as e:
            print("skip", lab, e)
            continue
        parts.append(tab)
        metas[lab] = meta
        for _, r in tab.iterrows():
            reg_rows.append({
                "timestamp": ts,
                "script": "models/rq4_leadlag.py",
                "model": f"{r['direction']}|{lab}|p{r['p']}",
                "asset": "BTC",
                "h": r["p"],
                "split": "train+valid",
                "n_oos": r["n"],
                "p_raw": r["p_raw"],
                "baseline": "n/a",
                "family": "rq4_robust",
                "scale": "raw",
                "note": NOTE_CALENDAR,
            })
        print(
            "robust", lab, "p=", meta["p"],
            "n_SR=", meta["n_sent_to_ret"], "n_RS=", meta["n_ret_to_sent"],
            "BIC=", {k: round(v, 2) for k, v in (meta.get("bic_table") or {}).items()},
        )

    summary = pd.concat(parts, ignore_index=True)
    # don't store list cols in csv awkwardly
    if "coefs_lags" in summary.columns:
        summary = summary.drop(columns=["coefs_lags"])
    sum_path = out_dir / "rq4_summary.csv"
    summary.to_csv(sum_path, index=False)
    summary.loc[summary["asset"] == "BTC"].to_csv(out_dir / "rq4_BTC.csv", index=False)
    summary.loc[summary["asset"] == "ETH"].to_csv(out_dir / "rq4_ETH.csv", index=False)

    bic = metas.get("BTC_main_noprice", {}).get("bic_table", {})
    if bic:
        bic_df = pd.DataFrame([{"p": k, "bic": v} for k, v in sorted(bic.items())])
        bic_df.to_csv(out_dir / "rq4_bic_BTC.csv", index=False)
        print("BIC table (common sample n=%s):" % metas["BTC_main_noprice"].get("n_common_bic"))
        print(bic_df.to_string(index=False))

    append_registry(reg_rows, out_dir / "test_registry.csv")
    (out_dir / "rq4_summary.meta.json").write_text(
        json.dumps({"seed": seed, "metas": metas, "n_registry": len(reg_rows), "note": NOTE_CALENDAR},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print("Summary ->", sum_path, "| registry +", len(reg_rows))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ4 lead-lag (calendar lags)")
    parser.add_argument("--n-boot-irf", type=int, default=N_BOOT_IRF)
    args = parser.parse_args()
    run_all(n_boot_irf=args.n_boot_irf)


if __name__ == "__main__":
    main()

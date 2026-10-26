"""RQ3 / PREREG họ 3 — Spec A, mô tả liên hệ cùng ngày.

Liên hệ AbnAtt_k (7 narrative chính, scope=all) với y ∈ {y_log_rv, y_abvol, y_ret}.
Chỉ train + valid qua load_model_table('A', ...); tuyệt đối không đọc split='test'.

Hồi quy (mỗi narrative riêng):
  y_t ~ const + z(AbnAtt_k,t) + kiểm soát trễ + 6 biến giả DOW(t)

Biến giả ngày trong tuần bắt buộc vì AbnAtt có mùa vụ cuối tuần mạnh
(xem Sai khác 1 trong docs/PREREG.md — không sửa file đó từ script này).

z(AbnAtt_k) = (x − mean)/std trên chính mẫu ước lượng (complete-case),
hệ số báo theo 1 SD.

NW/HAC bandwidth = 5 (tối thiểu PREREG Spec A).
BTC = mẫu chính (Holm 21 phép); ETH = độ bền (Holm riêng 21 phép).
Chẩn đoán lead/lag và độ bền: chỉ BTC, ngoài họ Holm.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.har_text import holm_adjust  # noqa: E402

SEED = 42
NW_BANDWIDTH = 5

NARRATIVES: tuple[str, ...] = (
    "regulation",
    "etf",
    "institutional_adoption",
    "hacks_security",
    "defi",
    "stablecoins",
    "monetary_policy",
)
OUTCOMES: tuple[str, ...] = ("y_log_rv", "y_abvol", "y_ret")

# Kiểm soát: CHỈ trễ hoặc lịch xác định (DOW). Không có y_t / biến thị trường cùng ngày.
CONTROL_MARKET: tuple[str, ...] = (
    "ret_lag1",
    "ret_lag2",
    "har_d",
    "har_w",
    "har_m",
    "logvol_lag1",
)
CONTROL_FRED: tuple[str, ...] = (
    "fred_VIXCLS",
    "fred_VIXCLS_d5",
    "fred_DGS10",
    "fred_DGS10_d5",
    "fred_DTWEXBGS",
    "fred_DTWEXBGS_d5",
)
CONTROL_COLS: tuple[str, ...] = CONTROL_MARKET + CONTROL_FRED + DOW_DUMMY_COLS

TOTAL_ABN = "all_finbert_AbnAttention_t"
ASSETS_MAIN = ("BTC", "ETH")


def project_root() -> Path:
    return ROOT


def load_params(root: Path | None = None) -> dict:
    import yaml
    root = root or project_root()
    return yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}


def abn_col(narrative: str) -> str:
    return f"AbnAtt_{narrative}"


def zscore_series(x: pd.Series) -> tuple[pd.Series, float, float]:
    """Z-score trên mẫu ước lượng; std=0 → NaN."""
    mu = float(np.nanmean(x.to_numpy(dtype=float)))
    sd = float(np.nanstd(x.to_numpy(dtype=float), ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        z = pd.Series(np.nan, index=x.index)
        return z, mu, sd
    z = (x.astype(float) - mu) / sd
    return z, mu, sd


def prepare_asset_frame(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Lọc asset, gắn DOW, sắp theo ngày."""
    out = df.loc[df["asset"] == asset].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    out = add_dow_dummies(out)
    return out.sort_values("date").reset_index(drop=True)


def load_train_valid_A(
    root: Path | None = None,
    panel: str = "title",
) -> pd.DataFrame:
    """Chỉ train+valid Spec A; không bao giờ gọi split='test'."""
    root = root or project_root()
    train = load_model_table("A", "train", root=root, panel=panel, drop_embargo=False)
    valid = load_model_table("A", "valid", root=root, panel=panel, drop_embargo=False)
    out = pd.concat([train, valid], ignore_index=True)
    if "split" in out.columns and (out["split"] == "test").any():
        raise RuntimeError("Mẫu RQ3 không được chứa dòng test")
    return out


def fit_ols_nw(y: pd.Series, X: pd.DataFrame, bandwidth: int = NW_BANDWIDTH):
    """OLS + Newey–West/HAC."""
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    return sm.OLS(y.astype(float), Xc, missing="drop").fit(
        cov_type="HAC", cov_kwds={"maxlags": int(bandwidth)}
    )


def _r2_incremental(y: np.ndarray, X_ctrl: np.ndarray, X_full: np.ndarray) -> float:
    """R²_full − R²_controls trên cùng mẫu (OLS thường)."""
    y = np.asarray(y, dtype=float)
    def _r2(X):
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.OLS(y, Xc).fit()
        return float(res.rsquared)
    return _r2(X_full) - _r2(X_ctrl)


def run_one_regression(
    df: pd.DataFrame,
    *,
    y_col: str,
    x_raw: pd.Series,
    control_cols: list[str] | tuple[str, ...],
    bandwidth: int = NW_BANDWIDTH,
) -> dict:
    """Hồi quy y ~ z(x) + controls trên complete-case; z từ mẫu ước lượng."""
    ctrl = list(control_cols)
    work = df[["date", y_col, *ctrl]].copy()
    work["_x_raw"] = x_raw.reindex(df.index).to_numpy()
    work = work.dropna(subset=[y_col, "_x_raw", *ctrl]).reset_index(drop=True)
    n = len(work)
    if n < len(ctrl) + 5:
        return {
            "n": n,
            "coef_1sd": float("nan"),
            "se_nw": float("nan"),
            "t_stat": float("nan"),
            "p_raw": float("nan"),
            "r2_incremental": float("nan"),
            "pct_change_rv": float("nan"),
            "z_mean": float("nan"),
            "z_std": float("nan"),
        }
    z, mu, sd = zscore_series(work["_x_raw"])
    work["z_x"] = z.to_numpy()
    y = work[y_col]
    X_ctrl = work[ctrl]
    X_full = work[ctrl + ["z_x"]]
    res = fit_ols_nw(y, X_full, bandwidth=bandwidth)
    coef = float(res.params["z_x"])
    se = float(res.bse["z_x"])
    t = float(res.tvalues["z_x"])
    p = float(res.pvalues["z_x"])
    r2_inc = _r2_incremental(
        y.to_numpy(),
        X_ctrl.to_numpy(dtype=float),
        X_full.to_numpy(dtype=float),
    )
    pct_rv = float(100.0 * (np.exp(coef) - 1.0)) if y_col == "y_log_rv" else float("nan")
    return {
        "n": int(n),
        "coef_1sd": coef,
        "se_nw": se,
        "t_stat": t,
        "p_raw": p,
        "r2_incremental": r2_inc,
        "pct_change_rv": pct_rv,
        "z_mean": mu,
        "z_std": sd,
        "nw_bandwidth": int(bandwidth),
    }


def run_main_for_asset(df_asset: pd.DataFrame, asset: str) -> pd.DataFrame:
    rows = []
    for narr in NARRATIVES:
        xcol = abn_col(narr)
        if xcol not in df_asset.columns:
            raise KeyError("Thiếu cột %s" % xcol)
        for y in OUTCOMES:
            stats = run_one_regression(
                df_asset, y_col=y, x_raw=df_asset[xcol], control_cols=CONTROL_COLS,
            )
            rows.append({
                "asset": asset,
                "narrative": narr,
                "y": y,
                "family": "rq3_main",
                "label": "main",
                "x_spec": "contemporaneous",
                **stats,
            })
    out = pd.DataFrame(rows)
    adj = holm_adjust(out["p_raw"].astype(float).tolist())
    out["p_holm"] = adj
    out["confirmed"] = out["p_holm"] < 0.05
    return out


def run_diagnostics_btc(df_btc: pd.DataFrame) -> pd.DataFrame:
    """Lead/lag AbnAtt — chỉ BTC, nhãn diagnostic, ngoài Holm."""
    rows = []
    for narr in NARRATIVES:
        xcol = abn_col(narr)
        x = df_btc[xcol]
        x_lag = x.shift(1)
        x_lead = x.shift(-1)
        for y in OUTCOMES:
            for spec, xraw in (("lag_t-1", x_lag), ("lead_t+1", x_lead)):
                stats = run_one_regression(
                    df_btc, y_col=y, x_raw=xraw, control_cols=CONTROL_COLS,
                )
                rows.append({
                    "asset": "BTC",
                    "narrative": narr,
                    "y": y,
                    "family": "rq3_diag",
                    "label": "diagnostic",
                    "x_spec": spec,
                    "p_holm": float("nan"),
                    "confirmed": False,
                    **stats,
                })
    return pd.DataFrame(rows)


def run_robust_btc(df_btc: pd.DataFrame) -> pd.DataFrame:
    """Độ bền ngoài PREREG — chỉ BTC: (a) +total AbnAtt; (b) joint 7; (c) |z|≤4."""
    rows = []
    # (a) thêm tổng AbnAttention vào kiểm soát
    ctrl_a = list(CONTROL_COLS) + [TOTAL_ABN]
    for narr in NARRATIVES:
        xcol = abn_col(narr)
        for y in OUTCOMES:
            stats = run_one_regression(
                df_btc, y_col=y, x_raw=df_btc[xcol], control_cols=ctrl_a,
            )
            rows.append({
                "asset": "BTC",
                "narrative": narr,
                "y": y,
                "family": "rq3_robust",
                "label": "robust_total_abn",
                "x_spec": "contemporaneous",
                "p_holm": float("nan"),
                "confirmed": False,
                **stats,
            })

    # (b) joint cả 7 narrative
    for y in OUTCOMES:
        work = df_btc[["date", y, *CONTROL_COLS, *[abn_col(n) for n in NARRATIVES]]].copy()
        work = work.dropna().reset_index(drop=True)
        zcols = []
        for narr in NARRATIVES:
            z, mu, sd = zscore_series(work[abn_col(narr)])
            zc = f"z_{narr}"
            work[zc] = z.to_numpy()
            zcols.append(zc)
        X_ctrl = work[list(CONTROL_COLS)]
        X_full = work[list(CONTROL_COLS) + zcols]
        res = fit_ols_nw(work[y], X_full)
        r2_inc = _r2_incremental(
            work[y].to_numpy(),
            X_ctrl.to_numpy(dtype=float),
            X_full.to_numpy(dtype=float),
        )
        for narr, zc in zip(NARRATIVES, zcols):
            coef = float(res.params[zc])
            se = float(res.bse[zc])
            t = float(res.tvalues[zc])
            p = float(res.pvalues[zc])
            pct_rv = float(100.0 * (np.exp(coef) - 1.0)) if y == "y_log_rv" else float("nan")
            rows.append({
                "asset": "BTC",
                "narrative": narr,
                "y": y,
                "family": "rq3_robust",
                "label": "robust_joint7",
                "x_spec": "contemporaneous_joint",
                "n": int(len(work)),
                "coef_1sd": coef,
                "se_nw": se,
                "t_stat": t,
                "p_raw": p,
                "p_holm": float("nan"),
                "confirmed": False,
                "r2_incremental": r2_inc,
                "pct_change_rv": pct_rv,
                "z_mean": float("nan"),
                "z_std": float("nan"),
                "nw_bandwidth": NW_BANDWIDTH,
            })

    # (c) bỏ |z| > 4 (z trên mẫu complete trước khi cắt)
    for narr in NARRATIVES:
        xcol = abn_col(narr)
        for y in OUTCOMES:
            work = df_btc[["date", y, xcol, *CONTROL_COLS]].copy()
            work = work.dropna(subset=[y, xcol, *CONTROL_COLS]).reset_index(drop=True)
            z, mu, sd = zscore_series(work[xcol])
            work["z_x"] = z.to_numpy()
            work = work.loc[work["z_x"].abs() <= 4].reset_index(drop=True)
            if len(work) < len(CONTROL_COLS) + 5:
                stats = {
                    "n": len(work),
                    "coef_1sd": float("nan"), "se_nw": float("nan"),
                    "t_stat": float("nan"), "p_raw": float("nan"),
                    "r2_incremental": float("nan"), "pct_change_rv": float("nan"),
                    "z_mean": mu, "z_std": sd, "nw_bandwidth": NW_BANDWIDTH,
                }
            else:
                # re-z trên mẫu đã cắt
                z2, mu2, sd2 = zscore_series(work[xcol])
                work["z_x"] = z2.to_numpy()
                yv = work[y]
                X_ctrl = work[list(CONTROL_COLS)]
                X_full = work[list(CONTROL_COLS) + ["z_x"]]
                res = fit_ols_nw(yv, X_full)
                coef = float(res.params["z_x"])
                stats = {
                    "n": int(len(work)),
                    "coef_1sd": coef,
                    "se_nw": float(res.bse["z_x"]),
                    "t_stat": float(res.tvalues["z_x"]),
                    "p_raw": float(res.pvalues["z_x"]),
                    "r2_incremental": _r2_incremental(
                        yv.to_numpy(),
                        X_ctrl.to_numpy(dtype=float),
                        X_full.to_numpy(dtype=float),
                    ),
                    "pct_change_rv": (
                        float(100.0 * (np.exp(coef) - 1.0)) if y == "y_log_rv" else float("nan")
                    ),
                    "z_mean": mu2,
                    "z_std": sd2,
                    "nw_bandwidth": NW_BANDWIDTH,
                }
            rows.append({
                "asset": "BTC",
                "narrative": narr,
                "y": y,
                "family": "rq3_robust",
                "label": "robust_drop_zgt4",
                "x_spec": "contemporaneous",
                "p_holm": float("nan"),
                "confirmed": False,
                **stats,
            })
    return pd.DataFrame(rows)


def append_registry(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for col, default in (("family", ""), ("scale", ""), ("baseline", "n/a"), ("note", "")):
            if col not in old.columns:
                old[col] = default
            if col not in new.columns:
                new[col] = default
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


def _registry_rows_from_df(df: pd.DataFrame, ts: str, script: str = "models/rq3_narratives.py") -> list[dict]:
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "timestamp": ts,
            "script": script,
            "model": f"{r['narrative']}|{r['y']}|{r.get('x_spec', '')}|{r.get('label', '')}",
            "asset": r["asset"],
            "h": "",
            "split": "train+valid",
            "n_oos": r["n"],
            "p_raw": r["p_raw"],
            "baseline": "n/a",
            "family": r["family"],
        })
    return rows


def run_all(
    root: Path | None = None,
    out_dir: Path | None = None,
    panel: str = "title",
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    root = root or project_root()
    params = load_params(root)
    seed = int(params.get("seed", seed))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (root / "outputs" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_train_valid_A(root=root, panel=panel)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    mains = []
    for asset in ASSETS_MAIN:
        df_a = prepare_asset_frame(raw, asset)
        main = run_main_for_asset(df_a, asset)
        main.to_csv(out_dir / f"rq3_main_{asset}.csv", index=False)
        mains.append(main)
        print("%s main: %d rows; confirmed Holm<0.05: %d" % (
            asset, len(main), int(main["confirmed"].sum()),
        ))

    main_all = pd.concat(mains, ignore_index=True)
    df_btc = prepare_asset_frame(raw, "BTC")
    diag = run_diagnostics_btc(df_btc)
    robust = run_robust_btc(df_btc)
    diag.to_csv(out_dir / "rq3_diag_BTC.csv", index=False)
    robust.to_csv(out_dir / "rq3_robust_BTC.csv", index=False)

    summary = pd.concat([main_all, diag, robust], ignore_index=True)
    sum_path = out_dir / "rq3_summary.csv"
    summary.to_csv(sum_path, index=False)

    reg_path = out_dir / "test_registry.csv"
    reg_rows = (
        _registry_rows_from_df(main_all, ts)
        + _registry_rows_from_df(diag, ts)
        + _registry_rows_from_df(robust, ts)
    )
    append_registry(reg_rows, reg_path)

    meta = {
        "seed": seed,
        "spec": "A",
        "split": "train+valid",
        "nw_bandwidth": NW_BANDWIDTH,
        "narratives": list(NARRATIVES),
        "outcomes": list(OUTCOMES),
        "controls": list(CONTROL_COLS),
        "holm": "21 tests per asset (7×3); BTC primary, ETH robustness",
        "dow_note": "DOW dummies required (AbnAtt weekend seasonality; Sai khác 1)",
        "n_main": len(main_all),
        "n_diag": len(diag),
        "n_robust": len(robust),
        "n_registry_appended": len(reg_rows),
        "summary_path": str(sum_path),
    }
    (out_dir / "rq3_summary.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Summary ->", sum_path)
    print("Registry appended", len(reg_rows), "->", reg_path)
    return {"main": main_all, "diag": diag, "robust": robust, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 narratives Spec A (train+valid)")
    parser.add_argument("--panel", default="title", choices=["title", "fulltext", "nlp_text"])
    args = parser.parse_args()
    run_all(panel=args.panel)


if __name__ == "__main__":
    main()

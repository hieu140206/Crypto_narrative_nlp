"""HAR-RV baseline (Spec B) — chỉ train + valid; không đụng test.

Hai baseline:
  har_plain: y_logrv_h ~ const + har_d + har_w + har_m
  har_dow:   har_plain + 6 biến giả DOW (dayofweek UTC của t, bỏ Monday)

OLS + Newey–West/HAC (bandwidth = max(h+4, 5)).

OOS valid: rolling-origin (expanding). Tập huấn luyện tại ngày t chỉ gồm dòng
có target_end_date_h <= t và ~embargo_drop_h.

Benchmark: B0 = mean expanding y; B1 = expanding y ~ const + har_m.
Báo R²_OOS har_dow vs har_plain và vs B1; CW một phía har_dow vs har_plain.

Đổi ŷ → rv bằng exp(ŷ) — không Jensen.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402

FEATURES = ("har_d", "har_w", "har_m")
FEATURES_DOW = FEATURES + DOW_DUMMY_COLS
ASSETS = ("BTC", "ETH")
SEED = 42


def project_root() -> Path:
    return ROOT


def load_params(root: Path | None = None) -> dict:
    import yaml
    root = root or project_root()
    return yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}


def nw_bandwidth(h: int) -> int:
    return max(int(h) + 4, 5)


def target_col(h: int) -> str:
    return f"y_logrv_{h}"


def embargo_col(h: int) -> str:
    return f"embargo_drop_{h}"


def target_end_col(h: int) -> str:
    return f"target_end_date_{h}"


def prepare_xy(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Giữ hàng đủ FEATURES + target; bổ sung target_end_date_h nếu thiếu."""
    y = target_col(h)
    tend = target_end_col(h)
    cols = ["date", "asset", y, *FEATURES]
    extra = []
    if embargo_col(h) in df.columns:
        extra.append(embargo_col(h))
    if f"y_rv_{h}" in df.columns:
        extra.append(f"y_rv_{h}")
    if tend in df.columns:
        extra.append(tend)
    out = df[[c for c in cols + extra if c in df.columns]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    if tend not in out.columns:
        out[tend] = out["date"] + pd.Timedelta(days=h)
    else:
        out[tend] = pd.to_datetime(out[tend]).dt.tz_localize(None).dt.normalize()
    out = out.dropna(subset=[y, *FEATURES])
    return out.sort_values("date").reset_index(drop=True)


def expanding_train(
    history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    h: int,
) -> pd.DataFrame:
    """Tập huấn luyện cho dự báo tại ngày t.

    Điều kiện:
    1) target_end_date_h <= t  (tương đương date <= t − h): cửa sổ target của dòng
       huấn luyện phải đã hoàn tất khi phát tín hiệu dự báo — tránh dùng y chưa quan sát.
    2) ~embargo_drop_h như cũ.
    """
    t = pd.Timestamp(forecast_date).normalize()
    tend_name = target_end_col(h)
    if tend_name in history.columns:
        tend = pd.to_datetime(history[tend_name]).dt.tz_localize(None).dt.normalize()
    else:
        tend = pd.to_datetime(history["date"]).dt.tz_localize(None).dt.normalize() + pd.Timedelta(days=h)
    mask = tend <= t
    ecol = embargo_col(h)
    if ecol in history.columns:
        mask = mask & ~history[ecol].fillna(False).astype(bool)
    return history.loc[mask].copy()


def fit_ols_nw(
    df: pd.DataFrame,
    h: int,
    feature_cols: tuple[str, ...] | list[str] | None = None,
):
    """OLS + HAC Newey–West. Mặc định FEATURES (HAR); có thể thêm cột văn bản."""
    cols = list(feature_cols) if feature_cols is not None else list(FEATURES)
    y = df[target_col(h)].astype(float)
    X = sm.add_constant(df[cols].astype(float), has_constant="add")
    bw = nw_bandwidth(h)
    return sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": bw})


def fit_b1_ols(df: pd.DataFrame, h: int):
    """B1: y ~ const + har_m (điểm dự báo; không cần HAC)."""
    y = df[target_col(h)].astype(float)
    X = sm.add_constant(df[["har_m"]].astype(float), has_constant="add")
    return sm.OLS(y, X, missing="drop").fit()


def predict_row(
    res,
    row: pd.Series,
    feature_cols: tuple[str, ...] | list[str] | None = None,
) -> float:
    """Dự báo một hàng theo đúng tên tham số trong res (const + feature_cols)."""
    cols = list(feature_cols) if feature_cols is not None else list(FEATURES)
    params = res.params
    const_name = "const" if "const" in params.index else params.index[0]
    val = float(params[const_name])
    for c in cols:
        val += float(params[c]) * float(row[c])
    return val


def predict_har(res, row: pd.Series) -> float:
    """Alias: dự báo HAR thuần (const + har_d + har_w + har_m)."""
    return predict_row(res, row, FEATURES)


def predict_b1(res, row: pd.Series) -> float:
    params = res.params
    const_name = "const" if "const" in params.index else params.index[0]
    return float(params[const_name] + params["har_m"] * float(row["har_m"]))


def clark_west_pvalue(d: np.ndarray, h: int) -> tuple[float, float, float]:
    """CW một phía: H0 E[d]≤0 vs H1 E[d]>0. Trả (mean_d, t_stat, p_one_sided)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return float("nan"), float("nan"), float("nan")
    bw = nw_bandwidth(h)
    mod = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": bw})
    mean_d = float(mod.params[0])
    se = float(mod.bse[0])
    t_stat = mean_d / se if se > 0 else float("nan")
    from scipy.stats import norm
    p = float(1.0 - norm.cdf(t_stat)) if np.isfinite(t_stat) else float("nan")
    return mean_d, t_stat, p


def cw_series(y: np.ndarray, yhat_m: np.ndarray, yhat_b: np.ndarray) -> np.ndarray:
    e_m = y - yhat_m
    e_b = y - yhat_b
    return (e_b ** 2 - e_m ** 2) + (yhat_m - yhat_b) ** 2


def qlike(rv: np.ndarray, rv_hat: np.ndarray) -> np.ndarray:
    """QLIKE = log(rv_hat) + rv/rv_hat  (Patton; thấp hơn = tốt hơn)."""
    rv = np.asarray(rv, dtype=float)
    rv_hat = np.asarray(rv_hat, dtype=float)
    out = np.full_like(rv, np.nan, dtype=float)
    ok = np.isfinite(rv) & np.isfinite(rv_hat) & (rv_hat > 0) & (rv > 0)
    out[ok] = np.log(rv_hat[ok]) + rv[ok] / rv_hat[ok]
    return out


def _r2_oos(sse_model: float, sse_bench: float) -> float:
    if sse_bench <= 0:
        return float("nan")
    return 1.0 - sse_model / sse_bench


def _is_coefs(res) -> dict:
    params, bse = res.params, res.bse
    name_const = "const" if "const" in params.index else params.index[0]
    out = {
        "coef_const": float(params[name_const]),
        "se_const": float(bse[name_const]),
        "coef_har_d": float(params["har_d"]),
        "se_har_d": float(bse["har_d"]),
        "coef_har_w": float(params["har_w"]),
        "se_har_w": float(bse["har_w"]),
        "coef_har_m": float(params["har_m"]),
        "se_har_m": float(bse["har_m"]),
        "r2_is": float(res.rsquared),
        "n_train": int(res.nobs),
    }
    return out


def run_asset_horizon(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    asset: str,
    h: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Rolling OOS: har_plain, har_dow, B0, B1. Trả (oos_df, [summary_plain, summary_dow])."""
    ycol = target_col(h)
    tr = add_dow_dummies(prepare_xy(train.loc[train["asset"] == asset], h))
    va = add_dow_dummies(prepare_xy(valid.loc[valid["asset"] == asset], h))
    if tr.empty:
        raise ValueError("Train rỗng sau dropna: asset=%s h=%s" % (asset, h))

    bw = nw_bandwidth(h)
    res_plain_is = fit_ols_nw(tr, h, FEATURES)
    res_dow_is = fit_ols_nw(tr, h, FEATURES_DOW)
    plain_is = _is_coefs(res_plain_is)
    dow_is = _is_coefs(res_dow_is)

    history = pd.concat([tr, va], ignore_index=True).sort_values("date").reset_index(drop=True)
    if embargo_col(h) not in history.columns:
        history[embargo_col(h)] = False

    rows = []
    for _, row in va.iterrows():
        t = row["date"]
        train_t = expanding_train(history, t, h)
        train_t = train_t.dropna(subset=[ycol, *FEATURES_DOW])
        if len(train_t) < len(FEATURES_DOW) + 2:
            continue

        res_plain = fit_ols_nw(train_t, h, FEATURES)
        res_dow = fit_ols_nw(train_t, h, FEATURES_DOW)
        res_b1 = fit_b1_ols(train_t, h)
        yhat_plain = predict_row(res_plain, row, FEATURES)
        yhat_dow = predict_row(res_dow, row, FEATURES_DOW)
        yhat_b0 = float(train_t[ycol].mean())
        yhat_b1 = predict_b1(res_b1, row)
        y = float(row[ycol])

        if f"y_rv_{h}" in row.index and pd.notna(row.get(f"y_rv_{h}")):
            rv = float(row[f"y_rv_{h}"])
        else:
            rv = float(np.exp(y))

        e_plain = y - yhat_plain
        e_dow = y - yhat_dow
        e_b0 = y - yhat_b0
        e_b1 = y - yhat_b1

        rows.append({
            "date": t,
            "asset": asset,
            "h": h,
            "n_train_t": len(train_t),
            "train_date_max": train_t["date"].max(),
            "y_logrv": y,
            "yhat_har_plain": yhat_plain,
            "yhat_har_dow": yhat_dow,
            "yhat_b0": yhat_b0,
            "yhat_b1": yhat_b1,
            "e_har_plain": e_plain,
            "e_har_dow": e_dow,
            "e_b0": e_b0,
            "e_b1": e_b1,
            "d_cw_dow_vs_plain": float(cw_series(np.array([y]), np.array([yhat_dow]), np.array([yhat_plain]))[0]),
            "d_cw_plain_vs_b0": float(cw_series(np.array([y]), np.array([yhat_plain]), np.array([yhat_b0]))[0]),
            "d_cw_plain_vs_b1": float(cw_series(np.array([y]), np.array([yhat_plain]), np.array([yhat_b1]))[0]),
            "d_cw_dow_vs_b1": float(cw_series(np.array([y]), np.array([yhat_dow]), np.array([yhat_b1]))[0]),
            "rv": rv,
            "rv_hat_har_plain": float(np.exp(yhat_plain)),
            "rv_hat_har_dow": float(np.exp(yhat_dow)),
            "rv_hat_b0": float(np.exp(yhat_b0)),
            "rv_hat_b1": float(np.exp(yhat_b1)),
            "qlike_har_plain": float(qlike(np.array([rv]), np.array([np.exp(yhat_plain)]))[0]),
            "qlike_har_dow": float(qlike(np.array([rv]), np.array([np.exp(yhat_dow)]))[0]),
            "qlike_b0": float(qlike(np.array([rv]), np.array([np.exp(yhat_b0)]))[0]),
            "qlike_b1": float(qlike(np.array([rv]), np.array([np.exp(yhat_b1)]))[0]),
            # alias cũ
            "yhat_har": yhat_plain,
            "e_har": e_plain,
        })

    oos = pd.DataFrame(rows)
    base_meta = {
        "asset": asset,
        "h": h,
        "nw_bandwidth": bw,
        "jensen_correction": False,
    }

    def _empty_sum(baseline: str, is_coefs: dict) -> dict:
        s = {**base_meta, "baseline": baseline, **is_coefs}
        for k in (
            "n_oos", "n_train_first", "n_train_last",
            "r2_oos_vs_b0", "r2_oos_vs_b1", "r2_oos_vs_har_plain",
            "rmse", "rmse_b0", "rmse_b1", "qlike", "qlike_b0", "qlike_b1",
            "cw_pvalue_vs_b0", "cw_pvalue_vs_b1", "cw_pvalue_vs_har_plain",
            "r2_oos", "cw_pvalue_onesided",
        ):
            s[k] = 0 if k.startswith("n_") else float("nan")
        return s

    if oos.empty:
        return oos, [_empty_sum("har_plain", plain_is), _empty_sum("har_dow", dow_is)]

    sse_plain = float((oos["e_har_plain"] ** 2).sum())
    sse_dow = float((oos["e_har_dow"] ** 2).sum())
    sse_b0 = float((oos["e_b0"] ** 2).sum())
    sse_b1 = float((oos["e_b1"] ** 2).sum())
    _, _, p_dow_plain = clark_west_pvalue(oos["d_cw_dow_vs_plain"].values, h)
    _, _, p_plain_b0 = clark_west_pvalue(oos["d_cw_plain_vs_b0"].values, h)
    _, _, p_plain_b1 = clark_west_pvalue(oos["d_cw_plain_vs_b1"].values, h)
    _, _, p_dow_b1 = clark_west_pvalue(oos["d_cw_dow_vs_b1"].values, h)

    n_meta = {
        "n_oos": len(oos),
        "n_train_first": int(oos["n_train_t"].iloc[0]),
        "n_train_last": int(oos["n_train_t"].iloc[-1]),
        "rmse_b0": float(np.sqrt((oos["e_b0"] ** 2).mean())),
        "rmse_b1": float(np.sqrt((oos["e_b1"] ** 2).mean())),
        "qlike_b0": float(oos["qlike_b0"].mean()),
        "qlike_b1": float(oos["qlike_b1"].mean()),
    }

    sum_plain = {
        **base_meta,
        "baseline": "har_plain",
        **plain_is,
        **n_meta,
        "r2_oos_vs_b0": _r2_oos(sse_plain, sse_b0),
        "r2_oos_vs_b1": _r2_oos(sse_plain, sse_b1),
        "r2_oos_vs_har_plain": 0.0,
        "rmse": float(np.sqrt((oos["e_har_plain"] ** 2).mean())),
        "qlike": float(oos["qlike_har_plain"].mean()),
        "cw_pvalue_vs_b0": p_plain_b0,
        "cw_pvalue_vs_b1": p_plain_b1,
        "cw_pvalue_vs_har_plain": float("nan"),
        "r2_oos": _r2_oos(sse_plain, sse_b0),
        "cw_pvalue_onesided": p_plain_b0,
        "rmse_har": float(np.sqrt((oos["e_har_plain"] ** 2).mean())),
        "qlike_har": float(oos["qlike_har_plain"].mean()),
    }
    sum_dow = {
        **base_meta,
        "baseline": "har_dow",
        **dow_is,
        **n_meta,
        "r2_oos_vs_b0": _r2_oos(sse_dow, sse_b0),
        "r2_oos_vs_b1": _r2_oos(sse_dow, sse_b1),
        "r2_oos_vs_har_plain": _r2_oos(sse_dow, sse_plain),
        "rmse": float(np.sqrt((oos["e_har_dow"] ** 2).mean())),
        "qlike": float(oos["qlike_har_dow"].mean()),
        "cw_pvalue_vs_b0": float("nan"),
        "cw_pvalue_vs_b1": p_dow_b1,
        "cw_pvalue_vs_har_plain": p_dow_plain,
        "r2_oos": _r2_oos(sse_dow, sse_b0),
        "cw_pvalue_onesided": p_dow_plain,
        "rmse_har": float(np.sqrt((oos["e_har_dow"] ** 2).mean())),
        "qlike_har": float(oos["qlike_har_dow"].mean()),
    }
    return oos, [sum_plain, sum_dow]


def load_train_valid(h: int, root: Path | None = None, panel: str = "title"):
    """Chỉ qua load_model_table; drop embargo theo h."""
    root = root or project_root()
    train = load_model_table("B", "train", horizon=h, drop_embargo=True, root=root, panel=panel)
    valid = load_model_table("B", "valid", horizon=h, drop_embargo=True, root=root, panel=panel)
    return train, valid


def run_all(
    horizons: list[int] | None = None,
    assets: tuple[str, ...] = ASSETS,
    root: Path | None = None,
    out_dir: Path | None = None,
    panel: str = "title",
    seed: int = SEED,
) -> pd.DataFrame:
    root = root or project_root()
    params = load_params(root)
    horizons = horizons or [int(x) for x in params.get("horizons", [1, 3, 7])]
    seed = int(params.get("seed", seed))
    random.seed(seed)
    np.random.seed(seed)

    out_dir = out_dir or (root / "outputs" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for h in horizons:
        train, valid = load_train_valid(h, root=root, panel=panel)
        for asset in assets:
            oos, sums = run_asset_horizon(train, valid, asset, h)
            oos_path = out_dir / f"har_baseline_{asset}_h{h}.csv"
            oos.to_csv(oos_path, index=False)
            summaries.extend(sums)
            for summary in sums:
                print(
                    "%s h=%d [%s] | n_train=%d R2_IS=%.4f | n_oos=%d "
                    "R2vsB0=%.4f R2vsB1=%.4f R2vsPlain=%.4f | "
                    "CW_vs_plain=%.4g CW_vs_B1=%.4g | %s"
                    % (
                        asset, h, summary["baseline"],
                        summary["n_train"], summary["r2_is"], summary["n_oos"],
                        summary.get("r2_oos_vs_b0", float("nan")),
                        summary.get("r2_oos_vs_b1", float("nan")),
                        summary.get("r2_oos_vs_har_plain", float("nan")),
                        summary.get("cw_pvalue_vs_har_plain", float("nan")),
                        summary.get("cw_pvalue_vs_b1", float("nan")),
                        oos_path.name,
                    )
                )

    summary_df = pd.DataFrame(summaries)
    sum_path = out_dir / "har_baseline_summary.csv"
    summary_df.to_csv(sum_path, index=False)
    meta = {
        "seed": seed,
        "panel": panel,
        "baselines": ["har_plain", "har_dow"],
        "dow": "6 dummies Tue-Sun from date t (Mon omitted); models.calendar_features",
        "jensen_correction": False,
        "note_rv": "rv_hat=exp(yhat_logrv); không +σ²/2",
        "train_rule": "target_end_date_h <= t and ~embargo_drop_h",
        "benchmarks": "B0=expanding mean y; B1=expanding const+har_m",
        "clark_west": "har_dow vs har_plain; har_* vs B1; one-sided nested HAC",
        "summary_path": str(sum_path),
    }
    (out_dir / "har_baseline_summary.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Summary ->", sum_path)
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="HAR-RV baseline Spec B (train+valid only)")
    parser.add_argument("--panel", default="title", choices=["title", "fulltext", "nlp_text"])
    args = parser.parse_args()
    run_all(panel=args.panel)


if __name__ == "__main__":
    main()

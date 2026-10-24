"""Họ 5 — Elastic Net + XGBoost (Spec B), bản market vs market+text.

Chỉ thiết kế + CV/refit trên dữ liệu đưa vào; đánh giá thật (valid/test)
gọi qua run_evaluation(stage=...). stage='test' bị chặn nếu thiếu ALLOW_TEST=1.

Siêu tham số: valid → CV trên train; test → CV trên train+valid.
Rolling-origin: target_end_date_h <= t và ~embargo_drop_h.
EN: Clark–West một phía (lồng nhau). XGB: DM–HLN một phía trên (e_mkt² − e_txt²).
Holm 6 phép (2 mô hình × 3 h) / tài sản.
Seed / random_state = 42. Lưới từ config/params.yaml khối ml: — không mở rộng.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import TestSplitLockedError, load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.har_baseline import (  # noqa: E402
    FEATURES,
    FEATURES_DOW,
    clark_west_pvalue,
    cw_series,
    fit_ols_nw,
    nw_bandwidth,
    predict_row,
    qlike,
    _r2_oos,
)
from models.rq3_narratives import holm_adjust, project_root  # noqa: E402

SEED = 42
MACRO_LEVEL_COLS = ("fred_VIXCLS", "fred_DGS10", "fred_DTWEXBGS")
GRID_KEYS = {"v1": "ml", "v2": "ml_v2"}

try:
    import xgboost as xgb
    XGB_BACKEND = "xgboost"
except ImportError:  # pragma: no cover
    xgb = None
    try:
        import lightgbm as lgb  # noqa: F401
        XGB_BACKEND = "lightgbm"
    except ImportError:
        XGB_BACKEND = None


def load_params(root: Path | None = None) -> dict:
    root = root or project_root()
    return yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}


def params_sha256(root: Path | None = None) -> str:
    root = root or project_root()
    raw = (root / "config" / "params.yaml").read_bytes()
    return hashlib.sha256(raw).hexdigest()


def ml_config(
    params: dict | None = None,
    *,
    grid: str = "v1",
) -> dict:
    """Lấy khối ml (v1) hoặc ml_v2. Mặc định grid='v1' = hành vi cũ."""
    params = params or load_params()
    grid = str(grid).lower()
    if grid not in GRID_KEYS:
        raise ValueError("grid ∈ {v1,v2}, nhận %r" % grid)
    key = GRID_KEYS[grid]
    ml = dict(params.get(key) or {})
    if not ml:
        raise KeyError("params.yaml thiếu khối %s:" % key)
    ml["_embargo_days"] = int(params.get("split", {}).get("embargo_days", 7))
    ml["_horizons"] = [int(h) for h in params.get("horizons", [1, 3, 7])]
    ml["_seed"] = int(params.get("seed", SEED))
    ml["_grid_version"] = grid
    return ml


def feature_lists(
    ml: dict | None = None,
    *,
    macro_mode: str = "level",
) -> tuple[list[str], list[str]]:
    """macro_mode='level': đủ FRED mức+_d5; 'diff': bỏ 3 cột mức, giữ _d5."""
    ml = ml or ml_config()
    macro_mode = str(macro_mode).lower()
    if macro_mode not in ("level", "diff"):
        raise ValueError("macro_mode ∈ {level,diff}, nhận %r" % macro_mode)
    market = list(ml["features_market"])
    if macro_mode == "diff":
        market = [c for c in market if c not in MACRO_LEVEL_COLS]
    text = list(ml["features_text"])
    full = market + [c for c in text if c not in market]
    return market, full


def output_tag(grid: str = "v1", macro_mode: str = "level") -> str:
    """Hậu tố tên file; rỗng khi v1+level (tương thích bản chạy đầu)."""
    if grid == "v1" and macro_mode == "level":
        return ""
    return f"_g{grid}_m{macro_mode}"


def append_ml_registry(
    rows: list[dict],
    path: Path,
    *,
    grid_version: str,
    macro_mode: str,
) -> None:
    """Append registry; thêm cột grid_version/macro_mode; backfill dòng ML cũ = v1/level."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    for r in rows:
        rr = dict(r)
        rr["grid_version"] = grid_version
        rr["macro_mode"] = macro_mode
        enriched.append(rr)
    new = pd.DataFrame(enriched)
    if path.exists():
        old = pd.read_csv(path)
        for col, default in (("family", ""), ("scale", ""), ("baseline", "n/a"), ("note", "")):
            if col not in old.columns:
                old[col] = default
        if "grid_version" not in old.columns:
            old["grid_version"] = ""
            ml_mask = old["script"].astype(str).str.contains("ml_forecast", na=False)
            old.loc[ml_mask, "grid_version"] = "v1"
        if "macro_mode" not in old.columns:
            old["macro_mode"] = ""
            ml_mask = old["script"].astype(str).str.contains("ml_forecast", na=False)
            old.loc[ml_mask, "macro_mode"] = "level"
        for col in old.columns:
            if col not in new.columns:
                new[col] = ""
        for col in new.columns:
            if col not in old.columns:
                old[col] = ""
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(path, index=False)


def target_col(h: int) -> str:
    return f"y_logrv_{h}"


def embargo_col(h: int) -> str:
    return f"embargo_drop_{h}"


def target_end_col(h: int) -> str:
    return f"target_end_date_{h}"


def prepare_ml_frame(df: pd.DataFrame, h: int, feature_cols: list[str]) -> pd.DataFrame:
    """Gắn DOW nếu thiếu; chuẩn hoá date/target_end; không dropna ở đây."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    if not set(DOW_DUMMY_COLS).issubset(out.columns):
        out = add_dow_dummies(out)
    y = target_col(h)
    tend = target_end_col(h)
    if tend not in out.columns:
        out[tend] = out["date"] + pd.Timedelta(days=h)
    else:
        out[tend] = pd.to_datetime(out[tend]).dt.tz_localize(None).dt.normalize()
    if embargo_col(h) not in out.columns:
        out[embargo_col(h)] = False
    return out.sort_values("date").reset_index(drop=True)


def common_complete_mask(df: pd.DataFrame, h: int, cols_market: list[str], cols_full: list[str]) -> pd.Series:
    """Mẫu chung: y + mọi cột full (text) không NaN → áp cho CẢ HAI bản."""
    y = target_col(h)
    need = [y, *cols_full]
    return df[need].notna().all(axis=1)


# ---------------------------------------------------------------------------
# Expanding-window CV
# ---------------------------------------------------------------------------

def expanding_cv_folds(
    dates: pd.Series,
    target_ends: pd.Series,
    *,
    n_folds: int = 5,
    embargo_days: int = 7,
) -> list[tuple[np.ndarray, np.ndarray, pd.Timestamp]]:
    """Trả list (train_idx, valid_idx, valid_start).

    Train chỉ gồm dòng có target_end <= valid_start − 0 và cách valid bằng embargo:
    target_end <= valid_start - embargo_days (an toàn hơn: date_train max + embargo < valid_start
    và target_end <= valid_start).
    """
    dates = pd.to_datetime(dates).dt.tz_localize(None).dt.normalize()
    target_ends = pd.to_datetime(target_ends).dt.tz_localize(None).dt.normalize()
    order = np.argsort(dates.to_numpy())
    d_sorted = dates.iloc[order].reset_index(drop=True)
    n = len(d_sorted)
    if n < n_folds + 10:
        raise ValueError("Quá ít quan sát cho CV: n=%d folds=%d" % (n, n_folds))

    # cắt điểm validation theo quantile thời gian (expanding)
    # fold k: valid = segment k; train = mọi thứ trước valid_start − embargo
    cuts = np.linspace(0.4, 1.0, n_folds + 1)  # phần đầu >=40% làm train tối thiểu
    folds = []
    for i in range(n_folds):
        v0 = int(cuts[i] * n)
        v1 = int(cuts[i + 1] * n)
        if v1 <= v0:
            continue
        valid_pos = np.arange(v0, v1)
        valid_start = pd.Timestamp(d_sorted.iloc[v0])
        # map back to original indices
        valid_idx = order[valid_pos]
        # train: date < valid_start - embargo AND target_end <= valid_start
        cutoff = valid_start - pd.Timedelta(days=int(embargo_days))
        train_mask = (dates.to_numpy() <= cutoff.to_datetime64()) & (
            target_ends.to_numpy() <= valid_start.to_datetime64()
        )
        train_idx = np.where(train_mask)[0]
        # loại mọi dòng overlap valid
        train_idx = np.setdiff1d(train_idx, valid_idx, assume_unique=False)
        if len(train_idx) < 20 or len(valid_idx) < 5:
            continue
        folds.append((train_idx, valid_idx, valid_start))
    if not folds:
        raise ValueError("Không tạo được fold CV hợp lệ")
    return folds


def assert_cv_no_leak(folds: list, dates: pd.Series, target_ends: pd.Series) -> None:
    dates = pd.to_datetime(dates).dt.normalize()
    target_ends = pd.to_datetime(target_ends).dt.normalize()
    for train_idx, valid_idx, valid_start in folds:
        te = target_ends.iloc[train_idx]
        assert (te <= valid_start).all(), "CV leak: target_end > valid_start"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _en_grid(ml: dict) -> list[dict]:
    cfg = ml["elastic_net"]
    return [
        {"alpha": a, "l1_ratio": r}
        for a, r in itertools.product(cfg["alpha"], cfg["l1_ratio"])
    ]


def _as_list(x) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _xgb_grid(ml: dict) -> list[dict]:
    cfg = ml["xgboost"]
    grid = []
    for depth, nest, lr, mcw in itertools.product(
        _as_list(cfg["max_depth"]),
        _as_list(cfg["n_estimators"]),
        _as_list(cfg["learning_rate"]),
        _as_list(cfg["min_child_weight"]),
    ):
        grid.append({
            "max_depth": int(depth),
            "n_estimators": int(nest),
            "learning_rate": float(lr),
            "subsample": float(cfg["subsample"]),
            "colsample_bytree": float(cfg["colsample_bytree"]),
            "min_child_weight": float(mcw),
            "reg_lambda": float(cfg["reg_lambda"]),
            "tree_method": str(cfg["tree_method"]),
            "n_jobs": int(cfg["n_jobs"]),
            "random_state": int(cfg["random_state"]),
        })
    return grid


class ScaledElasticNet:
    """ElasticNet với StandardScaler fit CHỈ trên train."""

    def __init__(self, alpha: float = 0.01, l1_ratio: float = 0.5, max_iter: int = 5000):
        self.alpha = float(alpha)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=self.alpha, l1_ratio=self.l1_ratio,
            max_iter=self.max_iter, random_state=SEED,
        )
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.coef_ = self.model.coef_.copy()
        self.intercept_ = float(self.model.intercept_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(X))


class XGBWrapper:
    """XGBoost (hoặc báo lỗi nếu thiếu)."""

    def __init__(self, **params):
        if XGB_BACKEND != "xgboost" or xgb is None:
            raise ImportError("Cần xgboost (backend=%s)" % XGB_BACKEND)
        self.params = dict(params)
        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            verbosity=0,
            **self.params,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


def fit_predict_mse(model_factory, X_tr, y_tr, X_va, y_va) -> float:
    m = model_factory()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_va)
    return float(np.mean((y_va - pred) ** 2))


def select_hyperparams_cv(
    df: pd.DataFrame,
    h: int,
    feature_cols: list[str],
    model_kind: str,
    ml: dict | None = None,
) -> dict:
    """Chọn siêu tham số bằng MSE trung bình qua expanding CV folds."""
    ml = ml or ml_config()
    ycol = target_col(h)
    tend = target_end_col(h)
    folds = expanding_cv_folds(
        df["date"], df[tend],
        n_folds=int(ml["cv"]["n_folds"]),
        embargo_days=int(ml["_embargo_days"]),
    )
    assert_cv_no_leak(folds, df["date"], df[tend])

    if model_kind == "elastic_net":
        grid = _en_grid(ml)
        def factory(hp):
            return lambda: ScaledElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"])
    elif model_kind == "xgboost":
        grid = _xgb_grid(ml)
        def factory(hp):
            return lambda: XGBWrapper(**hp)
    else:
        raise ValueError(model_kind)

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[ycol].to_numpy(dtype=float)
    best_hp, best_mse = None, float("inf")
    for hp in grid:
        mses = []
        fac = factory(hp)
        for tr_idx, va_idx, _ in folds:
            mse = fit_predict_mse(fac, X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx])
            mses.append(mse)
        mean_mse = float(np.mean(mses))
        if mean_mse < best_mse:
            best_mse, best_hp = mean_mse, hp
    return {"hyperparams": best_hp, "cv_mse": best_mse, "n_folds": len(folds), "model_kind": model_kind}


# ---------------------------------------------------------------------------
# Rolling-origin evaluation (thiết kế — gọi khi chạy thật)
# ---------------------------------------------------------------------------

def expanding_train_mask(df: pd.DataFrame, t: pd.Timestamp, h: int) -> np.ndarray:
    t = pd.Timestamp(t).normalize()
    tend = pd.to_datetime(df[target_end_col(h)]).dt.normalize()
    ecol = embargo_col(h)
    mask = (tend <= t) & (~df[ecol].fillna(False).astype(bool))
    return mask.to_numpy()


def should_refit(i: int, every: int) -> bool:
    """i = 0-based index trên chuỗi ngày đánh giá; refit khi i % every == 0."""
    every = max(1, int(every))
    return (i % every) == 0


def diebold_mariano_hln(
    e_model: np.ndarray,
    e_bench: np.ndarray,
    h: int,
) -> tuple[float, float]:
    """DM một phía H1: model tốt hơn bench (E[e_b² − e_m²] > 0), HLN + HAC."""
    import statsmodels.api as sm
    from scipy.stats import t as student_t

    d = np.asarray(e_bench, dtype=float) ** 2 - np.asarray(e_model, dtype=float) ** 2
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 5:
        return float("nan"), float("nan")
    bw = nw_bandwidth(h)
    mod = sm.OLS(d, np.ones((T, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": bw})
    mean_d = float(mod.params[0])
    se = float(mod.bse[0])
    dm = mean_d / se if se > 0 else float("nan")
    # HLN small-sample correction
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * hln if np.isfinite(dm) else float("nan")
    p = float(1.0 - student_t.cdf(dm_star, df=T - 1)) if np.isfinite(dm_star) else float("nan")
    return dm_star, p


def evaluate_pair_rolling(
    history: pd.DataFrame,
    h: int,
    cols_market: list[str],
    cols_full: list[str],
    model_kind: str,
    hp_market: dict,
    hp_text: dict,
    ml: dict | None = None,
    *,
    eval_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rolling OOS: market vs text trên mẫu chung. EN refit mỗi ngày; XGB mỗi 20 ngày.

    ``history`` = train (+valid nếu stage=test) ∪ eval — dùng cho expanding_train.
    ``eval_df`` = ngày OOS (valid hoặc test). Nếu None: eval = toàn bộ ``history``
    (tương thích test synth cũ).
    Đồng thời fit HAR+DOW OLS (cùng giao thức train) làm tham chiếu.
    """
    ml = ml or ml_config()
    every = (
        int(ml["refit"]["elastic_net_every_days"])
        if model_kind == "elastic_net"
        else int(ml["refit"]["xgboost_every_days"])
    )
    ycol = target_col(h)
    har_cols = list(FEATURES_DOW)
    df = history.sort_values("date").reset_index(drop=True)
    missing_har = [c for c in har_cols if c not in df.columns]
    if missing_har:
        raise KeyError("Thiếu cột HAR+DOW: %s" % missing_har)
    if eval_df is None:
        eval_dates = sorted(df["date"].unique())
    else:
        eval_dates = sorted(pd.to_datetime(eval_df["date"]).dt.normalize().unique())
    rows = []
    model_m = model_t = None
    har_res = None
    last_coef = None
    last_xgb_gain = None

    def make(kind, hp):
        if kind == "elastic_net":
            return ScaledElasticNet(alpha=hp["alpha"], l1_ratio=hp["l1_ratio"])
        return XGBWrapper(**hp)

    for i, t in enumerate(eval_dates):
        t = pd.Timestamp(t).normalize()
        row = df.loc[df["date"] == t]
        if row.empty:
            continue
        row = row.iloc[0]
        # cần đủ feature cho ngày gốc
        if row[cols_full + [ycol]].isna().any():
            continue
        tr_mask = expanding_train_mask(df, t, h)
        # loại chính ngày t khỏi train
        tr_mask = tr_mask & (df["date"] < t)
        # cần đủ HAR + full features trên train
        feat_ok = df[cols_full + har_cols + [ycol]].notna().all(axis=1).to_numpy()
        tr_ok = tr_mask & feat_ok
        if tr_ok.sum() < len(cols_full) + 5:
            continue
        did_refit = bool(should_refit(i, every) or model_m is None)
        if did_refit:
            Xtr_m = df.loc[tr_ok, cols_market].to_numpy(dtype=float)
            Xtr_t = df.loc[tr_ok, cols_full].to_numpy(dtype=float)
            ytr = df.loc[tr_ok, ycol].to_numpy(dtype=float)
            model_m = make(model_kind, hp_market).fit(Xtr_m, ytr)
            model_t = make(model_kind, hp_text).fit(Xtr_t, ytr)
            if model_kind == "elastic_net":
                last_coef = {
                    "market": dict(zip(cols_market, model_m.coef_.tolist())),
                    "text": dict(zip(cols_full, model_t.coef_.tolist())),
                }
            elif model_kind == "xgboost":
                booster = model_t.model.get_booster()
                score = booster.get_score(importance_type="gain")
                mapped = {}
                for k, v in score.items():
                    if k.startswith("f") and k[1:].isdigit():
                        idx = int(k[1:])
                        if idx < len(cols_full):
                            mapped[cols_full[idx]] = float(v)
                    else:
                        mapped[k] = float(v)
                last_xgb_gain = mapped
        # HAR+DOW OLS refit mỗi ngày gốc (nhẹ)
        har_res = fit_ols_nw(df.loc[tr_ok], h, har_cols)

        Xm = row[cols_market].to_numpy(dtype=float).reshape(1, -1)
        Xt = row[cols_full].to_numpy(dtype=float).reshape(1, -1)
        y = float(row[ycol])
        yhat_m = float(model_m.predict(Xm)[0])
        yhat_t = float(model_t.predict(Xt)[0])
        yhat_har = float(predict_row(har_res, row, har_cols))
        rv = float(np.exp(y))
        rows.append({
            "date": t,
            "h": h,
            "n_train": int(tr_ok.sum()),
            "refit": did_refit,
            "origin_i": int(i),
            "y_logrv": y,
            "yhat_market": yhat_m,
            "yhat_text": yhat_t,
            "yhat_har_dow": yhat_har,
            "e_market": y - yhat_m,
            "e_text": y - yhat_t,
            "e_har_dow": y - yhat_har,
            "rv": rv,
            "rv_hat_market": float(np.exp(yhat_m)),
            "rv_hat_text": float(np.exp(yhat_t)),
            "rv_hat_har_dow": float(np.exp(yhat_har)),
            "qlike_market": float(qlike(np.array([rv]), np.array([np.exp(yhat_m)]))[0]),
            "qlike_text": float(qlike(np.array([rv]), np.array([np.exp(yhat_t)]))[0]),
            "qlike_har_dow": float(qlike(np.array([rv]), np.array([np.exp(yhat_har)]))[0]),
            "d_cw": float(cw_series(np.array([y]), np.array([yhat_t]), np.array([yhat_m]))[0]),
        })
    oos = pd.DataFrame(rows)
    oos.attrs["last_coef"] = last_coef
    oos.attrs["last_xgb_gain"] = last_xgb_gain
    return oos


def summarize_oos(oos: pd.DataFrame, h: int, model_kind: str) -> dict:
    if oos.empty:
        return {"n_oos": 0}
    sse_t = float((oos["e_text"] ** 2).sum())
    sse_m = float((oos["e_market"] ** 2).sum())
    sse_h = float((oos["e_har_dow"] ** 2).sum())
    r2 = _r2_oos(sse_t, sse_m)
    if model_kind == "elastic_net":
        _, _, p = clark_west_pvalue(oos["d_cw"].values, h)
        test_name = "clark_west"
    else:
        _, p = diebold_mariano_hln(oos["e_text"].values, oos["e_market"].values, h)
        test_name = "dm_hln"
    return {
        "n_oos": len(oos),
        "r2_oos_vs_market": r2,
        "r2_oos_text_vs_har_dow": _r2_oos(sse_t, sse_h),
        "r2_oos_market_vs_har_dow": _r2_oos(sse_m, sse_h),
        "rmse_text": float(np.sqrt((oos["e_text"] ** 2).mean())),
        "rmse_market": float(np.sqrt((oos["e_market"] ** 2).mean())),
        "rmse_har_dow": float(np.sqrt((oos["e_har_dow"] ** 2).mean())),
        "qlike_text": float(oos["qlike_text"].mean()),
        "qlike_market": float(oos["qlike_market"].mean()),
        "qlike_har_dow": float(oos["qlike_har_dow"].mean()),
        "p_raw": p,
        "test": test_name,
    }


def load_stage_frames(
    stage: str,
    h: int,
    *,
    root: Path | None = None,
    panel: str = "title",
) -> pd.DataFrame:
    """stage='valid': train+valid cho CV-select trên train, eval trên valid.
    stage='test': yêu cầu ALLOW_TEST=1.
    """
    root = root or project_root()
    stage = str(stage).lower()
    if stage == "test":
        # load_model_table sẽ raise nếu thiếu ALLOW_TEST
        train = load_model_table("B", "train", horizon=h, drop_embargo=True, root=root, panel=panel)
        valid = load_model_table("B", "valid", horizon=h, drop_embargo=True, root=root, panel=panel)
        test = load_model_table("B", "test", horizon=h, drop_embargo=True, root=root, panel=panel)
        return pd.concat([train, valid, test], ignore_index=True)
    if stage != "valid":
        raise ValueError("stage ∈ {'valid','test'}, nhận %r" % stage)
    train = load_model_table("B", "train", horizon=h, drop_embargo=True, root=root, panel=panel)
    valid = load_model_table("B", "valid", horizon=h, drop_embargo=True, root=root, panel=panel)
    return pd.concat([train, valid], ignore_index=True)


def run_evaluation(
    stage: str = "valid",
    *,
    root: Path | None = None,
    out_dir: Path | None = None,
    assets: list[str] | None = None,
    horizons: list[int] | None = None,
    panel: str = "title",
    register: bool = True,
    grid: str = "v1",
    macro_mode: str = "level",
) -> pd.DataFrame:
    """Chạy đánh giá (valid hoặc test). Không gọi mặc định từ main."""
    root = root or project_root()
    params = load_params(root)
    grid = str(grid).lower()
    macro_mode = str(macro_mode).lower()
    ml = ml_config(params, grid=grid)
    seed = int(ml["_seed"])
    random.seed(seed)
    np.random.seed(seed)

    if stage == "test" and os.environ.get("ALLOW_TEST") != "1":
        raise TestSplitLockedError("stage='test' cần ALLOW_TEST=1")

    cols_m, cols_f = feature_lists(ml, macro_mode=macro_mode)
    assets = assets or list(ml["assets"])
    horizons = horizons or list(ml["_horizons"])
    out_dir = out_dir or (root / "outputs" / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tag = output_tag(grid, macro_mode)

    summaries = []
    reg_rows = []
    for asset in assets:
        asset_rows = []
        for h in horizons:
            raw = load_stage_frames(stage, h, root=root, panel=panel)
            raw = raw.loc[raw["asset"] == asset].copy()
            raw = prepare_ml_frame(raw, h, cols_f)
            mask = common_complete_mask(raw, h, cols_m, cols_f)
            df = raw.loc[mask].reset_index(drop=True)

            # CV pool: train (+valid nếu stage=test)
            if stage == "valid":
                cv_pool = df.loc[df["split"] == "train"].reset_index(drop=True)
                eval_df = df.loc[df["split"] == "valid"].reset_index(drop=True)
            else:
                cv_pool = df.loc[df["split"].isin(["train", "valid"])].reset_index(drop=True)
                eval_df = df.loc[df["split"] == "test"].reset_index(drop=True)

            for kind in ("elastic_net", "xgboost"):
                print("CV+eval", asset, "h=", h, kind, "grid=", grid, "macro=", macro_mode, flush=True)
                sel_m = select_hyperparams_cv(cv_pool, h, cols_m, kind, ml=ml)
                sel_t = select_hyperparams_cv(cv_pool, h, cols_f, kind, ml=ml)
                # registry CV
                for sel, tag_cv in ((sel_m, "market"), (sel_t, "text")):
                    reg_rows.append({
                        "timestamp": ts,
                        "script": "models/ml_forecast.py",
                        "model": f"{kind}|{tag_cv}|h{h}|cv",
                        "asset": asset,
                        "h": h,
                        "split": "train" if stage == "valid" else "train+valid",
                        "n_oos": sel["n_folds"],
                        "p_raw": "",
                        "baseline": "n/a",
                        "family": "ml_cv",
                        "scale": "",
                        "note": json.dumps({
                            "hyperparams": sel["hyperparams"],
                            "cv_mse": sel["cv_mse"],
                            "n_folds": sel["n_folds"],
                        }, sort_keys=True),
                    })
                oos = evaluate_pair_rolling(
                    df, h, cols_m, cols_f, kind,
                    sel_m["hyperparams"], sel_t["hyperparams"], ml=ml,
                    eval_df=eval_df,
                )
                summ = summarize_oos(oos, h, kind)
                detail = {
                    "asset": asset,
                    "h": h,
                    "model_kind": kind,
                    "stage": stage,
                    "grid_version": grid,
                    "macro_mode": macro_mode,
                    "hp_market": sel_m["hyperparams"],
                    "hp_text": sel_t["hyperparams"],
                    "cv_mse_market": sel_m["cv_mse"],
                    "cv_mse_text": sel_t["cv_mse"],
                    "n_folds": sel_m["n_folds"],
                    "features_market": cols_m,
                    "features_text": [c for c in cols_f if c not in cols_m],
                }
                if kind == "elastic_net" and oos.attrs.get("last_coef"):
                    coef_t = oos.attrs["last_coef"]["text"]
                    text_only = {c: coef_t[c] for c in detail["features_text"] if c in coef_t}
                    nonzero = {c: v for c, v in coef_t.items() if abs(v) > 1e-12}
                    text_zero = [c for c, v in text_only.items() if abs(v) <= 1e-12]
                    detail["en_coef_all_text_model"] = coef_t
                    detail["en_coef_text_vars"] = text_only
                    detail["en_n_nonzero"] = len(nonzero)
                    detail["en_text_zeroed"] = text_zero
                if kind == "xgboost" and oos.attrs.get("last_xgb_gain"):
                    gain = oos.attrs["last_xgb_gain"]
                    top10 = sorted(gain.items(), key=lambda kv: -kv[1])[:10]
                    detail["xgb_gain"] = gain
                    detail["xgb_gain_top10"] = top10

                (out_dir / f"ml_detail_{kind}_{asset}_h{h}_{stage}{tag}.json").write_text(
                    json.dumps(detail, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8",
                )
                row = {
                    "asset": asset,
                    "h": h,
                    "model_kind": kind,
                    "stage": stage,
                    "grid_version": grid,
                    "macro_mode": macro_mode,
                    "hp_market": json.dumps(sel_m["hyperparams"], sort_keys=True),
                    "hp_text": json.dumps(sel_t["hyperparams"], sort_keys=True),
                    "cv_mse_market": sel_m["cv_mse"],
                    "cv_mse_text": sel_t["cv_mse"],
                    "n_folds": sel_m["n_folds"],
                    **summ,
                }
                if kind == "elastic_net" and "en_n_nonzero" in detail:
                    row["en_n_nonzero"] = detail["en_n_nonzero"]
                    row["en_n_text_zeroed"] = len(detail["en_text_zeroed"])
                asset_rows.append(row)
                summaries.append(row)
                oos.to_csv(out_dir / f"ml_{kind}_{asset}_h{h}_{stage}{tag}.csv", index=False)
                reg_rows.append({
                    "timestamp": ts,
                    "script": "models/ml_forecast.py",
                    "model": f"{kind}|text_vs_market|h{h}",
                    "asset": asset,
                    "h": h,
                    "split": stage,
                    "n_oos": summ.get("n_oos", 0),
                    "p_raw": summ.get("p_raw", ""),
                    "baseline": "market_ml",
                    "family": "ml_main",
                    "scale": "",
                    "note": summ.get("test", ""),
                })
                print(
                    "  done %s %s h=%d n_oos=%d R2=%.4f p=%.4g"
                    % (asset, kind, h, summ.get("n_oos", 0),
                       summ.get("r2_oos_vs_market", float("nan")),
                       summ.get("p_raw", float("nan"))),
                    flush=True,
                )

        # Holm 6 phép / asset (2 mô hình × 3 h)
        asset_summ = [r for r in summaries if r["asset"] == asset]
        if asset_summ:
            # ổn định thứ tự: elastic_net then xgboost × h
            asset_summ_sorted = sorted(
                asset_summ, key=lambda r: (0 if r["model_kind"] == "elastic_net" else 1, r["h"])
            )
            adj = holm_adjust([float(r["p_raw"]) for r in asset_summ_sorted])
            for r, p_h in zip(asset_summ_sorted, adj):
                r["p_holm"] = p_h

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / f"ml_summary_{stage}{tag}.csv", index=False)
    if register:
        append_ml_registry(
            reg_rows, out_dir / "test_registry.csv",
            grid_version=grid, macro_mode=macro_mode,
        )
    (out_dir / f"ml_summary_{stage}{tag}.meta.json").write_text(
        json.dumps({
            "stage": stage,
            "grid_version": grid,
            "macro_mode": macro_mode,
            "seed": seed,
            "backend": XGB_BACKEND,
            "params_sha256": params_sha256(root),
            "features_market": cols_m,
            "features_full": cols_f,
        }, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary_df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ML forecast họ 5 (mặc định chỉ in config)")
    parser.add_argument("--run", choices=["valid", "test"], default=None,
                        help="Chạy đánh giá thật (mặc định: không chạy, chỉ in ml+hash)")
    parser.add_argument("--grid", choices=["v1", "v2"], default="v1",
                        help="Lưới siêu tham số (mặc định v1 = khối ml:)")
    parser.add_argument("--macro_mode", choices=["level", "diff"], default="level",
                        help="FRED mức+_d5 (level) hoặc chỉ _d5 (diff); mặc định level")
    return parser


def print_ml_block_and_hash(root: Path | None = None) -> None:
    root = root or project_root()
    params = load_params(root)
    print("=== config/params.yaml :: ml (v1) ===")
    print(yaml.safe_dump({"ml": params.get("ml")}, sort_keys=False, allow_unicode=True))
    if params.get("ml_v2"):
        print("=== config/params.yaml :: ml_v2 ===")
        print(yaml.safe_dump({"ml_v2": params.get("ml_v2")}, sort_keys=False, allow_unicode=True))
    print("sha256(config/params.yaml) =", params_sha256(root))
    print("xgboost_backend =", XGB_BACKEND)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    print_ml_block_and_hash()
    if args.run is None:
        print("DỪNG — chưa chạy mô hình trên dữ liệu thật (truyền --run valid|test để chạy).")
        return
    run_evaluation(stage=args.run, grid=args.grid, macro_mode=args.macro_mode)


if __name__ == "__main__":
    main()

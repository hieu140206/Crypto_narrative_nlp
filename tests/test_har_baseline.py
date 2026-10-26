"""Test HAR-RV baseline — rò rỉ target, DOW, B0/B1, tái lập, chốt test."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import TestSplitLockedError, load_model_table  # noqa: E402
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies, dow_dummy_frame  # noqa: E402
from models.har_baseline import (  # noqa: E402
    expanding_train,
    fit_ols_nw,
    prepare_xy,
    run_all,
    run_asset_horizon,
    target_end_col,
)


def test_load_test_locked_without_allow(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("B", "test", horizon=1, root=ROOT)


def test_har_baseline_rejects_test_split_env(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("B", "test", horizon=1, drop_embargo=True, root=ROOT)


def _synthetic_har(n_train=200, n_valid=40, seed=42, h=1, weekend_effect=0.0):
    rng = np.random.default_rng(seed)
    n = n_train + n_valid
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    har_d = rng.normal(-5.0, 0.5, n)
    har_w = har_d + rng.normal(0, 0.05, n)
    har_m = har_d + rng.normal(0, 0.08, n)
    eps = rng.normal(0, 0.02, n)
    y = 0.1 + 0.5 * har_d + 0.3 * har_w + 0.2 * har_m + eps
    if weekend_effect:
        dow = dates.dayofweek
        y = y + weekend_effect * ((dow >= 5).astype(float))
    embargo = np.zeros(n, dtype=bool)
    embargo[n_train - 3 : n_train] = True
    ycol = f"y_logrv_{h}"
    ecol = f"embargo_drop_{h}"
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "har_d": har_d,
        "har_w": har_w,
        "har_m": har_m,
        ycol: y,
        f"y_rv_{h}": np.exp(y),
        ecol: embargo,
        f"target_end_date_{h}": dates + pd.Timedelta(days=h),
        "split": ["train"] * n_train + ["valid"] * n_valid,
    })
    return df[df["split"] == "train"].copy(), df[df["split"] == "valid"].copy()


def test_coefficient_recovery_on_synthetic():
    train, _ = _synthetic_har(h=1)
    tr = train.loc[~train["embargo_drop_1"]].copy()
    res = fit_ols_nw(prepare_xy(tr, 1), 1)
    assert abs(res.params["har_d"] - 0.5) < 0.05
    assert abs(res.params["har_w"] - 0.3) < 0.05
    assert abs(res.params["har_m"] - 0.2) < 0.05
    assert abs(res.params["const"] - 0.1) < 0.05


def test_first_forecast_train_excludes_embargo():
    train, valid = _synthetic_har(h=1)
    history = pd.concat([train, valid], ignore_index=True)
    t0 = valid["date"].min()
    tr0 = expanding_train(history, t0, 1)
    assert tr0["embargo_drop_1"].sum() == 0
    assert (tr0["target_end_date_1"] <= t0).all()


def test_no_target_end_after_t_for_h7():
    train, valid = _synthetic_har(n_train=80, n_valid=30, h=7)
    history = prepare_xy(pd.concat([train, valid], ignore_index=True), 7)
    if "embargo_drop_7" not in history.columns:
        history["embargo_drop_7"] = False
    tend = target_end_col(7)
    for t in valid["date"]:
        tr = expanding_train(history, t, 7)
        assert (tr[tend] <= pd.Timestamp(t)).all()


def test_no_target_leak_h7_on_real_if_present():
    p = ROOT / "data" / "processed" / "model_table_B.parquet"
    if not p.exists():
        pytest.skip("chưa có model_table_B")
    train = load_model_table("B", "train", horizon=7, root=ROOT)
    valid = load_model_table("B", "valid", horizon=7, root=ROOT)
    asset = "BTC"
    history = prepare_xy(
        pd.concat([train[train.asset == asset], valid[valid.asset == asset]], ignore_index=True),
        7,
    )
    if "embargo_drop_7" not in history.columns:
        history["embargo_drop_7"] = False
    tend = target_end_col(7)
    dates = sorted(valid.loc[valid["asset"] == asset, "date"].unique())[::30]
    for t in dates:
        tr = expanding_train(history, t, 7)
        assert (tr[tend] <= pd.Timestamp(t).normalize()).all()


def test_har_dow_beats_plain_with_weekend_effect():
    """(a) hiệu ứng cuối tuần mạnh → har_dow thắng har_plain (R²_OOS>0, CW p nhỏ)."""
    train, valid = _synthetic_har(
        n_train=180, n_valid=60, seed=42, weekend_effect=0.8,
    )
    train = train.loc[~train["embargo_drop_1"]]
    _, sums = run_asset_horizon(train, valid, "BTC", 1)
    by = {s["baseline"]: s for s in sums}
    assert by["har_dow"]["r2_oos_vs_har_plain"] > 0.05
    assert by["har_dow"]["cw_pvalue_vs_har_plain"] < 0.05


def test_har_dow_not_significant_without_dow_effect():
    """(b) không hiệu ứng DOW → har_dow không thắng đáng kể."""
    train, valid = _synthetic_har(
        n_train=180, n_valid=60, seed=42, weekend_effect=0.0,
    )
    train = train.loc[~train["embargo_drop_1"]]
    _, sums = run_asset_horizon(train, valid, "BTC", 1)
    by = {s["baseline"]: s for s in sums}
    assert by["har_dow"]["r2_oos_vs_har_plain"] <= 0.02 or by["har_dow"]["cw_pvalue_vs_har_plain"] > 0.05


def test_dow_dummies_only_use_date_t():
    """(c) biến giả DOW chỉ phụ thuộc ngày t."""
    dates = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-06", "2021-03-07"])
    # Mon, Tue, Sat, Sun
    dum = dow_dummy_frame(dates)
    assert list(dum.columns) == list(DOW_DUMMY_COLS)
    assert dum.loc[0].sum() == 0  # Monday → all zero
    assert dum.loc[1, "dow_Tue"] == 1.0
    assert dum.loc[2, "dow_Sat"] == 1.0
    assert dum.loc[3, "dow_Sun"] == 1.0
    # đổi giá trị khác không ảnh hưởng
    df = pd.DataFrame({"date": dates, "noise": [9, 9, 9, 9]})
    out = add_dow_dummies(df)
    assert (out[list(DOW_DUMMY_COLS)].to_numpy() == dum.to_numpy()).all()


def test_b1_beats_b0_under_jensen_bias():
    rng = np.random.default_rng(42)
    n_train, n_valid = 150, 60
    n = n_train + n_valid
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    har_m = -6.0 + 0.01 * np.arange(n) + rng.normal(0, 0.05, n)
    har_d = har_m + rng.normal(0, 0.02, n)
    har_w = har_m + rng.normal(0, 0.02, n)
    y = har_m + 0.8 + rng.normal(0, 0.03, n)
    df = pd.DataFrame({
        "date": dates, "asset": "BTC",
        "har_d": har_d, "har_w": har_w, "har_m": har_m,
        "y_logrv_1": y, "y_rv_1": np.exp(y),
        "embargo_drop_1": False,
        "target_end_date_1": dates + pd.Timedelta(days=1),
        "split": ["train"] * n_train + ["valid"] * n_valid,
    })
    _, sums = run_asset_horizon(df[df.split == "train"], df[df.split == "valid"], "BTC", 1)
    plain = next(s for s in sums if s["baseline"] == "har_plain")
    assert plain["rmse_b1"] < plain["rmse_b0"]


def test_reproducible_twice_seed42(tmp_path):
    p = ROOT / "data" / "processed" / "model_table_B.parquet"
    if not p.exists():
        pytest.skip("chưa có model_table_B")
    s1 = run_all(horizons=[1], assets=("BTC",), root=ROOT, out_dir=tmp_path / "a", seed=42)
    s2 = run_all(horizons=[1], assets=("BTC",), root=ROOT, out_dir=tmp_path / "b", seed=42)
    cols = [
        "baseline", "coef_const", "coef_har_d", "coef_har_w", "coef_har_m",
        "r2_is", "r2_oos_vs_b0", "r2_oos_vs_b1", "r2_oos_vs_har_plain",
        "rmse", "n_oos", "cw_pvalue_vs_har_plain",
    ]
    pd.testing.assert_frame_equal(
        s1[cols].reset_index(drop=True),
        s2[cols].reset_index(drop=True),
        rtol=1e-10, atol=1e-10,
    )


def test_run_asset_horizon_synthetic_oos_smoke():
    train, valid = _synthetic_har(h=1)
    train2 = train.loc[~train["embargo_drop_1"]].copy()
    oos, sums = run_asset_horizon(train2, valid, "BTC", 1)
    assert len(sums) == 2
    assert {s["baseline"] for s in sums} == {"har_plain", "har_dow"}
    assert sums[0]["n_oos"] == len(valid)
    assert "yhat_har_dow" in oos.columns

"""Test HAR+text — mẫu chung, rò rỉ, tín hiệu/nhiễu, chốt test, tái lập."""
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
from models.har_baseline import FEATURES, FEATURES_DOW, expanding_train, fit_ols_nw, predict_row  # noqa: E402
from models.har_text import (  # noqa: E402
    ensure_registry_baseline_column,
    holm_adjust,
    prepare_xy_text,
    run_all,
    run_pair_oos,
)


def test_load_test_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("B", "test", horizon=1, root=ROOT)


def test_holm_monotonic():
    raw = [0.04, 0.01, 0.20, 0.03]
    adj = holm_adjust(raw)
    assert adj[1] == pytest.approx(min(1.0, 0.01 * 4))
    assert max(adj) <= 1.0


def _toy_panel(n_train=120, n_valid=40, seed=42, text_mode="signal"):
    rng = np.random.default_rng(seed)
    n = n_train + n_valid
    dates = pd.date_range("2021-06-01", periods=n, freq="D")
    har_d = rng.normal(-5.0, 0.4, n)
    har_w = har_d + rng.normal(0, 0.05, n)
    har_m = har_d + rng.normal(0, 0.08, n)
    if text_mode == "signal":
        text = rng.normal(0, 1.0, n)
        y = -0.5 + 0.4 * har_d + 0.3 * har_w + 0.2 * har_m + 0.6 * text + rng.normal(0, 0.05, n)
    else:
        text = rng.normal(0, 1.0, n)
        y = -0.5 + 0.4 * har_d + 0.3 * har_w + 0.2 * har_m + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "har_d": har_d,
        "har_w": har_w,
        "har_m": har_m,
        "all_finbert_Sent_t": text,
        "y_logrv_1": y,
        "y_rv_1": np.exp(y),
        "embargo_drop_1": False,
        "target_end_date_1": dates + pd.Timedelta(days=1),
        "split": ["train"] * n_train + ["valid"] * n_valid,
    })
    return df[df.split == "train"].copy(), df[df.split == "valid"].copy()


def test_common_sample_same_train_rows():
    train, valid = _toy_panel(text_mode="signal")
    text_col = "all_finbert_Sent_t"
    tr = prepare_xy_text(train, 1, text_col)
    va = prepare_xy_text(valid, 1, text_col)
    history = pd.concat([tr, va], ignore_index=True).sort_values("date").reset_index(drop=True)
    oos, _ = run_pair_oos(history, va.head(8), 1, text_col)
    assert "train_idx" in oos.columns
    t = va.iloc[3]["date"]
    train_t = expanding_train(history, t, 1).dropna(subset=["y_logrv_1", *FEATURES_DOW, text_col])
    row = va.iloc[3]
    res_h = fit_ols_nw(train_t, 1, FEATURES_DOW)
    res_x = fit_ols_nw(train_t, 1, list(FEATURES_DOW) + [text_col])
    assert abs(predict_row(res_h, row, FEATURES_DOW) - oos.loc[oos["date"] == t, "yhat_har_dow"].iloc[0]) < 1e-9
    assert abs(predict_row(res_x, row, list(FEATURES_DOW) + [text_col]) - oos.loc[oos["date"] == t, "yhat_text_dow"].iloc[0]) < 1e-9


def test_no_leak_future_text_or_y():
    train, valid = _toy_panel(text_mode="signal")
    text_col = "all_finbert_Sent_t"
    tr = prepare_xy_text(train, 1, text_col)
    va = prepare_xy_text(valid, 1, text_col)
    history = pd.concat([tr, va], ignore_index=True).sort_values("date").reset_index(drop=True)
    t = va.iloc[5]["date"]
    va_one = va.loc[va["date"] == t]
    oos0, _ = run_pair_oos(history, va_one, 1, text_col)
    y0 = float(oos0["yhat_text_dow"].iloc[0])
    hist2 = history.copy()
    hist2.loc[hist2["date"] > t, text_col] = 99.0
    hist2.loc[hist2["date"] > t, "y_logrv_1"] = 99.0
    oos1, _ = run_pair_oos(hist2, va_one, 1, text_col)
    assert abs(float(oos1["yhat_text_dow"].iloc[0]) - y0) < 1e-12


def test_signal_positive_r2_and_small_cw_p():
    train, valid = _toy_panel(n_train=150, n_valid=50, text_mode="signal", seed=42)
    text_col = "all_finbert_Sent_t"
    tr = prepare_xy_text(train, 1, text_col)
    va = prepare_xy_text(valid, 1, text_col)
    history = pd.concat([tr, va], ignore_index=True).sort_values("date").reset_index(drop=True)
    _, meta = run_pair_oos(history, va, 1, text_col)
    assert meta["r2_oos_vs_har_dow"] > 0
    assert meta["cw_pvalue_raw"] < 0.05


def test_noise_weak_or_insignificant():
    train, valid = _toy_panel(n_train=150, n_valid=50, text_mode="noise", seed=42)
    text_col = "all_finbert_Sent_t"
    tr = prepare_xy_text(train, 1, text_col)
    va = prepare_xy_text(valid, 1, text_col)
    history = pd.concat([tr, va], ignore_index=True).sort_values("date").reset_index(drop=True)
    _, meta = run_pair_oos(history, va, 1, text_col)
    assert meta["r2_oos_vs_har_dow"] <= 0.01 or meta["cw_pvalue_raw"] > 0.05


def test_registry_backfill_baseline(tmp_path):
    p = tmp_path / "test_registry.csv"
    pd.DataFrame({
        "timestamp": ["t0"], "script": ["x"], "model": ["M_fin"],
        "asset": ["BTC"], "h": [1], "split": ["valid"], "n_oos": [1], "p_raw": [0.5],
    }).to_csv(p, index=False)
    ensure_registry_baseline_column(p)
    r = pd.read_csv(p)
    assert "baseline" in r.columns
    assert r.loc[0, "baseline"] == "har_plain"


def test_reproducible_seed42(tmp_path):
    p = ROOT / "data" / "processed" / "model_table_B.parquet"
    if not p.exists():
        pytest.skip("chưa có model_table_B")
    models = {"M_fin": "all_finbert_Sent_t"}
    s1 = run_all(
        horizons=[1], assets=("BTC",), root=ROOT, out_dir=tmp_path / "a",
        seed=42, models=models, include_deseason_robust=False,
    )
    s2 = run_all(
        horizons=[1], assets=("BTC",), root=ROOT, out_dir=tmp_path / "b",
        seed=42, models=models, include_deseason_robust=False,
    )
    cols = [
        "r2_oos_vs_har_dow", "r2_oos_vs_har_plain", "rmse_text_dow", "rmse_har_dow",
        "cw_pvalue_raw", "coef_text", "se_text_nw", "n_oos",
    ]
    pd.testing.assert_frame_equal(
        s1[cols].reset_index(drop=True),
        s2[cols].reset_index(drop=True),
        rtol=1e-10, atol=1e-10,
    )

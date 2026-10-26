"""Test RQ1/RQ2 — bootstrap ΔR², Wald bất đối xứng, mẫu chung, Holm, tái lập."""
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
from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies  # noqa: E402
from models.rq1_rq2_sentiment import (  # noqa: E402
    CONTROL_A,
    CONTROL_B,
    COL_FIN_SENT,
    COL_NEG,
    COL_POS,
    COL_VAD_SENT,
    bootstrap_delta_r2_adj,
    common_sample,
    load_train_valid,
    prepare_asset_frame,
    run_all,
    run_rq1_for_asset,
    run_rq2_for_asset,
    wald_pos_neg,
)
from models.rq3_narratives import CONTROL_FRED, holm_adjust  # noqa: E402


def _ref_holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = np.argsort(np.asarray(pvalues, dtype=float))
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        raw = float(pvalues[int(idx)]) * (m - rank)
        val = min(1.0, max(running, raw))
        adj[int(idx)] = val
        running = val
    return adj.tolist()


def _base_controls(n: int, rng: np.random.Generator, dates: pd.DatetimeIndex) -> pd.DataFrame:
    har_d = rng.normal(-5.0, 0.3, n)
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "split": ["train"] * (n // 2) + ["valid"] * (n - n // 2),
        "ret_lag1": rng.normal(0, 0.02, n),
        "ret_lag2": rng.normal(0, 0.02, n),
        "har_d": har_d,
        "har_w": har_d + rng.normal(0, 0.05, n),
        "har_m": har_d + rng.normal(0, 0.08, n),
        "logvol_lag1": rng.normal(10, 0.2, n),
        "ret_t": rng.normal(0, 0.02, n),
        "logvol_t": rng.normal(10, 0.2, n),
        **{c: rng.normal(0, 1, n) for c in CONTROL_FRED},
    })
    return add_dow_dummies(df)


def test_load_test_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("A", "test", root=ROOT)


def test_no_test_rows_in_estimation():
    if not (ROOT / "data" / "processed" / "model_table_A.parquet").exists():
        pytest.skip("no model_table_A")
    df = load_train_valid("A", root=ROOT)
    assert "test" not in set(df["split"].astype(str))


def test_signal_finbert_beats_vader_bootstrap():
    """(a) Sent_F tín hiệu, Sent_V nhiễu → ΔR²_adj>0, CI loại 0."""
    rng = np.random.default_rng(42)
    n = 400
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    df = _base_controls(n, rng, dates)
    sent_f = rng.normal(0, 1, n)
    sent_v = rng.normal(0, 1, n)
    y = 0.5 * sent_f + 0.1 * df["har_d"].to_numpy() + rng.normal(0, 0.2, n)
    df["y_ret"] = y
    df["y_log_rv"] = y
    df[COL_FIN_SENT] = sent_f
    df[COL_VAD_SENT] = sent_v
    df = prepare_asset_frame(df, "BTC")
    out = run_rq1_for_asset(df, "BTC", n_boot=800, seed=42)
    row = out.loc[out["hypothesis"] == "H1a"].iloc[0]
    assert row["delta_r2_adj"] > 0
    assert row["ci_lo"] > 0 or row["p_boot"] < 0.05
    assert not (row["ci_lo"] <= 0 <= row["ci_hi"] and row["p_boot"] > 0.1)


def test_null_ci_covers_zero_most_sims():
    """(b) cả hai nhiễu: CI chứa 0 trong ≥90% trên 200 mô phỏng seed 42."""
    rng = np.random.default_rng(42)
    n_sims = 200
    cover = 0
    for _ in range(n_sims):
        seed_i = int(rng.integers(0, 1_000_000))
        rng_i = np.random.default_rng(seed_i)
        n = 250
        dates = pd.date_range("2021-03-01", periods=n, freq="D")
        df = _base_controls(n, rng_i, dates)
        sf = rng_i.normal(0, 1, n)
        sv = rng_i.normal(0, 1, n)
        y = 0.1 * df["har_d"].to_numpy() + rng_i.normal(0, 0.25, n)
        boot = bootstrap_delta_r2_adj(
            y,
            df[list(CONTROL_A)].to_numpy(dtype=float),
            sf, sv,
            n_boot=300, seed=seed_i,
        )
        if boot["ci_lo"] <= 0 <= boot["ci_hi"]:
            cover += 1
    assert cover / n_sims >= 0.90


def test_asymmetric_wald_rejects():
    """(c) β_neg = −2 β_pos (thang raw) → Wald bác bỏ; tổng < 0."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    df = _base_controls(n, rng, dates)
    pos = rng.uniform(0.1, 0.5, n)
    neg = rng.uniform(0.1, 0.5, n)
    y = 0.10 * pos - 0.20 * neg + rng.normal(0, 0.02, n)
    df["y_ret"] = y
    df[COL_POS] = pos
    df[COL_NEG] = neg
    work = common_sample(df, "y_ret", [COL_POS, COL_NEG], CONTROL_A)
    st = wald_pos_neg(work, "y_ret", CONTROL_A, bandwidth=5)
    assert st["p_wald"] < 0.05
    assert st["coef_sum"] < 0


def test_symmetric_wald_p_approx_uniform():
    """(d) đối xứng trên tỷ lệ gốc: tỷ lệ p_wald < 0.05 ≤ 10% trên 200 sim."""
    rng = np.random.default_rng(42)
    hits = 0
    n_sims = 200
    for _ in range(n_sims):
        seed_i = int(rng.integers(0, 1_000_000))
        rng_i = np.random.default_rng(seed_i)
        n = 300
        dates = pd.date_range("2021-03-01", periods=n, freq="D")
        df = _base_controls(n, rng_i, dates)
        pos = rng_i.uniform(0.1, 0.5, n)
        neg = rng_i.uniform(0.1, 0.5, n)
        # H0 đúng trên thang raw: β_pos = −β_neg
        y = 0.08 * pos - 0.08 * neg + rng_i.normal(0, 0.05, n)
        df["y_ret"] = y
        df[COL_POS] = pos
        df[COL_NEG] = neg
        work = common_sample(df, "y_ret", [COL_POS, COL_NEG], CONTROL_A)
        st = wald_pos_neg(work, "y_ret", CONTROL_A, bandwidth=5)
        if st["p_wald"] < 0.05:
            hits += 1
    assert hits / n_sims <= 0.10


def test_common_sample_drops_any_nan():
    """(e) mẫu chung: NaN ở F hoặc V đều bị bỏ."""
    rng = np.random.default_rng(0)
    n = 50
    dates = pd.date_range("2021-06-01", periods=n, freq="D")
    df = _base_controls(n, rng, dates)
    df["y_ret"] = rng.normal(0, 1, n)
    df[COL_FIN_SENT] = rng.normal(0, 1, n)
    df[COL_VAD_SENT] = rng.normal(0, 1, n)
    df.loc[3, COL_FIN_SENT] = np.nan
    df.loc[7, COL_VAD_SENT] = np.nan
    df.loc[10, "y_ret"] = np.nan
    work = common_sample(df, "y_ret", [COL_FIN_SENT, COL_VAD_SENT], CONTROL_A)
    assert len(work) == n - 3


def test_controls_spec_a_lagged_or_calendar():
    assert set(DOW_DUMMY_COLS).issubset(CONTROL_A)
    assert "y_ret" not in CONTROL_A
    assert "ret_t" not in CONTROL_A
    assert "ret_lag1" in CONTROL_A
    assert set(DOW_DUMMY_COLS).issubset(CONTROL_B)
    assert "ret_t" in CONTROL_B


def test_holm_matches_reference():
    """(h) Holm đúng."""
    raw = [0.04, 0.01, 0.20, 0.03]
    assert holm_adjust(raw) == pytest.approx(_ref_holm(raw))


def test_reproducible_rq1_subset(tmp_path):
    """(g) tái lập seed 42 trên RQ1 BTC (n_boot nhỏ)."""
    if not (ROOT / "data" / "processed" / "model_table_A.parquet").exists():
        pytest.skip("no tables")
    raw = load_train_valid("A", root=ROOT)
    df = prepare_asset_frame(raw, "BTC")
    a = run_rq1_for_asset(df, "BTC", n_boot=200, seed=42)
    b = run_rq1_for_asset(df, "BTC", n_boot=200, seed=42)
    cols = ["delta_r2_adj", "ci_lo", "ci_hi", "p_boot", "coef_F_1sd", "n"]
    pd.testing.assert_frame_equal(
        a[cols].reset_index(drop=True),
        b[cols].reset_index(drop=True),
        rtol=1e-12, atol=1e-12,
    )

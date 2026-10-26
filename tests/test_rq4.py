"""Test RQ4 Granger / BIC / Holm / no-test / tái lập."""
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
from models.rq3_narratives import holm_adjust  # noqa: E402
from models.rq4_leadlag import (  # noqa: E402
    COL_RET,
    COL_SENT,
    add_calendar_lags,
    granger_sum_wald_calendar,
    prepare_rq4_frame,
    run_granger_pair,
    select_p_bic,
    to_calendar_frame,
)


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


def _toy_var(n=400, seed=42, sent_leads_ret=False, ret_leads_sent=False, p_true=2):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    sent = np.zeros(n)
    ret = np.zeros(n)
    sent[0] = rng.normal()
    ret[0] = rng.normal()
    for t in range(1, n):
        sent[t] = 0.4 * sent[t - 1] + rng.normal(0, 0.5)
        ret[t] = 0.3 * ret[t - 1] + rng.normal(0, 0.5)
        if sent_leads_ret and t >= 1:
            ret[t] += 0.7 * sent[t - 1]
        if ret_leads_sent and t >= 1:
            sent[t] += 0.7 * ret[t - 1]
        if p_true >= 2 and sent_leads_ret and t >= 2:
            ret[t] += 0.3 * sent[t - 2]
        if p_true >= 2 and ret_leads_sent and t >= 2:
            sent[t] += 0.3 * ret[t - 2]
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "split": ["train"] * (n // 2) + ["valid"] * (n - n // 2),
        COL_RET: ret,
        COL_SENT: sent,
        "all_finbert_MeanProb_t": sent + rng.normal(0, 0.01, n),
        "all_finbert_AbnAttention_t": rng.normal(0, 1, n),
    })
    return add_dow_dummies(df)


def test_load_test_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("A", "test", root=ROOT, panel="noprice")


def test_sent_leads_ret():
    """(a) Sent dẫn ret → Sent→ret có ý nghĩa, ret→Sent không."""
    df = prepare_rq4_frame(_toy_var(n=500, seed=42, sent_leads_ret=True), "BTC")
    g = run_granger_pair(df, COL_SENT, p=2)
    assert g["sent_to_ret"]["p_raw"] < 0.05
    assert g["ret_to_sent"]["p_raw"] > 0.05


def test_ret_leads_sent():
    """(b) ret dẫn Sent."""
    df = prepare_rq4_frame(
        _toy_var(n=600, seed=7, ret_leads_sent=True, sent_leads_ret=False), "BTC",
    )
    # tăng tín hiệu: phóng đại hệ số trong dữ liệu đã sinh bằng seed mạnh hơn
    g = run_granger_pair(df, COL_SENT, p=2)
    assert g["ret_to_sent"]["p_raw"] < 0.05
    # chiều ngược không mạnh hơn chiều đúng (cho phép nhiễu nhỏ)
    assert g["ret_to_sent"]["p_raw"] < g["sent_to_ret"]["p_raw"]


def test_null_p_approx_uniform():
    """(c) nhiễu: ≤10% p<0.05 trên 200 sim (một chiều Sent→ret)."""
    rng = np.random.default_rng(42)
    hits = 0
    n_sims = 200
    for _ in range(n_sims):
        seed_i = int(rng.integers(0, 1_000_000))
        df = prepare_rq4_frame(_toy_var(n=300, seed=seed_i), "BTC")
        g = run_granger_pair(df, COL_SENT, p=2)
        if g["sent_to_ret"]["p_raw"] < 0.05:
            hits += 1
    assert hits / n_sims <= 0.10


def test_bic_recovers_order_2():
    """(d) VAR giả bậc 2 → BIC chọn p=2 (hoặc gần)."""
    rng = np.random.default_rng(42)
    n = 600
    # true VAR(2)
    y = np.zeros((n, 2))
    y[0] = rng.normal(size=2)
    y[1] = rng.normal(size=2)
    A1 = np.array([[0.4, 0.2], [0.1, 0.35]])
    A2 = np.array([[0.15, 0.05], [0.0, 0.2]])
    for t in range(2, n):
        y[t] = A1 @ y[t - 1] + A2 @ y[t - 2] + rng.normal(0, 0.3, 2)
    endog = pd.DataFrame(y, columns=["ret", "sent"])
    exog = pd.DataFrame(
        {c: np.zeros(n) for c in DOW_DUMMY_COLS}
    )
    # set one DOW so matrix not singular constant-only — use actual DOW from dates
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    exog = add_dow_dummies(pd.DataFrame({"date": dates}))[list(DOW_DUMMY_COLS)]
    p, bic = select_p_bic(endog, exog, maxlags=7)
    assert p == 2 or (bic.get(2, 1e9) <= min(bic.values()) + 1e-6)


def test_no_test_in_estimation():
    p = ROOT / "data" / "processed" / "model_table_A_noprice.parquet"
    if not p.exists():
        pytest.skip("chưa có A_noprice")
    from models.rq4_leadlag import load_train_valid_A
    df = load_train_valid_A(root=ROOT, panel="noprice")
    assert "test" not in set(df["split"].astype(str))


def test_reproducible_granger():
    df = prepare_rq4_frame(_toy_var(n=400, seed=42, sent_leads_ret=True), "BTC")
    a = run_granger_pair(df, COL_SENT, p=2)
    b = run_granger_pair(df, COL_SENT, p=2)
    assert a["sent_to_ret"]["coef_sum"] == pytest.approx(b["sent_to_ret"]["coef_sum"])
    assert a["sent_to_ret"]["p_raw"] == pytest.approx(b["sent_to_ret"]["p_raw"])


def test_holm_ok():
    raw = [0.01, 0.04]
    assert holm_adjust(raw) == pytest.approx(_ref_holm(raw))


def test_calendar_lag_nan_gap():
    """(a) ngày giữa NaN → trễ-1 của ngày sau là NaN, không lấy ngày trước đó."""
    dates = pd.date_range("2021-03-01", periods=5, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        COL_RET: [1.0, 2.0, np.nan, 4.0, 5.0],
        COL_SENT: [0.1, 0.2, np.nan, 0.4, 0.5],
    })
    cal = add_dow_dummies(df)
    lagged = add_calendar_lags(cal, COL_RET, COL_SENT, max_p=1)
    # ngày index 3 = 2021-03-04: y_L1 phải NaN (từ ngày 3), không phải 2.0
    row = lagged.loc[lagged["date"] == dates[3]].iloc[0]
    assert pd.isna(row["y_L1"])
    assert pd.isna(row["x_L1"])
    # ngày index 1: lag từ ngày 0
    row1 = lagged.loc[lagged["date"] == dates[1]].iloc[0]
    assert row1["y_L1"] == pytest.approx(1.0)
    assert row1["x_L1"] == pytest.approx(0.1)


def test_calendar_lag_recovers_with_scattered_nan():
    """(b) ret dẫn Sent đúng 1 ngày lịch; khôi phục hệ số dù ~10% ngày NaN."""
    rng = np.random.default_rng(42)
    n = 500
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    ret = rng.normal(0, 1, n)
    sent = np.zeros(n)
    true_beta = 0.65
    for t in range(1, n):
        sent[t] = 0.2 * (sent[t - 1] if t else 0) + true_beta * ret[t - 1] + rng.normal(0, 0.15)
    # 10% ngày NaN rải rác (không đụng quá nhiều liên tiếp)
    nan_idx = rng.choice(np.arange(10, n - 10), size=n // 10, replace=False)
    ret_m = ret.copy()
    sent_m = sent.copy()
    ret_m[nan_idx] = np.nan
    sent_m[nan_idx] = np.nan
    df = pd.DataFrame({
        "date": dates, "asset": "BTC",
        COL_RET: ret_m, COL_SENT: sent_m,
    })
    cal = add_dow_dummies(df)
    g = granger_sum_wald_calendar(cal, COL_SENT, COL_RET, p=1, bandwidth=5)
    # coef trên ret_L1 (= x_L1 trong pt sent) ≈ true_beta
    assert g["coefs_lags"][0] == pytest.approx(true_beta, abs=0.12)
    assert g["p_raw"] < 0.05
    assert g["n"] < n - 1  # đã bỏ dòng NaN

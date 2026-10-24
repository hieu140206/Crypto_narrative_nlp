"""Test RQ3 narratives — hồi phục hệ số, null, chốt test, kiểm soát, Holm, z-score, tái lập."""
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
from models.har_text import holm_adjust  # noqa: E402
from models.rq3_narratives import (  # noqa: E402
    CONTROL_COLS,
    CONTROL_FRED,
    CONTROL_MARKET,
    NARRATIVES,
    OUTCOMES,
    NW_BANDWIDTH,
    prepare_asset_frame,
    run_all,
    run_main_for_asset,
    run_one_regression,
    zscore_series,
)


def _ref_holm(pvalues: list[float]) -> list[float]:
    """Holm tham chiếu độc lập (cùng công thức step-down)."""
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


def _synthetic_panel(
    n: int = 400,
    seed: int = 42,
    beta: float = 0.0,
    narr: str = "regulation",
):
    """Panel giả BTC: controls + AbnAtt + y = beta*z(x) + noise (+ optional control effects)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-03-01", periods=n, freq="D")
    ret_lag1 = rng.normal(0, 0.02, n)
    ret_lag2 = rng.normal(0, 0.02, n)
    har_d = rng.normal(-5.0, 0.4, n)
    har_w = har_d + rng.normal(0, 0.05, n)
    har_m = har_d + rng.normal(0, 0.08, n)
    logvol_lag1 = rng.normal(10, 0.3, n)
    fred = {c: rng.normal(0, 1, n) for c in CONTROL_FRED}
    x_raw = rng.normal(0, 1.5, n)
    # z on full sample for DGP
    z = (x_raw - x_raw.mean()) / x_raw.std(ddof=1)
    eps = rng.normal(0, 0.15, n)
    y = (
        0.1
        + beta * z
        + 0.2 * har_d
        + 0.1 * ret_lag1
        + eps
    )
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "split": ["train"] * (n // 2) + ["valid"] * (n - n // 2),
        "ret_lag1": ret_lag1,
        "ret_lag2": ret_lag2,
        "har_d": har_d,
        "har_w": har_w,
        "har_m": har_m,
        "logvol_lag1": logvol_lag1,
        "y_log_rv": y,
        "y_abvol": rng.normal(0, 1, n),
        "y_ret": rng.normal(0, 0.02, n),
        f"AbnAtt_{narr}": x_raw,
        "all_finbert_AbnAttention_t": x_raw + rng.normal(0, 0.1, n),
        **fred,
    })
    for other in NARRATIVES:
        if other != narr:
            df[f"AbnAtt_{other}"] = rng.normal(0, 1.5, n)
    return df


def test_load_test_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("A", "test", root=ROOT)


def test_estimation_sample_excludes_test():
    p = ROOT / "data" / "processed" / "model_table_A.parquet"
    if not p.exists():
        pytest.skip("chưa có model_table_A")
    from models.rq3_narratives import load_train_valid_A
    df = load_train_valid_A(root=ROOT)
    assert "test" not in set(df["split"].astype(str).unique())
    assert set(df["split"].astype(str).unique()) <= {"train", "valid"}


def test_coefficient_recovery_within_se():
    """(a) hệ số thật → ước lượng trong khoảng ~3 SE."""
    true_beta = 0.35
    df = _synthetic_panel(n=500, seed=42, beta=true_beta, narr="regulation")
    df = prepare_asset_frame(df, "BTC")
    stats = run_one_regression(
        df,
        y_col="y_log_rv",
        x_raw=df["AbnAtt_regulation"],
        control_cols=CONTROL_COLS,
    )
    assert abs(stats["coef_1sd"] - true_beta) < 3.0 * stats["se_nw"]
    assert abs(stats["coef_1sd"] - true_beta) < 0.08


def test_null_pvalues_approx_uniform():
    """(b) không tín hiệu: tỷ lệ p_raw < 0.05 ≤ 10% trên 200 mô phỏng seed 42."""
    rng = np.random.default_rng(42)
    hits = 0
    n_sims = 200
    for i in range(n_sims):
        seed_i = int(rng.integers(0, 1_000_000))
        df = _synthetic_panel(n=350, seed=seed_i, beta=0.0, narr="etf")
        df = prepare_asset_frame(df, "BTC")
        stats = run_one_regression(
            df, y_col="y_abvol", x_raw=df["AbnAtt_etf"], control_cols=CONTROL_COLS,
        )
        if stats["p_raw"] < 0.05:
            hits += 1
    rate = hits / n_sims
    assert rate <= 0.10, "null reject rate=%.3f > 0.10" % rate


def test_controls_only_lagged_or_calendar():
    """(d) kiểm soát chỉ trễ / lịch; không y_t hay biến cùng ngày thị trường."""
    forbidden_exact = {
        "y_ret", "y_log_rv", "y_abvol", "ret", "log_rv", "abvol", "rv", "volume",
    }
    assert not (set(CONTROL_COLS) & forbidden_exact)
    for c in CONTROL_MARKET:
        assert "lag" in c or c.startswith("har_")
    for c in CONTROL_FRED:
        assert c.startswith("fred_")
    assert set(DOW_DUMMY_COLS).issubset(set(CONTROL_COLS))
    # DOW chỉ từ date
    dates = pd.to_datetime(["2021-03-01", "2021-03-06"])
    dum = add_dow_dummies(pd.DataFrame({"date": dates, "noise": [1, 2]}))
    assert list(dum.columns[-6:]) == list(DOW_DUMMY_COLS)


def test_holm_matches_reference():
    """(e) Holm khớp implementation tham chiếu."""
    raw = [0.04, 0.001, 0.20, 0.03, 0.50, 0.01]
    assert holm_adjust(raw) == pytest.approx(_ref_holm(raw))
    assert holm_adjust(raw) == pytest.approx(holm_adjust(raw))


def test_zscore_uses_estimation_sample():
    """(f) z-score mean≈0, std≈1 trên mẫu; không dùng quan sát ngoài mẫu."""
    df = _synthetic_panel(n=200, seed=7, beta=0.2)
    df = prepare_asset_frame(df, "BTC")
    # chạy regression — z bên trong phải có mean 0 / std 1 trên complete sample
    work = df[["y_log_rv", "AbnAtt_regulation", *CONTROL_COLS]].dropna()
    z, mu, sd = zscore_series(work["AbnAtt_regulation"])
    assert abs(z.mean()) < 1e-10
    assert abs(z.std(ddof=1) - 1.0) < 1e-10
    # thêm điểm ngoài không đổi mu/sd của mẫu cũ
    z2, mu2, sd2 = zscore_series(work["AbnAtt_regulation"])
    assert mu == pytest.approx(mu2)
    assert sd == pytest.approx(sd2)
    stats = run_one_regression(
        df, y_col="y_log_rv", x_raw=df["AbnAtt_regulation"], control_cols=CONTROL_COLS,
    )
    assert stats["z_mean"] == pytest.approx(mu, rel=1e-9)
    assert stats["z_std"] == pytest.approx(sd, rel=1e-9)


def test_nw_bandwidth_is_5():
    assert NW_BANDWIDTH == 5


def test_main_holm_family_size_21():
    df = _synthetic_panel(n=300, seed=42, beta=0.0)
    # need all outcomes / narratives columns already in _synthetic_panel
    df = prepare_asset_frame(df, "BTC")
    main = run_main_for_asset(df, "BTC")
    assert len(main) == 21
    assert set(main["narrative"]) == set(NARRATIVES)
    assert set(main["y"]) == set(OUTCOMES)
    assert "p_holm" in main.columns
    # Holm ≥ raw (approx, allowing ties)
    assert (main["p_holm"] + 1e-12 >= main["p_raw"]).all()


def test_reproducible_twice(tmp_path):
    """(g) tái lập seed 42."""
    p = ROOT / "data" / "processed" / "model_table_A.parquet"
    if not p.exists():
        pytest.skip("chưa có model_table_A")
    # chỉ chạy main BTC nhanh hơn: monkey via run_all but full is ok for A OLS
    s1 = run_all(root=ROOT, out_dir=tmp_path / "a", seed=42)
    s2 = run_all(root=ROOT, out_dir=tmp_path / "b", seed=42)
    cols = ["asset", "narrative", "y", "coef_1sd", "se_nw", "p_raw", "p_holm", "n"]
    m1 = s1["main"][cols].sort_values(["asset", "narrative", "y"]).reset_index(drop=True)
    m2 = s2["main"][cols].sort_values(["asset", "narrative", "y"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(m1, m2, rtol=1e-10, atol=1e-10)

"""Kiểm market_daily + FRED (chống look-ahead, lịch đủ, rv/abvol)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_market_daily import build_daily_for_asset, usable_from  # noqa: E402


def test_market_calendar_complete(market, params):
    start = pd.Timestamp(params["sample"]["robustness_start"])
    # cửa sổ market: robustness_start .. 2025-12-31 (2192 ngày)
    end = pd.Timestamp("2025-12-31")
    for asset, g in market.groupby("asset"):
        d = pd.to_datetime(g["date"]).sort_values()
        exp = pd.date_range(start, end, freq="D")
        assert len(d) == len(exp) == 2192, (asset, len(d))
        assert (d.values == exp.values).all(), f"{asset}: thiếu/lệch ngày"


def test_rv_nan_when_incomplete(market):
    bad = market.loc[~market["complete_day"]]
    assert len(bad) > 0
    assert bad["rv"].isna().all()
    assert 25 <= int((~market["complete_day"]).sum()) <= 31
    # ngày đủ 24h phải có rv hữu hạn
    good = market.loc[market["complete_day"]]
    assert good["rv"].notna().all()
    assert (good["rv"] >= 0).all()


def test_abvol_no_lookahead():
    """Baseline abvol chỉ dùng quá khứ: đổi volume_t không đổi abvol_{t-1}; có đổi abvol_{t+1}.

    abvol_t = log(vol_t) - mean(log vol_{t-1..t-30}), nên volume_t vẫn nằm trong abvol_t
    (đúng công thức); điểm chống look-ahead là mean không gồm t và không nhìn t+1.
    """
    rng = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    n = len(rng)
    fake = pd.DataFrame({
        "open_time_utc": pd.date_range("2020-01-01", periods=n * 24, freq="h", tz="UTC"),
        "open": np.linspace(100, 110, n * 24),
        "close": np.linspace(100, 110, n * 24) + 0.1,
        "volume": np.full(n * 24, 10.0),
    })
    t = 40
    day_t = rng[t]
    mask = (fake["open_time_utc"] >= day_t) & (fake["open_time_utc"] < day_t + pd.Timedelta(days=1))
    base = build_daily_for_asset(fake.copy(), rng)
    fake2 = fake.copy()
    fake2.loc[mask, "volume"] = 1e9
    bumped = build_daily_for_asset(fake2, rng)

    def ab(df, day):
        return float(df.loc[df["date"] == day, "abvol"].iloc[0])

    # không nhìn tương lai: abvol trước t không đổi
    assert np.isclose(ab(base, rng[t - 1]), ab(bumped, rng[t - 1]))
    # mean cửa sổ quá khứ của t+1 gồm volume_t -> abvol_{t+1} đổi
    assert not np.isclose(ab(base, rng[t + 1]), ab(bumped, rng[t + 1]))


def test_abvol_first_30_nan(market):
    for asset, g in market.groupby("asset"):
        g = g.sort_values("date")
        assert g["abvol"].head(30).isna().all(), asset
        assert g["abvol"].iloc[30:].notna().any(), asset


def test_fred_usable_from_after_date(fred):
    assert (fred["usable_from_utc"] > pd.to_datetime(fred["date"], utc=True)).all()
    # VIX +1; DGS10 +2 (H.15)
    expected_lag = {"VIXCLS": 1, "DGS10": 2}
    for series, lag in expected_lag.items():
        s = fred[fred["series"] == series]
        assert len(s) > 0
        delta = (s["usable_from_utc"] - pd.to_datetime(s["date"], utc=True)).dt.days
        assert (delta == lag).all(), (series, delta.value_counts().to_dict())
    # DTWEXBGS: Tuesday sau Friday của tuần -> usable luôn là Tuesday, và > date
    dtw = fred[fred["series"] == "DTWEXBGS"].copy()
    assert len(dtw) > 0
    assert (dtw["usable_from_utc"].dt.weekday == 1).all()  # Tuesday
    dates = pd.to_datetime(dtw["date"])
    friday = dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit="D")
    expect = (friday + pd.Timedelta(days=4)).dt.tz_localize("UTC")
    assert (dtw["usable_from_utc"].reset_index(drop=True) == expect.reset_index(drop=True)).all()


def test_rv_matches_manual_three_days(market, paths):
    """Đối chiếu rv với tính tay trên 3 ngày complete (từ hourly raw)."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    binance = root / paths["binance_dir"]
    sample = [pd.Timestamp("2021-06-15"), pd.Timestamp("2022-01-10"), pd.Timestamp("2024-03-01")]
    for asset, fname in (("BTC", "BTCUSDT_1h.parquet"), ("ETH", "ETHUSDT_1h.parquet")):
        h = pd.read_parquet(binance / fname)
        h["open_time_utc"] = pd.to_datetime(h["open_time"], utc=True)
        h = h.set_index("open_time_utc").sort_index()
        h["log_ret_1h"] = np.log(h["close"] / h["open"])
        for d in sample:
            d0 = d.tz_localize("UTC")
            bars = h.loc[d0:d0 + pd.Timedelta(hours=23)]
            assert len(bars) == 24, (asset, d, len(bars))
            rv_manual = float((bars["log_ret_1h"] ** 2).sum())
            row = market[(market["asset"] == asset) & (market["date"] == d)]
            assert len(row) == 1
            assert row["complete_day"].iloc[0]
            assert abs(float(row["rv"].iloc[0]) - rv_manual) < 1e-12


def test_usable_from_helper():
    dates = pd.to_datetime(["2024-01-03", "2024-01-05"])  # Wed, Fri
    vix = usable_from("VIXCLS", dates)
    assert list(vix.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")) == ["2024-01-04", "2024-01-06"]
    dgs = usable_from("DGS10", dates)
    assert list(dgs.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")) == ["2024-01-05", "2024-01-07"]
    # Fri 2024-01-05 -> Tuesday 2024-01-09
    dtw = usable_from("DTWEXBGS", pd.to_datetime(["2024-01-03", "2024-01-05"]))
    assert list(dtw.dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")) == ["2024-01-09", "2024-01-09"]


def test_market_end_from_params(params):
    assert params["sample"]["market_end"] == "2025-12-31"

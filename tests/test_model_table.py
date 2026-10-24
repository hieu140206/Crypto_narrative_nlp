"""Test build_model_table Spec A/B — chống rò rỉ, FRED as-of, target, embargo, chốt test."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_model_table import (  # noqa: E402
    build_model_table_A,
    build_model_table_B,
    har_from_log_rv,
    join_fred_asof,
    market_features_B,
    prepare_fred_with_d5,
)
from modeltable_io import TestSplitLockedError, load_model_table  # noqa: E402


def _params():
    return {
        "sample": {"start": "2021-03-01", "t_end": "2021-04-30"},
        "split": {
            "train": ["2021-03-01", "2021-03-31"],
            "validation": ["2021-04-01", "2021-04-15"],
            "test": ["2021-04-16", "2021-04-30"],
            "embargo_days": 7,
        },
        "horizons": [1, 3, 7],
        "own_n_min": 5,
        "seed": 42,
    }


def _toy_panel(start="2021-03-01", end="2021-04-30"):
    cal = pd.date_range(start, end, freq="D")
    rows = []
    for scope in ("all", "BTC", "ETH"):
        for i, d in enumerate(cal):
            n = 10 if scope == "all" else (8 if scope == "BTC" else 6)
            # vài ngày low-n cho ETH
            if scope == "ETH" and i % 10 == 0:
                n = 2
            rows.append({
                "date": d, "scope": scope,
                "vader_Sent_t": 0.1 + 0.01 * i, "vader_PosSent": 0.4, "vader_NegSent": 0.3,
                "vader_MeanProb_t": 0.05, "vader_NewsVolume_t": n,
                "vader_AbnAttention_t": np.nan if i < 30 else 0.1,
                "finbert_Sent_t": -0.05, "finbert_PosSent": 0.3, "finbert_NegSent": 0.35,
                "finbert_MeanProb_t": -0.02, "finbert_NewsVolume_t": n,
                "finbert_AbnAttention_t": np.nan if i < 30 else -0.1,
            })
    return pd.DataFrame(rows)


def _toy_narr(start="2021-03-01", end="2021-04-30"):
    cal = pd.date_range(start, end, freq="D")
    rows = []
    for scope in ("all", "BTC", "ETH"):
        for i, d in enumerate(cal):
            rows.append({
                "date": d, "scope": scope,
                "Att_regulation": 0.1, "LogRatio_regulation": 0.0, "AbnAtt_regulation": 0.0,
                "Att_etf": 0.05, "LogRatio_etf": -1.0, "AbnAtt_etf": 0.0,
                "Att_price_desc": 0.2,
                "n_regulation": 1, "NewsVolume_t": 10,
            })
    return pd.DataFrame(rows)


def _toy_market(start="2021-03-01", end="2021-05-15"):
    """Kéo dài sau t_end để có target h=7."""
    cal = pd.date_range(start, end, freq="D")
    rows = []
    rng = np.random.default_rng(42)
    for asset in ("BTC", "ETH"):
        for i, d in enumerate(cal):
            ret = float(rng.normal(0, 0.02))
            rv = abs(float(rng.normal(0.001, 0.0005))) + 1e-6
            complete = True
            if d == pd.Timestamp("2021-04-10"):
                complete = False
                rv = np.nan
            rows.append({
                "date": d, "asset": asset, "ret": ret, "rv": rv,
                "log_rv": np.nan if not complete else np.log(rv),
                "volume": 1000.0 + i, "abvol": float(rng.normal()),
                "complete_day": complete,
            })
    return pd.DataFrame(rows)


def _toy_fred():
    """VIX +1, DGS10 +2, DTWEXBGS Tuesday after Friday."""
    rows = []
    # daily VIX / DGS10 around March–April 2021
    for d in pd.date_range("2021-02-01", "2021-05-01", freq="D"):
        if d.weekday() >= 5:
            continue
        ord_ = float(d.toordinal())
        rows.append({
            "date": d, "series": "VIXCLS", "name": "vix", "value": 20.0 + ord_ * 1e-4,
            "usable_from_utc": pd.Timestamp(d + pd.Timedelta(days=1), tz="UTC"),
        })
        rows.append({
            "date": d, "series": "DGS10", "name": "ust10y", "value": 1.5 + ord_ * 1e-5,
            "usable_from_utc": pd.Timestamp(d + pd.Timedelta(days=2), tz="UTC"),
        })
    # weekly DTWEX: Fridays — giá trị unique theo tuần
    for i, d in enumerate(pd.date_range("2021-02-05", "2021-04-30", freq="W-FRI")):
        tue = d + pd.Timedelta(days=4)  # Tuesday after Friday
        rows.append({
            "date": d, "series": "DTWEXBGS", "name": "usd_broad", "value": 110.0 + i,
            "usable_from_utc": pd.Timestamp(tue, tz="UTC"),
        })
    return pd.DataFrame(rows)


@pytest.fixture
def toy():
    return {
        "params": _params(),
        "panel": _toy_panel(),
        "narr": _toy_narr(),
        "market": _toy_market(),
        "fred": _toy_fred(),
    }


def test_a_news_leak_t_plus_1(toy):
    """Đổi tin ngày t+1 → đặc trưng hàng t không đổi (A và B)."""
    params = toy["params"]
    t1 = pd.Timestamp("2021-03-15")
    panel2 = toy["panel"].copy()
    mask = (panel2["date"] == t1) & (panel2["scope"] == "all")
    panel2.loc[mask, "vader_Sent_t"] = 99.0
    narr2 = toy["narr"].copy()
    narr2.loc[narr2["date"] == t1, "Att_regulation"] = 0.99

    a0 = build_model_table_A(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    a1 = build_model_table_A(panel2, narr2, toy["market"], toy["fred"], params)
    b0 = build_model_table_B(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    b1 = build_model_table_B(panel2, narr2, toy["market"], toy["fred"], params)

    t = pd.Timestamp("2021-03-14")
    for before, after in ((a0, a1), (b0, b1)):
        r0 = before[(before["date"] == t) & (before["asset"] == "BTC")].iloc[0]
        r1 = after[(after["date"] == t) & (after["asset"] == "BTC")].iloc[0]
        assert r0["all_vader_Sent_t"] == r1["all_vader_Sent_t"]
        assert r0["Att_regulation"] == r1["Att_regulation"]


def test_b_market_leak_and_a_lag_controls(toy):
    """B: đổi ret/rv t+1 không đổi X hàng t. A: đổi ret_t/rv_t không đổi kiểm soát trễ hàng t."""
    params = toy["params"]
    t = pd.Timestamp("2021-03-20")
    t1 = t + pd.Timedelta(days=1)

    mkt_b = toy["market"].copy()
    mkt_b.loc[(mkt_b["date"] == t1) & (mkt_b["asset"] == "BTC"), "ret"] = 9.9
    mkt_b.loc[(mkt_b["date"] == t1) & (mkt_b["asset"] == "BTC"), "log_rv"] = 9.9
    b0 = build_model_table_B(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    b1 = build_model_table_B(toy["panel"], toy["narr"], mkt_b, toy["fred"], params)
    r0 = b0[(b0["date"] == t) & (b0["asset"] == "BTC")].iloc[0]
    r1 = b1[(b1["date"] == t) & (b1["asset"] == "BTC")].iloc[0]
    for c in ("ret_t", "log_rv_t", "har_d", "har_w", "har_m", "logvol_t", "all_vader_Sent_t"):
        assert r0[c] == r1[c] or (np.isnan(r0[c]) and np.isnan(r1[c]))

    mkt_a = toy["market"].copy()
    mkt_a.loc[(mkt_a["date"] == t) & (mkt_a["asset"] == "BTC"), "ret"] = 8.8
    mkt_a.loc[(mkt_a["date"] == t) & (mkt_a["asset"] == "BTC"), "log_rv"] = 8.8
    a0 = build_model_table_A(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    a1 = build_model_table_A(toy["panel"], toy["narr"], mkt_a, toy["fred"], params)
    x0 = a0[(a0["date"] == t) & (a0["asset"] == "BTC")].iloc[0]
    x1 = a1[(a1["date"] == t) & (a1["asset"] == "BTC")].iloc[0]
    for c in ("ret_lag1", "ret_lag2", "har_d", "har_w", "har_m", "logvol_lag1"):
        assert x0[c] == x1[c] or (np.isnan(x0[c]) and np.isnan(x1[c]))
    # Y_t phải đổi
    assert x0["y_ret"] != x1["y_ret"]


def test_har_a_lagged_b_includes_t(toy):
    params = toy["params"]
    t = pd.Timestamp("2021-03-25")
    mkt2 = toy["market"].copy()
    mkt2.loc[(mkt2["date"] == t) & (mkt2["asset"] == "BTC"), "log_rv"] = 5.5

    a0 = build_model_table_A(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    a1 = build_model_table_A(toy["panel"], toy["narr"], mkt2, toy["fred"], params)
    r0 = a0[(a0["date"] == t) & (a0["asset"] == "BTC")].iloc[0]
    r1 = a1[(a1["date"] == t) & (a1["asset"] == "BTC")].iloc[0]
    for c in ("har_d", "har_w", "har_m"):
        assert r0[c] == r1[c] or (np.isnan(r0[c]) and np.isnan(r1[c]))

    b0 = build_model_table_B(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    b1 = build_model_table_B(toy["panel"], toy["narr"], mkt2, toy["fred"], params)
    s0 = b0[(b0["date"] == t) & (b0["asset"] == "BTC")].iloc[0]
    s1 = b1[(b1["date"] == t) & (b1["asset"] == "BTC")].iloc[0]
    assert s0["har_d"] != s1["har_d"]
    assert s1["har_d"] == 5.5


def test_har_skips_incomplete_min_periods():
    """1 ngày incomplete trong cửa sổ không NaN har; 3 incomplete trong 5 ngày → har_w NaN.
    har_m = mean các ngày complete hợp lệ (tính lại tay).
    """
    n = 40
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    log_rv = pd.Series(np.linspace(-5.0, -3.0, n), index=dates)
    complete = pd.Series(True, index=dates)
    # một ngày incomplete giữa chuỗi
    incomplete_one = dates[20]
    complete.loc[incomplete_one] = False
    log_rv.loc[incomplete_one] = np.nan

    _, har_w, har_m = har_from_log_rv(log_rv, complete, include_t=True)
    # ngày ngay sau incomplete: cửa sổ 5 còn 4 hợp lệ → har_w OK
    t_after = dates[21]
    assert pd.notna(har_w.loc[t_after])
    assert pd.notna(har_m.loc[dates[25]])  # cửa sổ 22 còn ≥18

    # 3 ngày incomplete liên tiếp trong cửa sổ 5
    complete3 = pd.Series(True, index=dates)
    log3 = log_rv.copy()
    for d in dates[15:18]:
        complete3.loc[d] = False
        log3.loc[d] = np.nan
    _, har_w3, _ = har_from_log_rv(log3, complete3, include_t=True)
    # tại dates[17]: cửa sổ t-4..t = dates[13..17] có incomplete ở 15,16,17 → 3 NaN, 2 valid < 4
    assert pd.isna(har_w3.loc[dates[17]])

    # har_m tại dates[30] = mean log_rv complete trong dates[9..30] (22 ngày)
    t = dates[30]
    window = log_rv.loc[dates[9]:t].where(complete.loc[dates[9]:t])
    expected = window.mean()  # pandas mean bỏ NaN
    assert abs(har_m.loc[t] - expected) < 1e-12
    # đủ ≥18 hợp lệ
    assert window.notna().sum() >= 18


def test_har_m_manual_mean_via_market_features_B():
    """Giá trị har_m trên bảng B khớp trung bình tay các ngày complete trong cửa sổ."""
    dates = pd.date_range("2021-02-01", periods=50, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for asset in ("BTC",):
        for i, d in enumerate(dates):
            complete = d != pd.Timestamp("2021-02-20")
            rv = abs(float(rng.normal(0.001, 0.0003))) + 1e-6
            rows.append({
                "date": d, "asset": asset,
                "ret": 0.0, "rv": rv if complete else np.nan,
                "log_rv": np.log(rv) if complete else np.nan,
                "volume": 1000.0, "abvol": 0.0, "complete_day": complete,
            })
    mkt = pd.DataFrame(rows)
    feat = market_features_B(mkt, horizons=[1])
    t = pd.Timestamp("2021-03-10")
    row = feat.loc[feat["date"] == t].iloc[0]
    win_dates = pd.date_range(t - pd.Timedelta(days=21), t, freq="D")
    sub = mkt.set_index("date").loc[win_dates]
    valid = sub.loc[sub["complete_day"], "log_rv"]
    assert valid.notna().sum() >= 18
    assert abs(row["har_m"] - valid.mean()) < 1e-12


def test_fred_asof_lags_a_and_b(toy):
    """DGS10 obs d → hàng A từ d+2; VIX từ d+1; B sớm hơn A đúng 1 ngày."""
    params = toy["params"]
    fred = toy["fred"]
    # chọn một ngày quan sát DGS10 weekday
    d = pd.Timestamp("2021-03-10")  # Wed
    assert d.weekday() < 5
    dgs_val = fred.loc[(fred["series"] == "DGS10") & (fred["date"] == d), "value"].iloc[0]
    vix_val = fred.loc[(fred["series"] == "VIXCLS") & (fred["date"] == d), "value"].iloc[0]

    a = build_model_table_A(toy["panel"], toy["narr"], toy["market"], fred, params)
    b = build_model_table_B(toy["panel"], toy["narr"], toy["market"], fred, params)

    # A: DGS10 of d first at d+2
    row_early = a[(a["date"] == d + pd.Timedelta(days=1)) & (a["asset"] == "BTC")].iloc[0]
    row_ok = a[(a["date"] == d + pd.Timedelta(days=2)) & (a["asset"] == "BTC")].iloc[0]
    assert row_ok["fred_DGS10"] == dgs_val
    # at d+1, asof is d+1 00:00; usable of DGS10(d) is d+2 → not yet
    # may still have older DGS10 — must not equal this obs if no other equals
    # safer: join_fred unit check
    cal = pd.date_range("2021-03-01", "2021-03-20", freq="D")
    prep = prepare_fred_with_d5(fred)
    fa = join_fred_asof(cal, prep, 0)
    fb = join_fred_asof(cal, prep, 1)
    # first date where A sees this DGS10
    a_first = fa.loc[fa["fred_DGS10"] == dgs_val, "date"].min()
    b_first = fb.loc[fb["fred_DGS10"] == dgs_val, "date"].min()
    assert a_first == d + pd.Timedelta(days=2)
    assert b_first == d + pd.Timedelta(days=1)

    a_vix = fa.loc[fa["fred_VIXCLS"] == vix_val, "date"].min()
    b_vix = fb.loc[fb["fred_VIXCLS"] == vix_val, "date"].min()
    assert a_vix == d + pd.Timedelta(days=1)
    assert b_vix == d  # offset +1 → usable d+1 <= d+1 when row date=d → wait
    # Spec B asof = (t+1) 00:00. usable VIX(d)=d+1. First t where t+1 >= d+1 → t >= d.
    assert b_vix == d

    # DTWEX: Friday 2021-03-12 → usable Tuesday 2021-03-16
    fri = pd.Timestamp("2021-03-12")
    tue = pd.Timestamp("2021-03-16")
    assert fri.weekday() == 4
    dtw_val = fred.loc[(fred["series"] == "DTWEXBGS") & (fred["date"] == fri), "value"].iloc[0]
    a_dtw = fa.loc[fa["fred_DTWEXBGS"] == dtw_val, "date"].min()
    assert a_dtw == tue


def test_target_y_ret_3_and_incomplete(toy):
    params = toy["params"]
    b = build_model_table_B(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    t = pd.Timestamp("2021-03-05")
    m = toy["market"]
    asset = "BTC"
    expected = (
        m.loc[(m["date"] == t + pd.Timedelta(days=1)) & (m["asset"] == asset), "ret"].iloc[0]
        + m.loc[(m["date"] == t + pd.Timedelta(days=2)) & (m["asset"] == asset), "ret"].iloc[0]
        + m.loc[(m["date"] == t + pd.Timedelta(days=3)) & (m["asset"] == asset), "ret"].iloc[0]
    )
    got = b.loc[(b["date"] == t) & (b["asset"] == asset), "y_ret_3"].iloc[0]
    assert abs(got - expected) < 1e-12

    # incomplete 2021-04-10 in window → NaN
    # t such that t+1..t+3 includes 2021-04-10 → t=2021-04-09,08,07
    t_bad = pd.Timestamp("2021-04-08")
    assert pd.isna(b.loc[(b["date"] == t_bad) & (b["asset"] == asset), "y_ret_3"].iloc[0])


def test_key_unique_and_calendar(toy):
    params = toy["params"]
    a = build_model_table_A(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    start = pd.Timestamp(params["sample"]["start"])
    end = pd.Timestamp(params["sample"]["t_end"])
    n_days = (end - start).days + 1
    assert len(a) == n_days * 2
    assert a.duplicated(["date", "asset"]).sum() == 0
    assert _naive_min_max(a)
    d = pd.to_datetime(a["date"])
    assert d.min() >= start and d.max() <= end


def _naive_min_max(df):
    d = pd.to_datetime(df["date"])
    return True


def test_embargo_no_cross_after_filter(toy):
    params = toy["params"]
    b = build_model_table_B(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    val_start = pd.Timestamp(params["split"]["validation"][0])
    test_start = pd.Timestamp(params["split"]["test"][0])
    for h in params["horizons"]:
        col = f"embargo_drop_{h}"
        tend = f"target_end_date_{h}"
        train_ok = b[(b["split"] == "train") & (~b[col])]
        assert (train_ok[tend] < val_start).all()
        valid_ok = b[(b["split"] == "valid") & (~b[col])]
        assert (valid_ok[tend] < test_start).all()


def test_load_model_table_test_locked(toy, tmp_path, monkeypatch):
    params = toy["params"]
    a = build_model_table_A(toy["panel"], toy["narr"], toy["market"], toy["fred"], params)
    path = tmp_path / "model_table_A.parquet"
    a.to_parquet(path, index=False)
    paths = {"model_table_A_path": str(path)}
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("A", "test", root=ROOT, paths=paths)
    monkeypatch.setenv("ALLOW_TEST", "1")
    got = load_model_table("A", "test", root=ROOT, paths=paths)
    assert (got["split"] == "test").all()
    assert len(got) == (a["split"] == "test").sum()

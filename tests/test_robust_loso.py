"""Tests for LOSO robustness — no test split, Holm within group, exclude_sources."""
from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_daily_panel import prepare_docs  # noqa: E402
from build_narratives import filter_exclude_sources  # noqa: E402
from modeltable_io import HARD_DATE_CAP, TestSplitLockedError, load_model_table  # noqa: E402
from models.har_text import holm_adjust  # noqa: E402
from models.robust_loso import (  # noqa: E402
    EST_END,
    FAMILY,
    LOSO_SOURCES,
    compare_model_table_tv,
    feature_outcome_cols,
    loso_groups,
)
from models.rq1_rq2_sentiment import load_train_valid, run_rq1_vs_market  # noqa: E402


def test_family_and_groups():
    assert FAMILY == "robust_loso"
    g = loso_groups()
    assert g[0] == ("full", [])
    assert [x[0] for x in g[1:]] == list(LOSO_SOURCES)
    assert all(x[1] == [x[0]] for x in g[1:])


def test_prepare_docs_exclude():
    docs = pd.DataFrame({
        "doc_id": ["a", "b", "c"],
        "news_date": pd.to_datetime(["2021-03-01"] * 3),
        "source": ["coindesk", "decrypt", "cointelegraph"],
        "currencies": [["BTC"]] * 3,
        "in_main_sample": [True, True, True],
    })
    text = docs.copy()
    full = prepare_docs(text, exclude_sources=[])
    assert set(full["source"]) == {"coindesk", "decrypt", "cointelegraph"}
    drop = prepare_docs(text, exclude_sources=["decrypt"])
    assert set(drop["source"]) == {"coindesk", "cointelegraph"}
    none = prepare_docs(text, exclude_sources=None)
    assert len(none) == 3


def test_filter_exclude_sources_case():
    docs = pd.DataFrame({"source": ["CoinDesk", "decrypt"], "x": [1, 2]})
    out = filter_exclude_sources(docs, ["coindesk"])
    assert list(out["source"]) == ["decrypt"]
    assert len(filter_exclude_sources(docs, [])) == 2


def test_load_test_locked(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    with pytest.raises(TestSplitLockedError):
        load_model_table("B", "test", root=ROOT)


def test_load_train_valid_no_test_and_cap():
    p = ROOT / "data" / "processed" / "model_table_A.parquet"
    if not p.exists():
        pytest.skip("no model_table_A")
    df = load_train_valid("A", root=ROOT, exclude_sources=[])
    assert "test" not in set(df["split"].astype(str))
    d = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    assert (d >= pd.Timestamp(HARD_DATE_CAP)).sum() == 0
    assert d.max() <= pd.Timestamp(EST_END)


def test_holm_within_group_9():
    raw = [0.01, 0.02, 0.03, 0.04, 0.10, 0.20, 0.30, 0.40, 0.50]
    adj = holm_adjust(raw)
    assert len(adj) == 9
    assert adj[0] == pytest.approx(min(1.0, 0.01 * 9))
    # two groups independently
    g1 = holm_adjust([0.01, 0.04, 0.10])
    g2 = holm_adjust([0.01, 0.04, 0.10])
    assert g1 == g2
    # pooled 6 ≠ two groups of 3
    pooled = holm_adjust([0.01, 0.04, 0.10, 0.01, 0.04, 0.10])
    assert pooled[0] != g1[0]


def test_feature_outcome_skip_meta():
    df = pd.DataFrame({
        "date": [1], "asset": ["BTC"], "split": ["train"],
        "all_finbert_Sent_t": [0.1], "y_ret": [0.0],
        "embargo_drop_1": [False], "target_end_date_1": [1],
        "own_low_n": [False],
    })
    cols = feature_outcome_cols(df)
    assert "all_finbert_Sent_t" in cols
    assert "y_ret" in cols
    assert "split" not in cols
    assert "embargo_drop_1" not in cols


def test_compare_model_table_tv_tol():
    dates = pd.date_range("2021-03-01", periods=3, freq="D")
    ref = pd.DataFrame({
        "date": list(dates) + list(dates),
        "asset": ["BTC"] * 3 + ["ETH"] * 3,
        "split": ["train"] * 6,
        "all_finbert_Sent_t": [0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
        "y_ret": [0.0, 0.01, -0.01, 0.0, 0.01, -0.01],
    })
    reb = ref.copy()
    reb.loc[0, "all_finbert_Sent_t"] = 0.1 + 1e-12
    chk = compare_model_table_tv(reb, ref)
    assert chk["ok"] is True
    reb2 = ref.copy()
    reb2.loc[0, "all_finbert_Sent_t"] = 0.5
    chk2 = compare_model_table_tv(reb2, ref)
    assert chk2["ok"] is False
    assert chk2["worst_col"] == "all_finbert_Sent_t"


def test_run_rq1_vs_market_holm_6():
    from models.calendar_features import DOW_DUMMY_COLS, add_dow_dummies
    from models.rq1_rq2_sentiment import CONTROL_A, CONTROL_FRED, COL_FIN_SENT, COL_VAD_SENT

    rng = np.random.default_rng(42)
    n = 80
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    har = rng.normal(-5, 0.3, n)
    sent_f = rng.normal(0, 1, n)
    y = 0.3 * sent_f + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "ret_lag1": rng.normal(0, 0.02, n),
        "ret_lag2": rng.normal(0, 0.02, n),
        "har_d": har,
        "har_w": har + rng.normal(0, 0.05, n),
        "har_m": har + rng.normal(0, 0.08, n),
        "logvol_lag1": rng.normal(10, 0.2, n),
        COL_FIN_SENT: sent_f,
        COL_VAD_SENT: rng.normal(0, 1, n),
        "y_ret": y,
        "y_log_rv": har + 0.1 * sent_f,
        "y_abvol": rng.normal(0, 1, n),
        **{c: rng.normal(0, 1, n) for c in CONTROL_FRED},
    })
    df = add_dow_dummies(df)
    assert all(c in df.columns for c in CONTROL_A)
    out = run_rq1_vs_market(df, "BTC", n_boot=40, seed=42)
    assert len(out) == 6
    assert out["p_holm"].notna().all()
    assert set(out["model"]) == {"finbert", "vader"}
    assert set(out["y"]) == {"y_ret", "y_log_rv", "y_abvol"}


def test_full_repro_if_artifacts():
    full_b = ROOT / "data" / "processed" / "loso" / "full" / "model_table_B.parquet"
    ref = ROOT / "data" / "processed" / "model_table_B.parquet"
    if not full_b.exists() or not ref.exists():
        pytest.skip("LOSO full table chưa build")
    chk = compare_model_table_tv(pd.read_parquet(full_b), pd.read_parquet(ref))
    assert chk["ok"], chk


def test_outputs_no_test_dates_if_present():
    for name in ("robust_loso_A.csv", "robust_loso_B.csv"):
        p = ROOT / "outputs" / "tables" / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "date" in df.columns:
            d = pd.to_datetime(df["date"], errors="coerce")
            assert (d >= pd.Timestamp(HARD_DATE_CAP)).sum() == 0
        if "split" in df.columns:
            assert "test" not in set(df["split"].astype(str))
        if "family" in df.columns:
            assert set(df["family"].dropna().astype(str)) <= {FAMILY, ""}


def test_no_allow_test_env():
    assert os.environ.get("ALLOW_TEST") != "1"

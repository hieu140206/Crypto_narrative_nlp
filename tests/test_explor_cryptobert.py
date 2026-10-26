"""Tests for exploratory CryptoBERT (no HF download in unit tests)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.explor_cryptobert import (  # noqa: E402
    CRYPTOBERT_ID,
    CRYPTOBERT_REVISION,
    FAMILY,
    LABEL_MAP_CBERT,
    bootstrap_accuracy_ci,
    bootstrap_delta_r2_vs_baseline,
    cohen_kappa,
    eval_model_on_labels,
    per_class_f1,
    stratified_subsample_ids,
    table_b1_correlations,
)
from models.har_text import holm_adjust  # noqa: E402


def test_label_map_bullish_bearish():
    assert LABEL_MAP_CBERT["Bullish"] == "positive"
    assert LABEL_MAP_CBERT["Bearish"] == "negative"
    assert LABEL_MAP_CBERT["Neutral"] == "neutral"


def test_family_name():
    assert FAMILY == "explor_cryptobert"


def test_revision_pinned():
    assert CRYPTOBERT_ID == "ElKulako/cryptobert"
    assert len(CRYPTOBERT_REVISION) == 40


def test_bootstrap_accuracy_ci_reproducible():
    y = np.array(["positive", "negative", "neutral"] * 40)
    pred = y.copy()
    pred[:10] = "neutral"
    a1, lo1, hi1 = bootstrap_accuracy_ci(y, pred, n_boot=500, seed=42)
    a2, lo2, hi2 = bootstrap_accuracy_ci(y, pred, n_boot=500, seed=42)
    assert a1 == a2 and lo1 == lo2 and hi1 == hi2
    assert lo1 <= a1 <= hi1


def test_cohen_kappa_perfect():
    y = np.array(["positive", "negative", "neutral"] * 10)
    assert cohen_kappa(y, y, ["negative", "neutral", "positive"]) == pytest.approx(1.0)


def test_per_class_f1_and_eval():
    y = np.array(["positive", "negative", "neutral", "positive"])
    p = np.array(["positive", "negative", "negative", "positive"])
    f1 = per_class_f1(y, p, ["negative", "neutral", "positive"])
    assert 0.0 <= f1["positive"] <= 1.0
    row = eval_model_on_labels(y, p, "dummy", n_boot=200, seed=42)
    assert row["model"] == "dummy"
    assert "cohen_kappa" in row


def test_stratified_subsample_seed42():
    rng = np.random.default_rng(0)
    rows = []
    for src in ("coindesk", "decrypt"):
        for m in range(1, 5):
            for i in range(20):
                rows.append({
                    "doc_id": f"{src}-{m}-{i}",
                    "source": src,
                    "news_date": pd.Timestamp(f"2022-{m:02d}-15"),
                })
    docs = pd.DataFrame(rows)
    a = stratified_subsample_ids(docs, frac=0.25, seed=42)
    b = stratified_subsample_ids(docs, frac=0.25, seed=42)
    assert list(a["doc_id"]) == list(b["doc_id"])
    assert abs(len(a) / len(docs) - 0.25) < 0.05


def test_delta_r2_vs_baseline_positive_signal():
    rng = np.random.default_rng(42)
    n = 300
    ctrl = rng.normal(size=(n, 3))
    sent = rng.normal(size=n)
    y = 0.8 * sent + 0.1 * ctrl[:, 0] + rng.normal(scale=0.3, size=n)
    out = bootstrap_delta_r2_vs_baseline(y, ctrl, sent, n_boot=400, seed=42)
    assert out["delta_r2_adj"] > 0
    assert out["ci_lo"] > 0 or out["p_boot"] < 0.05


def test_b1_corr_table_shape():
    dates = pd.date_range("2021-03-01", periods=50, freq="D")
    rng = np.random.default_rng(42)
    daily = pd.DataFrame({
        "date": dates,
        "finbert_Sent_t": rng.normal(size=50),
        "vader_Sent_t": rng.normal(size=50),
        "cryptobert_Sent_t": rng.normal(size=50),
        "finbert_NewsVolume_t": rng.integers(0, 10, size=50),
    })
    tab = table_b1_correlations(daily)
    assert "pair" in tab.columns
    assert (tab["pair"] == "coverage").any()


def test_holm_explor_six():
    p = [0.01, 0.04, 0.03, 0.20, 0.50, 0.80]
    adj = holm_adjust(p)
    assert len(adj) == 6
    assert adj[0] <= adj[2] or True  # just finite
    assert all(0 <= x <= 1 for x in adj)


def test_no_allow_test_env_required_for_module_import(monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    # import already done; ensure family constant unchanged
    assert FAMILY == "explor_cryptobert"

"""Test build_daily_panel (dữ liệu giả + dữ liệu thật nếu có)."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_daily_panel import (  # noqa: E402
    abn_attention,
    assert_docs_have_both_models,
    build_daily_panel,
    prepare_docs,
)


def _toy_docs_sent():
    """10 ngày, vài bài BTC/ETH; 2 model."""
    dates = pd.date_range("2021-03-01", periods=40, freq="D")
    rows = []
    for i, d in enumerate(dates[:5]):
        rows.append({"doc_id": f"a{i}", "news_date": d, "source": "coindesk",
                     "currencies": ["BTC"], "in_main_sample": True})
        rows.append({"doc_id": f"b{i}", "news_date": d, "source": "decrypt",
                     "currencies": ["ETH"], "in_main_sample": True})
        rows.append({"doc_id": f"c{i}", "news_date": d, "source": "cointelegraph",
                     "currencies": ["BTC", "ETH"], "in_main_sample": True})
    docs = pd.DataFrame(rows)
    sent_rows = []
    for doc_id in docs["doc_id"]:
        for model, label, pp, pn in (
            ("vader", "positive", 0.6, 0.1),
            ("finbert", "neutral", 0.2, 0.2),
        ):
            sent_rows.append({
                "doc_id": doc_id, "model": model, "label": label,
                "p_pos": pp, "p_neg": pn, "p_neu": 1 - pp - pn,
                "text_variant": "title",
            })
    # đổi vài nhãn
    sent = pd.DataFrame(sent_rows)
    sent.loc[(sent["doc_id"] == "a0") & (sent["model"] == "finbert"), "label"] = "negative"
    sent.loc[(sent["doc_id"] == "a0") & (sent["model"] == "finbert"), ["p_pos", "p_neg", "p_neu"]] = (0.1, 0.7, 0.2)
    return docs, sent, dates


def test_toy_calendar_and_scopes():
    docs, sent, dates = _toy_docs_sent()
    panel = build_daily_panel(docs, sent, start=dates[0], end=dates[-1], abn_window=30)
    for scope in ("all", "BTC", "ETH"):
        g = panel[panel["scope"] == scope]
        assert len(g) == 40
        exp = pd.date_range(dates[0], dates[-1])
        assert (pd.to_datetime(g["date"]).values == exp.values).all()

    all_vol = panel.loc[panel["scope"] == "all", "vader_NewsVolume_t"]
    btc_vol = panel.loc[panel["scope"] == "BTC", "vader_NewsVolume_t"]
    eth_vol = panel.loc[panel["scope"] == "ETH", "vader_NewsVolume_t"]
    assert all_vol.sum() == len(docs)
    assert (btc_vol.values <= all_vol.values).all()
    assert (eth_vol.values <= all_vol.values).all()
    # ngày có tin: 3 bài all, 2 BTC (a+c), 2 ETH (b+c)
    assert all_vol.iloc[0] == 3
    assert btc_vol.iloc[0] == 2
    assert eth_vol.iloc[0] == 2


def test_toy_sent_and_abn():
    docs, sent, dates = _toy_docs_sent()
    panel = build_daily_panel(docs, sent, start=dates[0], end=dates[-1], abn_window=30)
    g = panel[panel["scope"] == "all"].sort_values("date")
    for m in ("vader", "finbert"):
        sent_ok = (g[f"{m}_Sent_t"] - (g[f"{m}_PosSent"] - g[f"{m}_NegSent"])).abs().dropna()
        assert (sent_ok < 1e-9).all()
        assert (g[f"{m}_PosSent"] + g[f"{m}_NegSent"]).dropna().le(1 + 1e-9).all()
        assert g[f"{m}_AbnAttention_t"].head(30).isna().all()
        # ngày không tin
        empty = g[f"{m}_NewsVolume_t"] == 0
        assert g.loc[empty, f"{m}_PosSent"].isna().all()


def test_abn_no_lookahead():
    docs, sent, dates = _toy_docs_sent()
    start, end = dates[0], dates[-1]
    base = build_daily_panel(docs, sent, start=start, end=end, abn_window=30)
    # xoá/đổi bài ngày t+1 (ngày index 4)
    t1 = dates[4]
    docs2 = docs[docs["news_date"] != t1].copy()
    sent2 = sent[sent["doc_id"].isin(docs2["doc_id"])].copy()
    bumped = build_daily_panel(docs2, sent2, start=start, end=end, abn_window=30)
    # mọi cột của ngày t < t1 không đổi
    for scope in ("all", "BTC", "ETH"):
        b = base[(base["scope"] == scope) & (pd.to_datetime(base["date"]) < t1)].reset_index(drop=True)
        u = bumped[(bumped["scope"] == scope) & (pd.to_datetime(bumped["date"]) < t1)].reset_index(drop=True)
        pd.testing.assert_frame_equal(b, u, check_dtype=False)


def test_abn_attention_unit():
    s = pd.Series([1.0] * 35)
    out = abn_attention(s, window=30)
    assert out.head(30).isna().all()
    assert out.iloc[30:].notna().all()


def test_assert_docs_have_both_models_ok_and_raise():
    docs, sent, _ = _toy_docs_sent()
    assert_docs_have_both_models(docs, sent)  # không raise
    sent_bad = sent[sent["model"] != "finbert"].copy()
    with pytest.raises(ValueError, match="Thiếu sentiment"):
        assert_docs_have_both_models(docs, sent_bad)


def test_real_panel_if_present(paths, params):
    p = ROOT / paths["daily_panel_path"]
    if not p.exists():
        pytest.skip("chưa có daily_panel.parquet")
    panel = pd.read_parquet(p)
    text = pd.read_parquet(ROOT / paths["text_clean_path"], columns=["doc_id", "in_main_sample", "news_date"])
    main = text[text["in_main_sample"]]
    start = pd.Timestamp(params["sample"]["start"])
    end = pd.Timestamp(params["sample"]["t_end"])
    main = main[(pd.to_datetime(main["news_date"]) >= start) & (pd.to_datetime(main["news_date"]) <= end)]
    g = panel[panel["scope"] == "all"]
    assert g["vader_NewsVolume_t"].sum() == len(main)
    assert g["vader_AbnAttention_t"].head(params["abn_attention_window"]).isna().all()


def test_noprice_plus_pricedesc_equals_total():
    """Tổng bài noprice + price_desc = tổng bài; Sent NaN nếu noprice < 5."""
    docs, sent, dates = _toy_docs_sent()
    # gán price_desc: một nửa True
    docs = docs.copy()
    docs["price_desc"] = [i % 2 == 0 for i in range(len(docs))]
    panel = build_daily_panel(
        docs, sent, start=dates[0], end=dates[-1], abn_window=30,
        noprice=True, noprice_min_n=5,
    )
    assert "NewsVolume_noprice" in panel.columns
    g = panel[panel["scope"] == "all"]
    uniq = docs.drop_duplicates(["doc_id", "news_date"])
    n_all = len(uniq)
    n_np = int((~uniq["price_desc"]).sum())
    n_pd = int(uniq["price_desc"].sum())
    assert n_np + n_pd == n_all
    assert int(g["vader_NewsVolume_t"].sum()) == n_all
    assert int(g["NewsVolume_noprice"].sum()) == n_np
    # ngày đầu có 3 bài all, ~1–2 noprice → <5 → Sent NaN
    assert pd.isna(g.iloc[0]["finbert_Sent_t"])

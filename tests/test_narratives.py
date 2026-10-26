"""Test narrative doc + daily features (dict v4)."""
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_narratives import (  # noqa: E402
    assign_doc_narratives,
    build_daily_narratives,
    is_template_price,
    load_narrative_config,
)


CFG = load_narrative_config(ROOT / "config" / "narratives.yaml")
NARR = {
    n: {"patterns": v["patterns"], "excludes": v["excludes"]}
    for n, v in CFG["narratives"].items()
}
FLAGS = CFG["flags"]
NAMES = list(NARR.keys())


def _toy():
    titles = pd.Series([
        "SEC sues crypto exchange over compliance failure",
        "BlackRock spot Bitcoin ETF sees record inflows",
        "Major DeFi protocol hacked, funds drained",
        "Fed rate hike and CPI inflation surprise markets",
        "Random chart with no narrative keywords here xyz",
        "USDT stablecoin depeg fears after Tether news",
    ])
    docs = pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(len(titles))],
        "title": titles,
        "news_date": pd.to_datetime([
            "2021-04-01", "2021-04-01", "2021-04-02",
            "2021-04-02", "2021-04-03", "2021-04-03",
        ]),
        "currencies": [["BTC"], ["BTC", "ETH"], ["ETH"], ["BTC"], ["ETH"], ["BTC"]],
    })
    assigned = assign_doc_narratives(docs["title"], NARR, flags=FLAGS)
    return pd.concat([docs, assigned], axis=1)


def test_multilabel_and_other():
    docs = _toy()
    assert docs.loc[0, "regulation"]
    assert docs.loc[1, "etf"]
    assert docs.loc[2, "hacks_security"] or docs.loc[2, "defi"]
    assert docs.loc[4, "other"]
    matched = ~docs["other"]
    assert docs.loc[matched, NAMES].sum(axis=1).ge(1).all()
    assert docs["n_narratives"].eq(docs[NAMES].sum(axis=1)).all()


def test_price_breach_not_hacks():
    """'Bitcoin Breaches $30K' là diễn biến giá — không khớp hacks_security."""
    titles = pd.Series(["Bitcoin Breaches $30K"])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert not bool(out.loc[0, "hacks_security"])
    assert bool(out.loc[0, "price_desc"])


def test_strategy_buys_bitcoin_institutional():
    titles = pd.Series(["Strategy buys Bitcoin for treasury"])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert bool(out.loc[0, "institutional_adoption"])


def test_template_price_clears_all_narratives():
    """is_template_price → không thuộc narrative nào; other=1; price_desc=1."""
    titles = pd.Series([
        "Price analysis: Bitcoin and Ethereum outlook",
        "Price predictions for crypto this week",
        "Price prediction BTC weekly",
        "SPX, DXY, BTC, ETH weekly snapshot",
        "SEC ETF approval and rate hike Powell",  # không template
    ])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    for i in range(4):
        assert bool(out.loc[i, "is_template_price"]), titles[i]
        assert bool(out.loc[i, "other"]), titles[i]
        assert bool(out.loc[i, "price_desc"]), titles[i]
        assert int(out.loc[i, "n_narratives"]) == 0
        assert not out.loc[i, NAMES].any()
    assert not bool(out.loc[4, "is_template_price"])
    assert out.loc[4, NAMES].any()


def test_is_template_price_helper():
    assert is_template_price("Price analysis of BTC")
    assert is_template_price("Price Predictions: weekly")
    assert is_template_price("SPX, DXY, BTC, ETH move higher")
    assert not is_template_price("BTC ETF inflows rise")
    assert not is_template_price("SPX, BTC rally")  # chỉ 2 ticker


def test_excludes_hack_vc_and_sec_filing():
    titles = pd.Series([
        "Hack VC leads new crypto fund raise",
        "Company files S-1 with the SEC for IPO",
        "Exchange hacked, $50M stolen",
    ])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert not bool(out.loc[0, "hacks_security"])
    assert not bool(out.loc[1, "regulation"])
    assert bool(out.loc[2, "hacks_security"])


def test_ai_general_not_howto_chatgpt():
    assert "ai_general" in NAMES
    assert "ai_tokens" not in NAMES
    titles = pd.Series([
        "How to use ChatGPT prompts for trading",
        "ChatGPT picks top crypto for 2024",
        "New GPU mining farm opens in Texas",
        "GPU token AI agent launches on chain",
        "New AI token launches on Bittensor network",
    ])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert not bool(out.loc[0, "ai_general"])
    assert not bool(out.loc[1, "ai_general"])
    assert not bool(out.loc[2, "ai_general"])
    assert bool(out.loc[3, "ai_general"])
    assert bool(out.loc[4, "ai_general"])


def test_cbdc_separate_from_monetary():
    assert "cbdc" in NAMES
    titles = pd.Series([
        "China expands digital yuan e-CNY pilot",
        "Fed discusses CBDC and digital dollar",
        "Powell signals rate cut after CPI print",
    ])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert bool(out.loc[0, "cbdc"])
    assert not bool(out.loc[0, "monetary_policy"])
    assert bool(out.loc[1, "cbdc"])
    assert not bool(out.loc[1, "monetary_policy"])
    assert bool(out.loc[2, "monetary_policy"])
    assert not bool(out.loc[2, "cbdc"])


def test_regulation_excludes_coinbase_intl_and_private_lawsuit():
    titles = pd.Series([
        "Coinbase International launches new perpetual futures",
        "Acme Corp sues rival Inc over trademark",
        "SEC charges exchange with unregistered securities",
        "FCA bans misleading crypto ads in UK",
    ])
    out = assign_doc_narratives(titles, NARR, flags=FLAGS)
    assert not bool(out.loc[0, "regulation"])
    assert not bool(out.loc[1, "regulation"])
    assert bool(out.loc[2, "regulation"])
    assert bool(out.loc[3, "regulation"])


def test_daily_att_calendar_and_bounds():
    docs = _toy()
    daily = build_daily_narratives(
        docs, start="2021-04-01", end="2021-05-15",
        narrative_names=NAMES, abn_window=30, flag_names=["price_desc"],
    )
    for scope in ("all", "BTC", "ETH"):
        g = daily[daily["scope"] == scope]
        exp = pd.date_range("2021-04-01", "2021-05-15")
        assert len(g) == len(exp)
        assert (pd.to_datetime(g["date"]).values == exp.values).all()
    g = daily[daily["scope"] == "all"]
    for n in NAMES:
        att = g[f"Att_{n}"]
        assert att.dropna().between(0, 1).all()
        empty = g["NewsVolume_t"] == 0
        assert att.loc[empty].isna().all()
        assert g[f"AbnAtt_{n}"].head(30).isna().all()
    assert "Att_price_desc" in g.columns
    assert g["Att_price_desc"].dropna().between(0, 1).all()


def test_no_lookahead_narrative():
    docs = _toy()
    start, end = "2021-04-01", "2021-05-15"
    base = build_daily_narratives(
        docs, start, end, NAMES, abn_window=30, flag_names=["price_desc"]
    )
    docs2 = docs[docs["news_date"] != pd.Timestamp("2021-04-03")].copy()
    bumped = build_daily_narratives(
        docs2, start, end, NAMES, abn_window=30, flag_names=["price_desc"]
    )
    cut = pd.Timestamp("2021-04-03")
    for scope in ("all", "BTC", "ETH"):
        b = base[(base["scope"] == scope) & (pd.to_datetime(base["date"]) < cut)].reset_index(drop=True)
        u = bumped[(bumped["scope"] == scope) & (pd.to_datetime(bumped["date"]) < cut)].reset_index(drop=True)
        pd.testing.assert_frame_equal(b, u, check_dtype=False)


def test_real_daily_narratives_if_present(paths, params):
    key = "daily_narratives_path"
    if key not in paths:
        pytest.skip("paths thiếu daily_narratives_path")
    p = ROOT / paths[key]
    if not p.exists():
        pytest.skip("chưa có daily_narratives.parquet")
    daily = pd.read_parquet(p)
    start = pd.Timestamp(params["sample"]["start"])
    end = pd.Timestamp(params["sample"]["t_end"])
    g = daily[daily["scope"] == "all"]
    exp = pd.date_range(start, end)
    assert len(g) == len(exp)
    for n in NAMES:
        if f"Att_{n}" not in g.columns:
            continue
        assert g[f"Att_{n}"].dropna().between(0, 1).all()
        assert g[f"AbnAtt_{n}"].head(params["abn_attention_window"]).isna().all()
    assert "Att_price_desc" in g.columns
    # cột cũ ai_tokens không còn
    assert "Att_ai_tokens" not in daily.columns or "Att_ai_general" in daily.columns

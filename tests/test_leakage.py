"""Test chống leakage và chất lượng. Chạy: pytest -q (bỏ qua test nếu chưa có file)."""
import re
import pandas as pd

FORBIDDEN = {"Open", "High", "Low", "Close", "Volume", "Movement_OpenClose_%",
             "Movement_HighLow_%", "Market_Move", "sentiment_label", "sentiment_score"}


def test_no_price_or_raw_sentiment_columns(text):
    assert not FORBIDDEN & set(text.columns)
    bad = [c for c in text.columns if "sentiment" in c.lower() and not c.startswith("cv_ref_")]
    assert not bad, bad


def test_window_matches_params(text, params):
    start = pd.Timestamp(params["sample"]["start"], tz="UTC")
    end = pd.Timestamp(params["sample"]["t_end"], tz="UTC") + pd.Timedelta(days=1)
    t = text["published_at_utc"]
    assert t.max() < end, f"có bài sau T_END: max={t.max()}"
    assert text["in_main_sample"].nunique() > 1 or t.min() >= start, "in_main_sample luôn True nhưng dữ liệu có trước ngày bắt đầu"
    main = text[text["in_main_sample"]]
    assert len(main) > 0
    assert main["published_at_utc"].min() >= start
    assert not main["source"].isin(params["exclude_sources_main"]).any(), "nguồn bị loại còn trong mẫu chính"


def test_news_date_and_use_date(text):
    assert "news_date" in text.columns, "thiếu cột news_date (spec A cần)"
    nd = pd.to_datetime(text["news_date"])
    assert (nd == text["published_at_utc"].dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()).all()
    assert ((pd.to_datetime(text["use_date"]) - nd).dt.days == 1).all()


def test_no_dump_days_in_main_sample(text, params):
    t = text["published_at_utc"]
    m = text[(t >= pd.Timestamp(params["sample"]["start"], tz="UTC"))
             & (t < pd.Timestamp(params["sample"]["t_end"], tz="UTC") + pd.Timedelta(days=1))]
    m = m[~m["source"].isin(params["exclude_sources_main"])]
    per = m.groupby(["source", m["published_at_utc"].dt.date]).size()
    assert per.max() <= params["dump_max_per_source_day"], per.sort_values().tail(3).to_dict()


def test_no_junk_titles(text, params):
    low = text["title"].str.strip().str.lower()
    junk = low.isin([j for j in params["junk_titles"]]) | low.str.contains(r"server error")
    assert not junk.any(), text.loc[junk, "title"].head().tolist()
    assert (text["title"].str.len() >= params["min_title_chars"]).all()


def test_doc_id_unique(text):
    assert text["doc_id"].is_unique


def test_panel_abn_attention_nan_at_start(panel, params):
    cols = [c for c in panel.columns if c.endswith("AbnAttention_t")]
    assert cols, "panel thiếu AbnAttention_t"
    p = panel.sort_values("date")
    for c in cols:
        head = p[c].head(params["abn_attention_window"])
        assert head.isna().all(), f"{c}: 30 ngày đầu phải là NaN, không phải 0"


def test_panel_sent_consistency(panel):
    for m in ("vader", "finbert", "cryptobert"):
        if f"{m}_Sent_t" in panel:
            ok = (panel[f"{m}_Sent_t"] - (panel[f"{m}_PosSent"] - panel[f"{m}_NegSent"])).abs().dropna() < 1e-9
            assert ok.all()
            assert (panel[f"{m}_PosSent"] + panel[f"{m}_NegSent"]).dropna().le(1 + 1e-9).all()


def test_panel_complete_calendar(panel, params):
    # panel lịch sử (2021-03→T_END). Panel RSS holdout nằm file riêng, không kiểm ở đây.
    d = pd.to_datetime(panel["date"]).drop_duplicates().sort_values()
    exp = pd.date_range(d.min(), d.max())
    assert len(exp) == len(d), "panel thiếu ngày"
    assert d.max() <= pd.Timestamp(params["sample"]["t_end"]), "panel vượt T_END"


def test_panel_no_forbidden_columns(panel):
    assert not any(c.startswith("cv_ref_") or c in FORBIDDEN for c in panel.columns)

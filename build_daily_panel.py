"""build_daily_panel.py - Gộp sentiment cấp bài → daily_panel (TV3).

Bản chính (title): data/processed/daily_panel.parquet
Robustness (nlp_text): data/processed/daily_panel_fulltext.parquet

Khoá: date (= news_date, UTC naive) × scope ∈ {all, BTC, ETH}.
Mỗi model m ∈ {vader, finbert}:
  m_PosSent, m_NegSent = tỷ lệ label positive/negative trong ngày
  m_Sent_t = PosSent − NegSent
  m_NewsVolume_t = số bài (giống nhau giữa các model)
  m_MeanProb_t = mean(p_pos − p_neg)
  m_AbnAttention_t = log(1+N_t) − mean(log(1+N) của t−1..t−30); 30 ngày đầu NaN

Ngày không tin: NewsVolume=0, Sent/Pos/Neg/MeanProb=NaN.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)

MODELS = ("vader", "finbert")
SCOPES = ("all", "BTC", "ETH")


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_config(root: Path) -> tuple[dict, dict]:
    import yaml
    params = yaml.safe_load((root / "config" / "params.yaml").read_text(encoding="utf-8")) or {}
    paths = yaml.safe_load((root / "config" / "paths.yaml").read_text(encoding="utf-8")) or {}
    return params, paths


def abs_path(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def currency_has(currencies, tag: str) -> bool:
    """True nếu currencies (list/ndarray/None) chứa tag BTC/ETH."""
    if currencies is None or (isinstance(currencies, float) and np.isnan(currencies)):
        return False
    try:
        if isinstance(currencies, str):
            return tag in currencies.upper()
        return tag in {str(x).upper() for x in list(currencies)}
    except TypeError:
        return False


def filter_scope(docs: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all":
        return docs
    mask = docs["currencies"].map(lambda c: currency_has(c, scope))
    return docs.loc[mask]


def abn_attention(news_volume: pd.Series, window: int = 30) -> pd.Series:
    """AbnAttention_t = log(1+N_t) − mean(log(1+N)_{t-1..t-window}); đầu cửa sổ = NaN."""
    logn = np.log1p(news_volume.astype(float))
    past_mean = logn.shift(1).rolling(window, min_periods=window).mean()
    return logn - past_mean


def _agg_model_frame(merged: pd.DataFrame, model: str) -> pd.DataFrame:
    """Gộp theo news_date cho một model."""
    sub = merged[merged["model"] == model]
    if sub.empty:
        return pd.DataFrame(columns=[
            "date", f"{model}_PosSent", f"{model}_NegSent", f"{model}_Sent_t",
            f"{model}_MeanProb_t",
        ])
    g = sub.groupby("news_date", sort=True)
    pos = g["label"].apply(lambda s: (s == "positive").mean())
    neg = g["label"].apply(lambda s: (s == "negative").mean())
    mean_prob = g.apply(lambda x: (x["p_pos"] - x["p_neg"]).mean(), include_groups=False)
    out = pd.DataFrame({
        "date": pos.index,
        f"{model}_PosSent": pos.values,
        f"{model}_NegSent": neg.values,
        f"{model}_Sent_t": (pos - neg).values,
        f"{model}_MeanProb_t": mean_prob.reindex(pos.index).values,
    })
    return out


def build_scope_panel(
    docs: pd.DataFrame,
    sent: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    scope: str,
    models: tuple[str, ...] = MODELS,
    abn_window: int = 30,
    *,
    noprice: bool = False,
    noprice_min_n: int = 5,
) -> pd.DataFrame:
    """Thuần: docs (doc_id, news_date, currencies, source) + sent → panel một scope.

    noprice=True: sentiment chỉ trên price_desc==0; NewsVolume_t = mọi bài;
    thêm NewsVolume_noprice; Sent/Pos/Neg/MeanProb = NaN nếu noprice < noprice_min_n.
    """
    scoped = filter_scope(docs, scope)
    merged = scoped.merge(sent, on="doc_id", how="inner")
    base = pd.DataFrame({"date": calendar})
    base["scope"] = scope

    if merged.empty:
        for m in models:
            base[f"{m}_PosSent"] = np.nan
            base[f"{m}_NegSent"] = np.nan
            base[f"{m}_Sent_t"] = np.nan
            base[f"{m}_NewsVolume_t"] = 0
            base[f"{m}_MeanProb_t"] = np.nan
        if noprice:
            base["NewsVolume_noprice"] = 0
        out = base
    else:
        merged = merged.copy()
        merged["news_date"] = pd.to_datetime(merged["news_date"]).dt.normalize()
        # NewsVolume gốc = mọi bài
        vol = (
            merged.drop_duplicates(["doc_id", "news_date"])
            .groupby("news_date")["doc_id"].size()
            .rename("NewsVolume")
        )
        out = base.merge(vol.rename_axis("date").reset_index(), on="date", how="left")
        out["NewsVolume"] = out["NewsVolume"].fillna(0).astype(int)

        if noprice:
            if "price_desc" not in merged.columns:
                raise KeyError("noprice yêu cầu cột price_desc trên docs/merged")
            # unique docs for volume counts
            uniq = merged.drop_duplicates(["doc_id", "news_date"])
            vol_np = (
                uniq.loc[~uniq["price_desc"].astype(bool)]
                .groupby("news_date")["doc_id"].size()
                .rename("NewsVolume_noprice")
            )
            out = out.merge(vol_np.rename_axis("date").reset_index(), on="date", how="left")
            out["NewsVolume_noprice"] = out["NewsVolume_noprice"].fillna(0).astype(int)
            sent_src = merged.loc[~merged["price_desc"].astype(bool)]
        else:
            sent_src = merged

        for m in models:
            mf = _agg_model_frame(sent_src, m)
            out = out.merge(mf, on="date", how="left")
            out[f"{m}_NewsVolume_t"] = out["NewsVolume"]
            empty = out["NewsVolume"] == 0
            for col in (f"{m}_PosSent", f"{m}_NegSent", f"{m}_Sent_t", f"{m}_MeanProb_t"):
                out.loc[empty, col] = np.nan
            if noprice:
                low = out["NewsVolume_noprice"] < int(noprice_min_n)
                for col in (f"{m}_PosSent", f"{m}_NegSent", f"{m}_Sent_t", f"{m}_MeanProb_t"):
                    out.loc[low, col] = np.nan
        out = out.drop(columns=["NewsVolume"])

    out = out.sort_values("date").reset_index(drop=True)
    ref_vol = out[f"{models[0]}_NewsVolume_t"]
    abn = abn_attention(ref_vol, window=abn_window)
    for m in models:
        out[f"{m}_AbnAttention_t"] = abn.values
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out


def build_daily_panel(
    docs: pd.DataFrame,
    sent: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    models: tuple[str, ...] = MODELS,
    scopes: tuple[str, ...] = SCOPES,
    abn_window: int = 30,
    *,
    noprice: bool = False,
    noprice_min_n: int = 5,
) -> pd.DataFrame:
    """Thuần: trả panel đầy đủ date×scope trong [start, end]."""
    start = pd.Timestamp(start).tz_localize(None).normalize()
    end = pd.Timestamp(end).tz_localize(None).normalize()
    calendar = pd.date_range(start, end, freq="D")
    docs = docs.copy()
    docs["news_date"] = pd.to_datetime(docs["news_date"]).dt.tz_localize(None).dt.normalize()
    # chỉ bài trong cửa sổ lịch
    docs = docs[(docs["news_date"] >= start) & (docs["news_date"] <= end)]
    sent = sent.copy()
    parts = [
        build_scope_panel(
            docs, sent, calendar, scope, models=models, abn_window=abn_window,
            noprice=noprice, noprice_min_n=noprice_min_n,
        )
        for scope in scopes
    ]
    panel = pd.concat(parts, ignore_index=True)
    # sắp xếp cột
    base = ["date", "scope"]
    feat_cols = []
    for m in models:
        feat_cols.extend([
            f"{m}_Sent_t", f"{m}_PosSent", f"{m}_NegSent",
            f"{m}_NewsVolume_t", f"{m}_MeanProb_t", f"{m}_AbnAttention_t",
        ])
    if noprice and "NewsVolume_noprice" in panel.columns:
        feat_cols.append("NewsVolume_noprice")
    cols = base + [c for c in feat_cols if c in panel.columns]
    return panel[cols].sort_values(["scope", "date"]).reset_index(drop=True)


def assert_price_desc_complete(docs: pd.DataFrame, narr_flags: pd.DataFrame) -> pd.DataFrame:
    """Mọi doc_id in_main_sample phải có price_desc không NaN. Trả docs đã gắn cờ."""
    flags = narr_flags[["doc_id", "price_desc"]].drop_duplicates("doc_id")
    merged = docs.merge(flags, on="doc_id", how="left")
    n_miss = int(merged["price_desc"].isna().sum())
    if n_miss:
        raise AssertionError(
            "Thiếu price_desc (NaN) cho %d/%d doc_id in_main_sample"
            % (n_miss, len(merged))
        )
    merged["price_desc"] = merged["price_desc"].astype(bool)
    return merged


def noprice_exclusion_report(docs: pd.DataFrame) -> dict:
    """Tỷ lệ bài price_desc theo năm + số ngày (all) sẽ NaN Sent nếu min_n=5."""
    d = docs.copy()
    d["news_date"] = pd.to_datetime(d["news_date"]).dt.tz_localize(None).dt.normalize()
    d["year"] = d["news_date"].dt.year
    by_year = (
        d.groupby("year")
        .agg(n=("doc_id", "size"), n_price=("price_desc", "sum"))
        .reset_index()
    )
    by_year["share_price_desc"] = by_year["n_price"] / by_year["n"]
    # days with noprice < 5 (scope=all, unique docs/day)
    uniq = d.drop_duplicates(["doc_id", "news_date"])
    daily = uniq.groupby("news_date").agg(
        n_all=("doc_id", "size"),
        n_noprice=("price_desc", lambda s: int((~s.astype(bool)).sum())),
    )
    n_nan_days = int((daily["n_noprice"] < 5).sum())
    return {
        "by_year": by_year.to_dict(orient="records"),
        "n_docs": int(len(uniq)),
        "n_price_desc": int(uniq["price_desc"].astype(bool).sum()),
        "n_noprice": int((~uniq["price_desc"].astype(bool)).sum()),
        "share_price_desc": float(uniq["price_desc"].astype(bool).mean()),
        "n_nan_sent_days_all_min5": n_nan_days,
        "n_calendar_days_with_news": int(len(daily)),
    }


def prepare_docs(
    text: pd.DataFrame,
    exclude_sources: list[str] | None = None,
) -> pd.DataFrame:
    """Lọc in_main_sample (+ exclude nguồn tuỳ chọn)."""
    docs = text.loc[text["in_main_sample"], ["doc_id", "news_date", "source", "currencies"]].copy()
    if exclude_sources:
        excl = {s.lower() for s in exclude_sources}
        docs = docs[~docs["source"].str.lower().isin(excl)]
    return docs.reset_index(drop=True)


def assert_docs_have_both_models(
    docs: pd.DataFrame,
    sent: pd.DataFrame,
    models: tuple[str, ...] = MODELS,
) -> None:
    """Mọi doc_id trong docs phải có điểm cho CẢ models; thiếu → raise kèm số lượng + 5 ví dụ."""
    need = set(models)
    have = (
        sent.loc[sent["model"].isin(models)]
        .groupby("doc_id")["model"]
        .agg(lambda s: set(s))
    )
    missing_ids: list[str] = []
    for doc_id in docs["doc_id"].unique():
        got = have.get(doc_id, set())
        if not need.issubset(got):
            missing_ids.append(str(doc_id))
    if missing_ids:
        examples = missing_ids[:5]
        raise ValueError(
            "Thiếu sentiment (cần cả %s) cho %d/%d doc_id in_main_sample. "
            "Ví dụ (tối đa 5): %s"
            % (sorted(need), len(missing_ids), docs["doc_id"].nunique(), examples)
        )


def panel_log(panel: pd.DataFrame, docs: pd.DataFrame, sent: pd.DataFrame, meta: dict) -> dict:
    log = dict(meta)
    log["rows_panel"] = len(panel)
    log["days_per_scope"] = panel.groupby("scope")["date"].nunique().to_dict()
    log["n_docs"] = int(docs["doc_id"].nunique())
    log["n_sent_rows"] = len(sent)
    vol = "vader_NewsVolume_t"
    if vol in panel.columns:
        log["zero_news_days"] = (
            panel.loc[panel[vol] == 0].groupby("scope").size().to_dict()
        )
        log["total_newsvolume_all"] = int(
            panel.loc[panel["scope"] == "all", vol].sum()
        )
    if "label" in sent.columns and "model" in sent.columns:
        log["label_dist"] = {
            m: g["label"].value_counts().to_dict()
            for m, g in sent.groupby("model")
        }
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily sentiment panel")
    parser.add_argument("--text-variant", default="title", choices=["title", "nlp_text"])
    parser.add_argument(
        "--panel", default="title",
        choices=["title", "fulltext", "nlp_text", "noprice"],
        help="title/fulltext; noprice = loại price_desc khỏi sentiment",
    )
    parser.add_argument("--exclude-sources", nargs="*", default=None,
                        help="Bỏ nguồn khỏi panel (vd decrypt) — robustness theo nguồn")
    parser.add_argument("--out-suffix", default="",
                        help="Hậu tố file out, vd _nodeecrypt")
    parser.add_argument("--sentiment", default=None, help="Override đường dẫn sentiment parquet")
    parser.add_argument("--output", default=None, help="Override đường dẫn panel out")
    parser.add_argument("--noprice-min-n", type=int, default=5)
    args = parser.parse_args()

    root = project_root()
    params, paths = load_config(root)
    random.seed(params.get("seed", 42))
    start = params["sample"]["start"]
    end = params["sample"]["t_end"]
    abn_window = int(params.get("abn_attention_window", 30))

    # đồng bộ --panel với text-variant
    if args.panel in ("fulltext", "nlp_text"):
        args.text_variant = "nlp_text"
    noprice = args.panel == "noprice"

    text = pd.read_parquet(
        abs_path(root, paths["text_clean_path"]),
        columns=["doc_id", "news_date", "source", "currencies", "in_main_sample"],
    )
    docs = prepare_docs(text, exclude_sources=args.exclude_sources)

    if args.sentiment:
        sent_path = abs_path(root, args.sentiment)
    elif args.text_variant == "nlp_text":
        sent_path = abs_path(root, paths["sentiment_doc_fulltext_path"])
    else:
        sent_path = abs_path(root, paths["sentiment_doc_path"])
    if not sent_path.exists():
        sys.exit("Không thấy sentiment: %s" % sent_path)

    sent = pd.read_parquet(sent_path)
    if "text_variant" in sent.columns:
        sent = sent[sent["text_variant"] == args.text_variant]
    sent = sent[sent["model"].isin(MODELS)]
    assert_docs_have_both_models(docs, sent, models=MODELS)

    excl_report = None
    if noprice:
        narr_path = abs_path(root, paths["narrative_doc_path"])
        narr = pd.read_parquet(narr_path, columns=["doc_id", "price_desc"])
        docs = assert_price_desc_complete(docs, narr)
        excl_report = noprice_exclusion_report(docs)
        print("price_desc exclusion by year:")
        for row in excl_report["by_year"]:
            print(
                "  %s: n=%d price_desc=%d share=%.4f"
                % (row["year"], row["n"], row["n_price"], row["share_price_desc"])
            )
        print(
            "  overall share_price_desc=%.4f | nan_sent_days(all,min5)=%d / %d days_with_news"
            % (
                excl_report["share_price_desc"],
                excl_report["n_nan_sent_days_all_min5"],
                excl_report["n_calendar_days_with_news"],
            )
        )

    panel = build_daily_panel(
        docs, sent, start=start, end=end, abn_window=abn_window,
        noprice=noprice, noprice_min_n=args.noprice_min_n,
    )

    default_title = abs_path(root, paths["daily_panel_path"])
    if args.exclude_sources is not None and not args.output and not args.out_suffix:
        sys.exit("--exclude-sources cần --output hoặc --out-suffix (không ghi đè daily_panel.parquet)")

    if args.output:
        out_path = abs_path(root, args.output)
    elif noprice:
        out_path = abs_path(root, paths.get(
            "daily_panel_noprice_path", "data/processed/daily_panel_noprice.parquet"
        ))
    elif args.text_variant == "nlp_text":
        out_path = abs_path(root, paths.get(
            "daily_panel_fulltext_path", "data/processed/daily_panel_fulltext.parquet"
        ))
    else:
        out_path = abs_path(root, paths["daily_panel_path"])
    if args.out_suffix:
        out_path = out_path.with_name(out_path.stem + args.out_suffix + out_path.suffix)
    if args.exclude_sources is not None and out_path.resolve() == default_title.resolve():
        sys.exit("exclude_sources: từ chối ghi đè daily_panel.parquet mặc định")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    # bất biến nhanh
    assert not any(c.startswith("cv_ref_") for c in panel.columns)
    for scope, g in panel.groupby("scope"):
        d = pd.to_datetime(g["date"]).sort_values()
        exp = pd.date_range(d.min(), d.max())
        assert len(d) == len(exp), scope
    if noprice:
        # tổng bài noprice + price_desc = tổng (scope=all)
        assert "NewsVolume_noprice" in panel.columns
        g = panel.loc[panel["scope"] == "all"].copy()
        # reconstruct price_desc volume from docs in window
        docs_w = docs[
            (docs["news_date"] >= pd.Timestamp(start))
            & (docs["news_date"] <= pd.Timestamp(end))
        ]
        uniq = docs_w.drop_duplicates(["doc_id", "news_date"])
        n_all = len(uniq)
        n_np = int((~uniq["price_desc"].astype(bool)).sum())
        n_pd = int(uniq["price_desc"].astype(bool).sum())
        assert n_np + n_pd == n_all
        assert int(g["vader_NewsVolume_t"].sum()) == n_all
        assert int(g["NewsVolume_noprice"].sum()) == n_np

    log = panel_log(panel, docs, sent, {
        "text_variant": args.text_variant,
        "panel": args.panel,
        "noprice": noprice,
        "start": start,
        "t_end": end,
        "exclude_sources": args.exclude_sources,
        "out_path": str(out_path),
        "abn_window": abn_window,
        "noprice_exclusion": excl_report,
    })
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Da luu %d dong -> %s" % (len(panel), out_path))
    print("  days/scope:", log["days_per_scope"])
    print("  zero_news_days:", log.get("zero_news_days"))
    print("  total NewsVolume(all):", log.get("total_newsvolume_all"))
    if noprice:
        nan_sent = int(
            panel.loc[
                (panel["scope"] == "all") & panel["finbert_Sent_t"].isna()
                & (panel["vader_NewsVolume_t"] > 0)
            ].shape[0]
        )
        print("  days Sent NaN (all, có tin nhưng noprice<min_n hoặc empty):", nan_sent)
    print("Log:", log_path)


if __name__ == "__main__":
    main()

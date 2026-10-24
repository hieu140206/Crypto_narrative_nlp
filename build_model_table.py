"""build_model_table.py — bảng mô hình Spec A (đồng thời) và Spec B (dự báo).

Không chạy hồi quy. Đọc paths/params từ config. Biến thể fulltext: --panel fulltext.
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

ASSETS = ("BTC", "ETH")
MODELS = ("vader", "finbert")
SENT_SUFFIXES = (
    "Sent_t", "PosSent", "NegSent", "MeanProb_t", "NewsVolume_t", "AbnAttention_t",
)
FRED_SERIES = ("VIXCLS", "DGS10", "DTWEXBGS")


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


def _naive_day(s: pd.Series | pd.DatetimeIndex) -> pd.Series:
    out = pd.to_datetime(s)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.dt.normalize()


def assign_split(dates: pd.Series, params: dict) -> pd.Series:
    train_a, train_b = (pd.Timestamp(x) for x in params["split"]["train"])
    val_a, val_b = (pd.Timestamp(x) for x in params["split"]["validation"])
    test_a, test_b = (pd.Timestamp(x) for x in params["split"]["test"])
    d = _naive_day(dates)

    def one(x: pd.Timestamp) -> str:
        if train_a <= x <= train_b:
            return "train"
        if val_a <= x <= val_b:
            return "valid"
        if test_a <= x <= test_b:
            return "test"
        return "out"

    return d.map(one)


def prepare_fred_with_d5(fred: pd.DataFrame) -> pd.DataFrame:
    """Tính thay đổi 5 quan sát trên chuỗi gốc (theo date), giữ usable_from_utc."""
    parts = []
    for series, g in fred.groupby("series", sort=False):
        g = g.sort_values("date").copy()
        g["value_d5"] = g["value"].astype(float) - g["value"].astype(float).shift(5)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["usable_from_utc"] = pd.to_datetime(out["usable_from_utc"], utc=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out


def join_fred_asof(
    calendar: pd.DatetimeIndex,
    fred_prep: pd.DataFrame,
    asof_offset_days: int,
) -> pd.DataFrame:
    """As-of: usable_from_utc <= (date + asof_offset_days) 00:00 UTC."""
    base = pd.DataFrame({"date": pd.to_datetime(calendar).normalize()})
    # mốc as-of UTC
    asof = (base["date"] + pd.Timedelta(days=asof_offset_days)).dt.tz_localize("UTC")
    keys = pd.DataFrame({"_asof": asof, "_i": np.arange(len(base))}).sort_values("_asof")

    for series in FRED_SERIES:
        sub = fred_prep.loc[fred_prep["series"] == series, ["usable_from_utc", "value", "value_d5"]]
        sub = sub.sort_values("usable_from_utc")
        merged = pd.merge_asof(
            keys, sub,
            left_on="_asof", right_on="usable_from_utc",
            direction="backward",
        )
        merged = merged.sort_values("_i")
        base[f"fred_{series}"] = merged["value"].values
        base[f"fred_{series}_d5"] = merged["value_d5"].values
    return base


def _rename_sent(scope_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """scope panel → prefix_model_suffix (vd all_vader_Sent_t)."""
    out = scope_df[["date"]].copy()
    for m in MODELS:
        for suf in SENT_SUFFIXES:
            src = f"{m}_{suf}"
            if src in scope_df.columns:
                out[f"{prefix}_{m}_{suf}"] = scope_df[src].values
    return out


def apply_own_low_n(own: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """Nếu NewsVolume (bất kỳ model) < n_min → NaN sentiment (giữ volume) + own_low_n."""
    out = own.copy()
    vol_cols = [c for c in out.columns if c.endswith("_NewsVolume_t")]
    if not vol_cols:
        out["own_low_n"] = False
        return out
    vol = out[vol_cols[0]].astype(float)
    low = vol.isna() | (vol < n_min)
    out["own_low_n"] = low
    # NaN các cột cảm xúc / meanprob / abn (không xoá volume)
    for c in out.columns:
        if c in ("date", "own_low_n") or c.endswith("_NewsVolume_t"):
            continue
        if c.startswith("own_"):
            out.loc[low, c] = np.nan
    return out


def narrative_all_features(narr: pd.DataFrame) -> pd.DataFrame:
    """scope=all: Att_*, LogRatio_*, AbnAtt_*, Att_price_desc."""
    g = narr.loc[narr["scope"] == "all"].copy()
    g["date"] = _naive_day(g["date"])
    keep = ["date"]
    for c in g.columns:
        if c.startswith(("Att_", "LogRatio_", "AbnAtt_")):
            keep.append(c)
    return g[keep].drop_duplicates("date")


def har_from_log_rv(
    log_rv: pd.Series,
    complete_day: pd.Series,
    *,
    include_t: bool,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """HAR từ log_rv; trung bình chỉ trên ngày complete_day.

    Ngày incomplete: coi log_rv = NaN (bỏ khỏi trung bình, không điền khuyết).
    Cửa sổ lịch cố định; min_periods = số ngày hợp lệ tối thiểu trong cửa sổ.

    - include_t=False (Spec A, kiểm soát trễ):
      har_d = log_rv_{t-1};
      har_w = mean(t-5..t-1), rolling(5, min_periods=4);
      har_m = mean(t-22..t-1), rolling(22, min_periods=18).
    - include_t=True (Spec B, hết ngày t):
      har_d = log_rv_t;
      har_w = mean(t-4..t), rolling(5, min_periods=4);
      har_m = mean(t-21..t), rolling(22, min_periods=18).
    """
    complete = complete_day.fillna(False).astype(bool)
    lr = log_rv.astype(float).where(complete)  # incomplete → NaN
    base = lr if include_t else lr.shift(1)
    har_d = base  # Spec A: đã shift; Spec B: log_rv_t (NaN nếu incomplete)
    har_w = base.rolling(5, min_periods=4).mean()
    har_m = base.rolling(22, min_periods=18).mean()
    return har_d, har_w, har_m


def market_controls_A(mkt: pd.DataFrame) -> pd.DataFrame:
    """Spec A: ret_lag1/2, HAR trễ, logvol_lag1; Y cùng ngày: ret, log_rv, abvol."""
    parts = []
    for asset, g in mkt.groupby("asset"):
        g = g.sort_values("date").copy()
        g["date"] = _naive_day(g["date"])
        g["ret_lag1"] = g["ret"].shift(1)
        g["ret_lag2"] = g["ret"].shift(2)
        g["har_d"], g["har_w"], g["har_m"] = har_from_log_rv(
            g["log_rv"], g["complete_day"], include_t=False
        )
        g["logvol_lag1"] = np.log1p(g["volume"].astype(float)).shift(1)
        # Y_t
        g["y_ret"] = g["ret"]
        g["y_log_rv"] = g["log_rv"]
        g["y_abvol"] = g["abvol"]
        g["asset"] = asset
        parts.append(g)
    cols = [
        "date", "asset", "complete_day",
        "y_ret", "y_log_rv", "y_abvol",
        "ret_lag1", "ret_lag2", "har_d", "har_w", "har_m", "logvol_lag1",
    ]
    return pd.concat(parts, ignore_index=True)[cols]


def market_features_B(mkt: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Spec B: X thị trường hết ngày t + target cửa sổ t+1..t+h."""
    parts = []
    for asset, g in mkt.groupby("asset"):
        g = g.sort_values("date").copy()
        g["date"] = _naive_day(g["date"])
        g["ret_t"] = g["ret"]
        g["log_rv_t"] = g["log_rv"]
        g["rv_t"] = g["rv"]
        g["logvol_t"] = np.log1p(g["volume"].astype(float))
        g["har_d"], g["har_w"], g["har_m"] = har_from_log_rv(
            g["log_rv"], g["complete_day"], include_t=True
        )

        for h in horizons:
            # forward window t+1..t+h
            rets = []
            rvs = []
            completes = []
            for k in range(1, h + 1):
                rets.append(g["ret"].shift(-k))
                rvs.append(g["rv"].shift(-k))
                completes.append(g["complete_day"].shift(-k))
            ret_sum = sum(rets)
            # mean rv; NaN nếu bất kỳ ngày incomplete hoặc thiếu
            rv_stack = np.vstack([r.values.astype(float) for r in rvs])
            # NaN/None trong complete_day shift → coi incomplete
            comp_stack = np.vstack([(c == True).to_numpy() for c in completes])
            any_incomplete = ~comp_stack.all(axis=0)
            any_nan_rv = np.isnan(rv_stack).any(axis=0)
            any_nan_ret = np.isnan(np.vstack([r.values.astype(float) for r in rets])).any(axis=0)
            bad = any_incomplete | any_nan_rv | any_nan_ret
            y_rv = np.full(rv_stack.shape[1], np.nan, dtype=float)
            good = ~bad
            if good.any():
                y_rv[good] = rv_stack[:, good].mean(axis=0)
            y_ret = ret_sum.values.astype(float)
            y_ret = np.where(bad, np.nan, y_ret)
            g[f"y_ret_{h}"] = y_ret
            g[f"y_rv_{h}"] = y_rv
            with np.errstate(divide="ignore", invalid="ignore"):
                g[f"y_logrv_{h}"] = np.where(np.isnan(y_rv) | (y_rv <= 0), np.nan, np.log(y_rv))
            g[f"target_end_date_{h}"] = g["date"] + pd.Timedelta(days=h)

        g["asset"] = asset
        parts.append(g)

    base_cols = [
        "date", "asset", "complete_day",
        "ret_t", "log_rv_t", "rv_t", "logvol_t", "har_d", "har_w", "har_m",
    ]
    extra = []
    for h in horizons:
        extra += [f"y_ret_{h}", f"y_rv_{h}", f"y_logrv_{h}", f"target_end_date_{h}"]
    return pd.concat(parts, ignore_index=True)[base_cols + extra]


def embargo_flags(
    dates: pd.Series,
    splits: pd.Series,
    params: dict,
    horizons: list[int],
) -> pd.DataFrame:
    """embargo_drop_h: gần ranh giới (embargo_days) hoặc target_end sang tập sau."""
    embargo = int(params["split"]["embargo_days"])
    val_start = pd.Timestamp(params["split"]["validation"][0])
    test_start = pd.Timestamp(params["split"]["test"][0])
    d = _naive_day(dates)
    near = (
        ((d >= val_start - pd.Timedelta(days=embargo)) & (d < val_start))
        | ((d >= test_start - pd.Timedelta(days=embargo)) & (d < test_start))
    )
    out = pd.DataFrame(index=dates.index)
    for h in horizons:
        tend = d + pd.Timedelta(days=h)
        crosses = (
            ((splits == "train") & (tend >= val_start))
            | ((splits == "valid") & (tend >= test_start))
        )
        out[f"embargo_drop_{h}"] = (near | crosses).astype(bool)
    return out


def build_skeleton(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    assets: tuple[str, ...] = ASSETS,
) -> pd.DataFrame:
    calendar = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    rows = [{"date": d, "asset": a} for a in assets for d in calendar]
    return pd.DataFrame(rows)


def attach_sentiment(
    base: pd.DataFrame,
    panel: pd.DataFrame,
    own_n_min: int,
) -> pd.DataFrame:
    panel = panel.copy()
    panel["date"] = _naive_day(panel["date"])
    all_s = _rename_sent(panel.loc[panel["scope"] == "all"], "all")
    own_parts = []
    for asset in ASSETS:
        o = _rename_sent(panel.loc[panel["scope"] == asset], "own")
        o = apply_own_low_n(o, own_n_min)
        o["asset"] = asset
        own_parts.append(o)
    own = pd.concat(own_parts, ignore_index=True)
    out = base.merge(all_s, on="date", how="left")
    out = out.merge(own, on=["date", "asset"], how="left")
    return out


def build_model_table_A(
    panel: pd.DataFrame,
    narr: pd.DataFrame,
    market: pd.DataFrame,
    fred: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    start, end = params["sample"]["start"], params["sample"]["t_end"]
    own_n_min = int(params.get("own_n_min", 5))
    base = build_skeleton(start, end)
    base = attach_sentiment(base, panel, own_n_min)
    base = base.merge(narrative_all_features(narr), on="date", how="left")

    mkt = market.loc[market["asset"].isin(ASSETS)].copy()
    ctrl = market_controls_A(mkt)
    base = base.merge(ctrl, on=["date", "asset"], how="left")

    cal = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    fred_prep = prepare_fred_with_d5(fred)
    fred_a = join_fred_asof(cal, fred_prep, asof_offset_days=0)
    base = base.merge(fred_a, on="date", how="left")

    base["split"] = assign_split(base["date"], params)
    base = base.loc[base["split"] != "out"].copy()
    return base.sort_values(["asset", "date"]).reset_index(drop=True)


def build_model_table_B(
    panel: pd.DataFrame,
    narr: pd.DataFrame,
    market: pd.DataFrame,
    fred: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    start, end = params["sample"]["start"], params["sample"]["t_end"]
    own_n_min = int(params.get("own_n_min", 5))
    horizons = [int(h) for h in params["horizons"]]
    base = build_skeleton(start, end)
    base = attach_sentiment(base, panel, own_n_min)
    base = base.merge(narrative_all_features(narr), on="date", how="left")

    mkt = market.loc[market["asset"].isin(ASSETS)].copy()
    feat = market_features_B(mkt, horizons)
    base = base.merge(feat, on=["date", "asset"], how="left")

    cal = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    fred_prep = prepare_fred_with_d5(fred)
    # điểm gốc dự báo = hết ngày t → as-of (t+1) 00:00 UTC
    fred_b = join_fred_asof(cal, fred_prep, asof_offset_days=1)
    base = base.merge(fred_b, on="date", how="left")

    base["split"] = assign_split(base["date"], params)
    flags = embargo_flags(base["date"], base["split"], params, horizons)
    base = pd.concat([base, flags], axis=1)
    base = base.loc[base["split"] != "out"].copy()
    return base.sort_values(["asset", "date"]).reset_index(drop=True)


def _nan_report(df: pd.DataFrame, split_col: str = "split") -> dict:
    """Số NaN theo split — chỉ cột có ≥1 NaN. Không in split=test."""
    out = {}
    for sp, g in df.groupby(split_col):
        if sp == "test":
            continue
        nan_cols = {}
        for c in g.columns:
            if c == split_col:
                continue
            n = int(g[c].isna().sum())
            if n > 0:
                nan_cols[c] = n
        out[str(sp)] = nan_cols
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model tables Spec A/B")
    parser.add_argument(
        "--panel", default="title",
        choices=["title", "fulltext", "nlp_text", "noprice"],
        help="title=bản chính; fulltext/nlp_text=độ bền; noprice=loại price_desc",
    )
    args = parser.parse_args()

    root = project_root()
    params, paths = load_config(root)
    seed = int(params.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    is_full = args.panel in ("fulltext", "nlp_text")
    is_noprice = args.panel == "noprice"
    if is_noprice:
        panel_key = "daily_panel_noprice_path"
        out_a_key = "model_table_A_noprice_path"
        out_b_key = "model_table_B_noprice_path"
    elif is_full:
        panel_key = "daily_panel_fulltext_path"
        out_a_key = "model_table_A_fulltext_path"
        out_b_key = "model_table_B_fulltext_path"
    else:
        panel_key = "daily_panel_path"
        out_a_key = "model_table_A_path"
        out_b_key = "model_table_B_path"

    panel = pd.read_parquet(abs_path(root, paths[panel_key]))
    narr = pd.read_parquet(abs_path(root, paths["daily_narratives_path"]))
    market = pd.read_parquet(abs_path(root, paths["market_daily_path"]))
    fred = pd.read_parquet(abs_path(root, paths["fred_path"]))

    table_a = build_model_table_A(panel, narr, market, fred, params)
    table_b = build_model_table_B(panel, narr, market, fred, params)

    out_a = abs_path(root, paths[out_a_key])
    out_b = abs_path(root, paths[out_b_key])
    out_a.parent.mkdir(parents=True, exist_ok=True)
    table_a.to_parquet(out_a, index=False)
    table_b.to_parquet(out_b, index=False)

    # báo cáo (không đụng test)
    def split_ranges(df: pd.DataFrame) -> dict:
        r = {}
        for sp, g in df.groupby("split"):
            if sp == "test":
                r[sp] = {"n": len(g), "note": "locked — không thống kê"}
                continue
            d = _naive_day(g["date"])
            r[sp] = {
                "n": len(g),
                "date_min": str(d.min().date()),
                "date_max": str(d.max().date()),
            }
        return r

    abn_col = "all_vader_AbnAttention_t"
    first_abn = None
    if abn_col in table_a.columns:
        sub = table_a.loc[table_a["asset"] == "BTC", ["date", abn_col]].dropna()
        if len(sub):
            first_abn = str(_naive_day(sub["date"]).min().date())

    horizons = [int(h) for h in params["horizons"]]
    embargo_counts = {}
    for sp in ("train", "valid"):
        g = table_b[table_b["split"] == sp]
        embargo_counts[sp] = {
            str(h): int(g[f"embargo_drop_{h}"].sum()) for h in horizons
        }

    log = {
        "panel": args.panel,
        "shape_A": list(table_a.shape),
        "shape_B": list(table_b.shape),
        "split_ranges_A": split_ranges(table_a),
        "split_ranges_B": split_ranges(table_b),
        "nan_by_split_A": _nan_report(table_a),
        "nan_by_split_B": _nan_report(table_b),
        "first_all_AbnAttention": first_abn,
        "embargo_drop_counts_B": embargo_counts,
        "out_A": str(out_a),
        "out_B": str(out_b),
    }
    log_path = out_a.with_name(out_a.stem + "_build.log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("A", table_a.shape, "->", out_a)
    print("B", table_b.shape, "->", out_b)
    print("split A:", log["split_ranges_A"])
    print("split B:", log["split_ranges_B"])
    print("first all_AbnAttention:", first_abn)
    print("embargo_drop B:", embargo_counts)
    print("Log:", log_path)


if __name__ == "__main__":
    main()

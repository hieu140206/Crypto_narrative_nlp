"""build_market_daily.py - Binance 1h -> market_daily.parquet + FRED -> fred.parquet (TV2).

Logic lấy từ market_data.py (build_daily / usable_from), đọc parquet sẵn trong
data/raw/binance (không tải lại). Binance 2026 để sau (holdout).

Công thức:
  ret     = log(close_ngày / open_ngày)
  rv      = tổng bình phương 24 log-return giờ; thiếu giờ -> NaN, complete_day=False
  log_rv  = log(rv) khi rv > 0
  volume  = tổng volume giờ trong ngày
  abvol   = log(volume_t) - mean(log volume_{t-1..t-30}); 30 ngày đầu = NaN

FRED (VIXCLS, DGS10, DTWEXBGS): cột usable_from_utc; join theo usable_from_utc
(không theo date). VIX: date+1; DGS10: date+2 (H.15 công bố chiều ngày làm việc sau);
DTWEXBGS: thứ Ba sau Friday của tuần.

FRED ở đây là current-vintage (fredgraph), KHÔNG phải point-in-time ALFRED —
giá trị có thể bị revise sau phát hành.
"""
from __future__ import annotations

import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)

SYMBOL_TO_ASSET = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
DEFAULT_FRED = {"VIXCLS": "vix", "DTWEXBGS": "usd_broad", "DGS10": "ust10y"}
# VIX +1; DGS10 +2 (tránh look-ahead nhẹ khi H.15 ra ~20-21h UTC ngày làm việc sau)
DAILY_LAG_DAYS = {"VIXCLS": 1, "DGS10": 2}
WEEKLY_H10_SERIES = {"DTWEXBGS"}


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


def usable_from(series: str, dates: pd.Series) -> pd.Series:
    """usable_from_utc (tz-aware UTC) theo rule phát hành FRED."""
    dates = pd.to_datetime(pd.Series(dates))
    if series in DAILY_LAG_DAYS:
        out = dates + pd.Timedelta(days=DAILY_LAG_DAYS[series])
    elif series in WEEKLY_H10_SERIES:
        # Friday của tuần chứa date, rồi +4 ngày = Tuesday sau đó
        friday = dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit="D")
        out = friday + pd.Timedelta(days=4)
    else:
        out = dates + pd.Timedelta(days=7)
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize("UTC")
    else:
        out = out.dt.tz_convert("UTC")
    return out.reset_index(drop=True)

def load_hourly(binance_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(binance_dir.glob("*USDT_1h.parquet")):
        symbol = path.name.split("_")[0]
        if symbol not in SYMBOL_TO_ASSET:
            continue
        frame = pd.read_parquet(path)
        frame = frame.copy()
        frame["open_time_utc"] = pd.to_datetime(frame["open_time"], utc=True)
        frame["symbol"] = symbol
        frame["asset"] = SYMBOL_TO_ASSET[symbol]
        frames.append(frame)
    if not frames:
        sys.exit("Không tìm thấy *USDT_1h.parquet trong %s" % binance_dir)
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(["symbol", "open_time_utc"]).sort_values(
        ["symbol", "open_time_utc"]
    )
    return data.reset_index(drop=True)


def build_daily_for_asset(g: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Gộp giờ -> ngày; reindex đủ calendar; rv NaN nếu thiếu giờ."""
    g = g.set_index("open_time_utc").sort_index()
    g["log_ret_1h"] = np.log(g["close"] / g["open"])
    day = g.index.floor("D")
    daily = pd.DataFrame({
        "bars": g.groupby(day)["close"].size(),
        "open": g.groupby(day)["open"].first(),
        "close": g.groupby(day)["close"].last(),
        "volume": g.groupby(day)["volume"].sum(),
        "rv_sumsq": g.groupby(day)["log_ret_1h"].apply(lambda s: float((s ** 2).sum())),
    })
    # calendar tz-aware UTC midnight
    daily = daily.reindex(calendar)
    daily["bars"] = daily["bars"].fillna(0).astype(int)
    daily["complete_day"] = daily["bars"].eq(24)
    daily["rv"] = daily["rv_sumsq"].where(daily["complete_day"], np.nan)
    daily["ret"] = np.log(daily["close"] / daily["open"])
    daily["log_rv"] = np.log(daily["rv"].where(daily["rv"] > 0))
    logvol = np.log(daily["volume"].where(daily["volume"] > 0))
    past_mean = logvol.shift(1).rolling(30, min_periods=30).mean()
    daily["abvol"] = logvol - past_mean
    daily = daily.drop(columns=["rv_sumsq", "open", "close", "bars"])
    daily.index.name = "date"
    return daily.reset_index()


def build_market_daily(hourly: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    calendar = pd.date_range(start, end, freq="D", tz="UTC")
    rows = []
    for asset, g in hourly.groupby("asset"):
        # cắt giờ trong cửa sổ (ngày end gồm đủ 24h)
        t0 = pd.Timestamp(start, tz="UTC")
        t1 = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        gg = g[(g["open_time_utc"] >= t0) & (g["open_time_utc"] < t1)]
        daily = build_daily_for_asset(gg, calendar)
        daily.insert(1, "asset", asset)
        # date naive UTC (khớp news_date)
        daily["date"] = daily["date"].dt.tz_localize(None)
        rows.append(daily)
    out = pd.concat(rows, ignore_index=True)
    cols = ["date", "asset", "ret", "rv", "log_rv", "volume", "abvol", "complete_day"]
    return out[cols].sort_values(["asset", "date"]).reset_index(drop=True)


def http_get(url: str, retries: int = 6, timeout: int = 90) -> str:
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (research-topic10)"})
    last_err = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200 and response.text.strip():
                return response.text
            last_err = "HTTP %s" % response.status_code
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(2 ** min(attempt, 5))
    raise RuntimeError("FRED download failed: %s" % last_err)


def build_fred(series_map: dict[str, str] | None = None) -> pd.DataFrame:
    series_map = series_map or DEFAULT_FRED
    frames = []
    for series, name in series_map.items():
        print("  FRED", series, "...")
        text = http_get(FRED_URL.format(series=series))
        frame = pd.read_csv(io.StringIO(text))
        date_col, value_col = frame.columns[0], frame.columns[1]
        frame = frame.rename(columns={date_col: "date", value_col: "value"})
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame["series"], frame["name"] = series, name
        frame["date"] = pd.to_datetime(frame["date"])
        frame["usable_from_utc"] = usable_from(series, frame["date"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["value"]).sort_values(["series", "date"]).reset_index(drop=True)
    return data[["date", "series", "name", "value", "usable_from_utc"]]

def main() -> None:
    root = project_root()
    params, paths = load_config(root)
    random.seed(params.get("seed", 42))
    start = params.get("sample", {}).get("robustness_start", "2020-01-01")
    end = params.get("sample", {}).get("market_end", "2025-12-31")

    binance_dir = abs_path(root, paths["binance_dir"])
    market_out = abs_path(root, paths["market_daily_path"])
    fred_out = abs_path(root, paths.get("fred_path", "data/interim/fred.parquet"))

    print("Doc Binance:", binance_dir)
    hourly = load_hourly(binance_dir)
    market = build_market_daily(hourly, start, end)
    market_out.parent.mkdir(parents=True, exist_ok=True)
    market.to_parquet(market_out, index=False)

    # Manual RV check seed: 3 ngày complete cố định (ghi vào log)
    sample_days = [pd.Timestamp("2021-06-15"), pd.Timestamp("2022-01-10"), pd.Timestamp("2024-03-01")]
    manual = {}
    for asset, g in hourly.groupby("asset"):
        g = g.set_index("open_time_utc").sort_index()
        g["log_ret_1h"] = np.log(g["close"] / g["open"])
        for d in sample_days:
            d0 = d.tz_localize("UTC")
            day_bars = g.loc[d0:d0 + pd.Timedelta(hours=23)]
            if len(day_bars) != 24:
                continue
            rv_m = float((day_bars["log_ret_1h"] ** 2).sum())
            row = market[(market["asset"] == asset) & (market["date"] == d)]
            rv_t = float(row["rv"].iloc[0]) if len(row) else float("nan")
            manual["%s_%s" % (asset, d.date())] = {
                "rv_manual": rv_m, "rv_table": rv_t, "match": abs(rv_m - rv_t) < 1e-12
            }

    print("Tai FRED...")
    fred = build_fred()
    fred_out.parent.mkdir(parents=True, exist_ok=True)
    fred.to_parquet(fred_out, index=False)

    log = {
        "start": start,
        "end": end,
        "rows_market": len(market),
        "days_per_asset": market.groupby("asset")["date"].nunique().to_dict(),
        "incomplete_per_asset": market.groupby("asset")["complete_day"].apply(
            lambda s: int((~s).sum())
        ).to_dict(),
        "incomplete_total": int((~market["complete_day"]).sum()),
        "missing_hours_note": "Binance thiếu 31 giờ trong cửa sổ; ngày thiếu giờ: rv=NaN",
        "rv_manual_check": manual,
        "fred_rows": len(fred),
        "fred_by_series": fred.groupby("series").size().to_dict(),
        "market_path": str(market_out),
        "fred_path": str(fred_out),
    }
    log_path = market_out.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("Da luu %d dong -> %s" % (len(market), market_out))
    print("  days/asset:", log["days_per_asset"])
    print("  incomplete/asset:", log["incomplete_per_asset"], "total", log["incomplete_total"])
    print("Da luu FRED %d dong -> %s" % (len(fred), fred_out))
    print("Log:", log_path)
    assert all(v == 2192 for v in log["days_per_asset"].values()), log["days_per_asset"]
    assert 25 <= log["incomplete_total"] <= 31, log["incomplete_total"]
    assert all(x["match"] for x in manual.values()), manual


if __name__ == "__main__":
    main()

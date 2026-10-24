"""Market data for topic 10: Binance spot klines (1h) and FRED macro series.

Commands
  binance-download  fetch monthly kline zips + CHECKSUM, verify sha256, skip existing
  binance-build     parse zips -> hourly parquet (UTC) -> daily feature table
  fred              fetch FRED series (no API key; public fredgraph CSV)

Point-in-time rules baked in
  * Every timestamp is normalised to UTC. Binance spot files switch from
    milliseconds to microseconds (and some files carry a header row); both are
    detected, never assumed.
  * Daily features for day t use only bars with open time in [t 00:00, t+1 00:00) UTC.
  * AbVol_t = log(Vol_t) - mean(log Vol_{t-1..t-30})  (past-only baseline).
  * FRED values are dated by observation day but are only knowable later.
    `usable_from_utc` encodes a conservative release rule per series (daily
    H.15 / VIX: observation date + 1 day; weekly H.10 dollar index: the Tuesday
    after the week's Friday; unknown series: +7 days). Join on `usable_from_utc`,
    never on `date`. FRED serves current-vintage values; revised series are not
    strictly point-in-time (use ALFRED if that matters).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BINANCE_BASE = "https://data.binance.vision"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
# DXY itself (ICE) is not on FRED; the Fed's broad trade-weighted dollar index is.
DEFAULT_FRED = {"VIXCLS": "vix", "DTWEXBGS": "usd_broad", "DGS10": "ust10y"}
# Release rules:
#   VIXCLS : ~1 day lag -> +1 day
#   DGS10  : H.15 out ~20-21h UTC next business day -> +2 days (tránh look-ahead nhẹ)
#   DTWEXBGS: weekly H.10 through Friday -> Tuesday after that Friday
# FRED fredgraph = current vintage, NOT ALFRED point-in-time.
DAILY_LAG_DAYS = {"VIXCLS": 1, "DGS10": 2}
WEEKLY_H10_SERIES = {"DTWEXBGS"}


def usable_from(series: str, dates: pd.Series) -> pd.Series:
    if series in DAILY_LAG_DAYS:
        out = dates + pd.Timedelta(days=DAILY_LAG_DAYS[series])
    elif series in WEEKLY_H10_SERIES:
        friday = dates + pd.to_timedelta((4 - dates.dt.weekday) % 7, unit="D")   # Friday of that week
        out = friday + pd.Timedelta(days=4)                                       # following Tuesday
    else:
        print(f"WARNING: no release rule for {series}; assuming +7 days", file=sys.stderr)
        out = dates + pd.Timedelta(days=7)
    return out.dt.tz_localize("UTC")


# --------------------------------------------------------------- download
def month_range(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def http_get(session: requests.Session, url: str, retries: int = 4, timeout: int = 60):
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code in (200, 404):
                return response
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    return None


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command_binance_download(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    log_rows = []
    for symbol in args.symbols:
        for month in month_range(args.start, args.end):
            name = f"{symbol}-{args.interval}-{month}.zip"
            target = out / name
            url = f"{args.base_url}/data/spot/monthly/klines/{symbol}/{args.interval}/{name}"
            if target.exists():
                log_rows.append((name, "exists", ""))
                continue
            data = http_get(session, url)
            if data is None or data.status_code != 200:
                log_rows.append((name, "missing", "" if data is None else str(data.status_code)))
                continue
            checksum = http_get(session, url + ".CHECKSUM")
            expected = checksum.text.split()[0] if checksum is not None and checksum.status_code == 200 else ""
            actual = sha256_of(data.content)
            if expected and expected != actual:
                log_rows.append((name, "checksum_mismatch", f"{expected} != {actual}"))
                continue
            target.write_bytes(data.content)
            log_rows.append((name, "ok" if expected else "ok_no_checksum", actual))
    log = pd.DataFrame(log_rows, columns=["file", "status", "detail"])
    log.to_csv(out / "download_log.csv", index=False)
    print(log["status"].value_counts().to_string())
    bad = log[~log["status"].isin(["ok", "exists"])]
    if len(bad):
        print("PROBLEMS:\n" + bad.to_string(index=False))
        raise SystemExit(2)


# ------------------------------------------------------------------ parse
def read_kline_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected one CSV, found {names}")
        raw = archive.read(names[0])
    first = raw[:200].decode("utf-8", errors="ignore").split("\n", 1)[0]
    has_header = not first.split(",")[0].strip().isdigit()
    frame = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    frame = frame.iloc[:, :12]
    frame.columns = KLINE_COLUMNS
    return frame


def to_utc(open_time: pd.Series) -> pd.Series:
    """Detect ms vs us per file from magnitude (>= 1e14 means microseconds)."""
    values = pd.to_numeric(open_time, errors="raise").astype("int64")
    unit = np.where(values >= 10**14, "us", "ms")
    if len(set(unit)) != 1:
        raise ValueError("mixed ms/us timestamps inside one file")
    stamps = pd.to_datetime(values, unit=str(unit[0]), utc=True)
    if stamps.min() < pd.Timestamp("2017-01-01", tz="UTC") or stamps.max() > pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2):
        raise ValueError(f"timestamps outside 2017..now after unit detection: {stamps.min()} .. {stamps.max()}")
    return stamps


def build_hourly(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_dir.glob("*.zip")):
        symbol = path.name.split("-")[0]
        frame = read_kline_zip(path)
        frame["open_time_utc"] = to_utc(frame["open_time"])
        frame["symbol"] = symbol
        frame["source_file"] = path.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No kline zips in {raw_dir}")
    data = pd.concat(frames, ignore_index=True)
    for column in ["open", "high", "low", "close", "volume", "quote_volume", "trades"]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    data = data.drop_duplicates(["symbol", "open_time_utc"]).sort_values(["symbol", "open_time_utc"])
    return data[["symbol", "open_time_utc", "open", "high", "low", "close", "volume",
                 "quote_volume", "trades", "source_file"]].reset_index(drop=True)


def build_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for symbol, g in hourly.groupby("symbol"):
        g = g.set_index("open_time_utc").sort_index()
        g["log_ret_1h"] = np.log(g["close"] / g["open"])   # own-bar return, no cross-day dependence
        day = g.index.floor("D")
        daily = pd.DataFrame({
            "bars": g.groupby(day)["close"].size(),
            "open": g.groupby(day)["open"].first(),
            "close": g.groupby(day)["close"].last(),
            "volume": g.groupby(day)["volume"].sum(),
            "quote_volume": g.groupby(day)["quote_volume"].sum(),
            "trades": g.groupby(day)["trades"].sum(),
            # realised variance: sum of the 24 squared hourly log returns of the day (needs all 24 bars)
            "rv": g.groupby(day)["log_ret_1h"].apply(lambda s: float((s ** 2).sum()) if len(s) == 24 else np.nan),
        })
        daily.index.name = "date_utc"
        full = daily["bars"].eq(24)
        daily["complete_day"] = full
        daily["ret"] = np.log(daily["close"] / daily["open"])
        daily["log_rv"] = np.log(daily["rv"].where(daily["rv"] > 0))
        logvol = np.log(daily["volume"].where(daily["volume"] > 0))
        past_mean = logvol.shift(1).rolling(30, min_periods=30).mean()
        daily["abvol"] = logvol - past_mean
        daily.insert(0, "symbol", symbol)
        rows.append(daily.reset_index())
    return pd.concat(rows, ignore_index=True)


def command_binance_build(args: argparse.Namespace) -> None:
    hourly = build_hourly(Path(args.raw_dir))
    daily = build_daily(hourly)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out / "klines_1h_utc.parquet", index=False)
    daily.to_parquet(out / "daily_features.parquet", index=False)
    gaps = daily[~daily["complete_day"]]
    gaps.to_csv(out / "incomplete_days.csv", index=False)
    print(f"hourly {len(hourly):,} rows; daily {len(daily):,} rows; incomplete days {len(gaps)}")
    print(daily.groupby("symbol")["date_utc"].agg(["min", "max"]).to_string())


# ------------------------------------------------------------------- FRED
def command_fred(args: argparse.Namespace) -> None:
    session = requests.Session()
    out = Path(args.output)
    series_map = dict(item.split("=") for item in args.series) if args.series else DEFAULT_FRED
    frames = []
    for series, name in series_map.items():
        response = http_get(session, args.base_url.format(series=series))
        if response is None or response.status_code != 200:
            raise RuntimeError(f"FRED {series}: no response or HTTP error")
        frame = pd.read_csv(io.StringIO(response.text))
        date_col, value_col = frame.columns[0], frame.columns[1]
        frame = frame.rename(columns={date_col: "date", value_col: "value"})
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")   # "." = holiday gap
        frame["series"], frame["name"] = series, name
        frame["date"] = pd.to_datetime(frame["date"])
        frame["usable_from_utc"] = usable_from(series, frame["date"])
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["value"]).sort_values(["series", "date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out, index=False)
    print(data.groupby("series")["date"].agg(["min", "max", "count"]).to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Market data for topic 10")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("binance-download")
    d.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    d.add_argument("--interval", default="1h")
    d.add_argument("--start", default="2020-12")      # one month of warm-up before 2021-01-01
    d.add_argument("--end", default="2025-08")
    d.add_argument("--output-dir", required=True)
    d.add_argument("--base-url", default=BINANCE_BASE)
    d.set_defaults(func=command_binance_download)
    b = sub.add_parser("binance-build")
    b.add_argument("--raw-dir", required=True)
    b.add_argument("--output-dir", required=True)
    b.set_defaults(func=command_binance_build)
    f = sub.add_parser("fred")
    f.add_argument("--output", required=True)
    f.add_argument("--series", nargs="*", help="SERIES=name pairs; default VIXCLS DTWEXBGS DGS10")
    f.add_argument("--base-url", default=FRED_URL)
    f.set_defaults(func=command_fred)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

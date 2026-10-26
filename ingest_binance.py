"""TV1 - Nạp Binance Vision klines 1h (BTCUSDT, ETHUSDT), xử lý ms/us, ghép thành 1 file/asset.

Chạy:  python src/data/ingest_binance.py
Output: data/raw/binance/BTCUSDT_1h.parquet, ETHUSDT_1h.parquet
"""
import zipfile

import pandas as pd

from common import ROOT, load_paths, p, save_df, set_seed, write_source_log

set_seed(42)

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "n_trades", "taker_base_vol", "taker_quote_vol", "ignore"]
SYMBOLS = ["BTCUSDT", "ETHUSDT"]


def to_datetime_auto(s: pd.Series) -> pd.Series:
    """Nhận ms (13 chữ số) hay us (16 chữ số) cho TỪNG FILE rồi đổi sang UTC.
    Binance đổi sang microsecond từ file 2025-01 trở đi."""
    unit = "us" if int(s.iloc[0]) > 1e14 else "ms"
    return pd.to_datetime(s, unit=unit, utc=True)


def read_month_file(fobj) -> pd.DataFrame:
    df = pd.read_csv(fobj, header=None, names=COLS)  # file Binance không có header
    if str(df.iloc[0, 0]).strip().lower() == "open_time":  # phòng khi có header
        df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric)
    df["open_time"] = to_datetime_auto(df["open_time"])
    df["close_time"] = to_datetime_auto(df["close_time"])
    return df.drop(columns=["ignore"])  # cột ignore luôn = 0


def main():
    cfg = load_paths()
    zpath = p(cfg["download"]["binance_zip"])
    out_dir = p(cfg["raw"]["binance_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zpath) as z:
        for sym in SYMBOLS:
            names = sorted(n for n in z.namelist() if n.startswith(f"{sym}/") and n.endswith(".csv"))
            print(f"[{sym}] {len(names)} file tháng")
            parts = []
            for n in names:
                with z.open(n) as f:
                    parts.append(read_month_file(f))
            df = (pd.concat(parts).drop_duplicates("open_time")
                    .sort_values("open_time").reset_index(drop=True))
            df.insert(0, "symbol", sym)

            out = save_df(df, out_dir / f"{sym}_1h.parquet")
            print(f"   {len(df):,} nến | {df.open_time.min()} -> {df.open_time.max()} | lưu {out.relative_to(ROOT)}")
            write_source_log(f"Binance Vision {sym} 1h", len(df), df.open_time.min().date(),
                             df.open_time.max().date(), "ms for 2020-2024, us for 2025")


if __name__ == "__main__":
    main()

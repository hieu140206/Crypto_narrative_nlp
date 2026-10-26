"""TV1 - Nạp CryptoVision, DROP 10 cột leakage, chuẩn hóa UTC, log Max date.

Chạy:  python src/data/ingest_cryptovision.py
Input : zip CryptoVision (đường dẫn trong config/paths.yaml)
Output: data/raw/cryptovision/cryptovision_clean.parquet
"""
import zipfile

import pandas as pd

from common import ROOT, load_paths, p, save_df, set_seed, write_source_log

set_seed(42)

# 10 cột leakage (market + sentiment có sẵn của CryptoVision) - phải bỏ hết
LEAKAGE_COLS = [
    "sentiment_label", "sentiment_score",
    "Open", "High", "Low", "Close", "Volume",
    "Movement_OpenClose_%", "Movement_HighLow_%", "Market_Move",
]
KEEP_COLS = ["URL", "Title", "Date Time", "Coin Type"]


def read_raw(cfg) -> pd.DataFrame:
    zpath = p(cfg["download"]["cryptovision_zip"])
    with zipfile.ZipFile(zpath) as z:
        name = [n for n in z.namelist() if n.endswith("CryptoDataSet.csv")][0]
        with z.open(name) as f:
            # low_memory=False: file gốc có vài dòng bị vỡ nên cột bị lẫn kiểu dữ liệu
            return pd.read_csv(f, low_memory=False)


def main():
    cfg = load_paths()
    df = read_raw(cfg)
    n_raw = len(df)
    print(f"[1] Đọc xong: {n_raw:,} dòng, {df.shape[1]} cột")

    # --- DROP leakage ngay lập tức ---
    df = df.drop(columns=LEAKAGE_COLS)
    df = df[KEEP_COLS].rename(columns={"Date Time": "date_time", "Coin Type": "coin_type",
                                       "URL": "url", "Title": "title"})
    assert not any(c in df.columns for c in LEAKAGE_COLS), "Còn cột leakage!"
    print(f"[2] Đã drop 10 cột leakage. Còn lại: {df.columns.tolist()}")

    # --- Chuẩn hóa thời gian về UTC ---
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True, errors="coerce")

    # --- Loại dòng hỏng (thiếu URL/Title/ngày/coin) ---
    bad = df[["url", "title", "date_time", "coin_type"]].isna().any(axis=1)
    print(f"[3] Dòng hỏng/thiếu giá trị bị loại: {bad.sum():,}")
    df = df[~bad].copy()

    # --- Bỏ dòng trùng (cùng bài + cùng coin). Cùng 1 URL xuất hiện ở nhiều coin là BÌNH THƯỜNG, giữ lại ---
    n0 = len(df)
    df = df.drop_duplicates(subset=["url", "coin_type"])
    print(f"[4] Bỏ trùng (url, coin_type): {n0 - len(df):,} dòng")

    df = df.sort_values("date_time").reset_index(drop=True)

    # --- Log Max date (CryptoVision KHÔNG phủ hết 2025) ---
    mn, mx = df["date_time"].min(), df["date_time"].max()
    msg = f"CryptoVision: min={mn}, MAX={mx}, rows={len(df):,}, unique_url={df['url'].nunique():,}"
    print("[5]", msg)
    log_path = p(cfg["logs"]["max_date_log"]); log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(msg + "\n", encoding="utf-8")

    # --- Lưu ---
    out_dir = p(cfg["raw"]["cryptovision_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    out = save_df(df, out_dir / "cryptovision_clean.parquet")
    write_source_log("CryptoVision (Mendeley)", len(df), mn.date(), mx.date(),
                     f"raw={n_raw}; dropped 10 leakage cols")
    print(f"[6] Đã lưu {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

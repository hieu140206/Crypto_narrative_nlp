"""TV1 - Kiểm tra chất lượng dữ liệu (Milestone 1, 2 + check gap). Chạy sau 2 script ingest.

Chạy:  python src/data/verify_data.py
"""
import pandas as pd

from common import load_df, load_paths, p

cfg = load_paths()

# ---------- CryptoVision ----------
cv = load_df(p(cfg["raw"]["cryptovision_dir"]) / "cryptovision_clean.parquet")
leak = ["sentiment_label", "sentiment_score", "Sentiment_Label", "Open", "High", "Low", "Close",
        "Volume", "Market_Move", "Movement_OpenClose_%", "Movement_HighLow_%"]
assert not any(c in cv.columns for c in leak), "FAIL: còn cột leakage"
print("OK  Milestone 1: không còn cột leakage; cột =", cv.columns.tolist())
print("    Timezone:", cv["date_time"].dt.tz, "| Max date:", cv["date_time"].max())
if cv["date_time"].max() < pd.Timestamp("2025-12-31", tz="UTC"):
    print(f"WARN CryptoVision chỉ tới {cv['date_time'].max().date()} (< 31/12/2025) "
          "-> báo TV2 lấp bằng Free Crypto News, ghi vào Limitations")

# ---------- Binance ----------
for sym in ["BTCUSDT", "ETHUSDT"]:
    b = load_df(p(cfg["raw"]["binance_dir"]) / f"{sym}_1h.parquet")
    assert b["open_time"].min() >= pd.Timestamp("2020-01-01", tz="UTC"), "FAIL: timestamp lệch"
    assert b["open_time"].max() < pd.Timestamp("2026-01-01", tz="UTC"), "FAIL: timestamp lệch (ms/us lẫn)"
    print(f"OK  Milestone 2 {sym}: {b.open_time.min()} -> {b.open_time.max()}")
    full = pd.date_range("2020-01-01", "2025-12-31 23:00", freq="1h", tz="UTC")
    missing = full.difference(pd.DatetimeIndex(b["open_time"]))
    print(f"    Số nến: {len(b):,} / kỳ vọng {len(full):,} | thiếu {len(missing)} nến")
    if len(missing):
        print("    Ngày thiếu nến (top 5):", pd.Series(missing).dt.date.value_counts().sort_index().head(5).to_dict())
    assert b["open_time"].is_unique, "FAIL: trùng open_time"
    assert (b[["open", "high", "low", "close"]] > 0).all().all(), "FAIL: giá <= 0"
    assert (b["high"] >= b["low"]).all(), "FAIL: high < low"
print("\nXong. Không có FAIL -> bàn giao data/raw/ cho nhóm.")

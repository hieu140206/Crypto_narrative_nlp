"""TV1 - Xem dữ liệu trực quan: xuất file CSV mở bằng Excel + vẽ 3 biểu đồ PNG.

Chạy:  python src/data/preview_data.py
Output: thư mục preview/ (mở bằng File Explorer)
"""
import pandas as pd

from common import ROOT, load_df, load_paths, p

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("Chưa cài matplotlib. Gõ: pip install matplotlib   rồi chạy lại.")

cfg = load_paths()
out = ROOT / "preview"; out.mkdir(exist_ok=True)

cv = load_df(p(cfg["raw"]["cryptovision_dir"]) / "cryptovision_clean.parquet")
btc = load_df(p(cfg["raw"]["binance_dir"]) / "BTCUSDT_1h.parquet")
eth = load_df(p(cfg["raw"]["binance_dir"]) / "ETHUSDT_1h.parquet")

# ---------- 1) CSV để mở bằng Excel ----------
def tz_naive(df):  # Excel không thích múi giờ, bỏ đuôi +00:00 (dữ liệu vẫn là UTC)
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime64"):
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df

# Tin BTC/ETH (utf-8-sig để Excel đọc đúng ký tự đặc biệt)
tz_naive(cv[cv["coin_type"].isin(["Bitcoin", "Ethereum"])]).to_csv(out / "tin_tuc_BTC_ETH.csv", index=False, encoding="utf-8-sig")
# Giá theo NGÀY (gộp từ nến 1h) cho dễ nhìn - file 1h gốc có 52k dòng
daily = {}
for name, b in [("BTC", btc), ("ETH", eth)]:
    d = b.set_index("open_time").resample("1D").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"))
    daily[name] = d
    tz_naive(d.reset_index()).to_csv(out / f"gia_{name}_theo_ngay.csv", index=False, encoding="utf-8-sig")
# 200 dòng đầu của giá 1h
tz_naive(btc.head(200)).to_csv(out / "gia_BTC_1h_200_dong_dau.csv", index=False, encoding="utf-8-sig")

# ---------- 2) Biểu đồ ----------
# Hình 1: giá đóng cửa BTC và ETH
fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for a, (name, d) in zip(ax, daily.items()):
    a.plot(d.index, d["close"], lw=1); a.set_ylabel(f"{name} (USDT)"); a.grid(alpha=.3)
ax[0].set_title("Giá đóng cửa theo ngày (Binance, UTC)")
fig.tight_layout(); fig.savefig(out / "1_gia_BTC_ETH.png", dpi=120); plt.close(fig)

# Hình 2: số bài tin mỗi tháng (BTC vs ETH) -> thấy rõ chỗ thiếu cuối 2025
n = cv[cv["coin_type"].isin(["Bitcoin", "Ethereum"])].copy()
n["month"] = n["date_time"].dt.tz_convert("UTC").dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
cnt = n.groupby(["month", "coin_type"]).size().unstack(fill_value=0)
cnt = cnt[cnt.index >= "2020-01-01"]
fig, ax = plt.subplots(figsize=(11, 4))
cnt.plot(ax=ax); ax.set_title("Số bài tin mỗi tháng (CryptoVision) - thấy đường rơi ở 08/2025")
ax.set_xlabel(""); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "2_so_bai_moi_thang.png", dpi=120); plt.close(fig)

# Hình 3: nguồn tin nào nhiều nhất
src = cv["url"].str.extract(r"https?://(?:www\.)?([^/]+)")[0].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 4))
src.sort_values().plot.barh(ax=ax); ax.set_title("Top 10 nguồn tin trong CryptoVision")
fig.tight_layout(); fig.savefig(out / "3_top_nguon_tin.png", dpi=120); plt.close(fig)

print("Xong! Mở thư mục:", out)
for f in sorted(out.iterdir()):
    print("  -", f.name)

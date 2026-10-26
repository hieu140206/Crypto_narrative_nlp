# Câu hỏi / cần quyết định (không tự quyết)

7. **params.yaml hash PREREG**  
   Đã thêm `own_n_min: 5` vào `config/params.yaml` (ngưỡng own sentiment). Hash PREREG (`dc6270e8…`) sẽ lệch cho đến khi nhóm duyệt lại / cập nhật mục Sai khác.

9. **params.yaml hash sau khối `ml:` (họ 5)**  
   Thêm khối `ml:` (lưới EN/XGB cố định) → sha256 mới `cc7ff1f83b01077067daabef1242b258dfd0517725c57f3a347308d208616610`. Hash đóng băng trong PREREG (`f02edc93…`) lệch. Cần ghi Sai khác / cập nhật fingerprint khi duyệt — agent không sửa `docs/PREREG.md`.

8. **embargo_drop theo horizon**  
   Spec yêu cầu cột `embargo_drop` (số ít) nhưng báo cáo “theo split và h”. Đã tạo `embargo_drop_1/3/7` + `target_end_date_1/3/7`. `load_model_table(..., horizon=h)` lọc đúng cột; `horizon=None` loại nếu bất kỳ cờ True. Có muốn thêm cột tổng `embargo_drop = any(...)` không?

1. **Thiếu 3 file md spec trong `docs/`**  
   PACK yêu cầu đặt 3 file md của dự án (guide / analysis / …) vào `docs/`. Hiện repo chỉ có `README.md` (đã copy sang `docs/README.md`). Cần cung cấp 3 file spec gốc để agent không tự bịa.

2. **Thiếu `data/raw/cryptovision/cryptovision_v1.csv`**  
   Không chạy lại được `clean_merge.py` trên dữ liệu thật → các test `text_*` / `panel_*` vẫn skip. Cần đường dẫn file CryptoVision (hoặc đặt đúng `config/paths.yaml`).

3. **FINDINGS #7–#8 (market_daily / daily_panel / Binance 2026)**  
   Chưa có script TV2/TV3 tạo `market_daily.parquet` và `daily_panel.parquet`, cũng chưa có `data/raw/binance`. Có nên scaffold `build_market_daily.py` + `build_daily_panel.py` ngay, hay chờ raw Binance + sentiment_doc?

4. **Cursor auto-run / denylist**  
   Đã thêm `.cursor/permissions.json` (block git commit/push). UI Cursor: Settings → Agents → Approvals & Execution nên để **Allowlist** (không Auto-run). Xác nhận giúp nếu key permissions khác trên bản Cursor đang dùng.

5. **abvol look-ahead wording**  
   Spec ảnh: "đổi volume ngày t không đổi abvol ngày t". Công thức abvol_t = log(vol_t) − mean(t−1..t−30) thì volume_t *có* đổi abvol_t. Đã test chống nhìn tương lai (abvol_{t−1} bất biến). Xác nhận công thức đúng như vậy.

6. **FinBERT full run / GPU**  
   Trial 500 trên **CPU** (~286 ms/doc → ước **~5 giờ** cho 62 758 bài). Cài torch CUDA thất bại vì **ổ C: hết chỗ** (~2 GB trống; GPU thực tế là RTX 3050 Laptop, không phải 3070 Ti). Cần dọn C: rồi `pip install torch --index-url https://download.pytorch.org/whl/cu124` trước khi chạy full.

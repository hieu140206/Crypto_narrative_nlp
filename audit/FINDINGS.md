# Phát hiện từ audit 23/10/2026 (cần sửa theo thứ tự)
1. Khung mẫu: T_END=2025-06-30; bắt đầu 2021-03-01. Hiện clean_merge dùng 2020-01-01 → 2025-08-17.
2. Thêm `news_date`; `in_main_sample` phải theo cửa sổ chính (hiện luôn True).
3. Cryptonews: ngày dump 2023-06-26 (499 bài) và 2023-10-23 (941 bài) tạo AbnAttention giả (3,17 và 3,84). Gắn cờ theo ngưỡng bài/nguồn/ngày; bỏ Cryptonews khỏi bản chính.
4. Tiêu đề rác ("X icon" ×13, "Error 500"), tiêu đề < 20 ký tự (mục từ điển), 94 dòng trùng tiêu đề.
5. `nlp_text` RSS (title+summary) ≠ CryptoVision (title). Thống nhất; gắn thẻ BTC/ETH cho RSS bằng từ khoá; thu hẹp regex `crypto_relevant`.
6. `clean_rss.py`: dedup tiêu đề phải chặn nhầm tiêu đề chỉ khác con số.
7. Bảng thị trường ngày chưa có; Binance cần tải thêm 2026 cho holdout.
8. daily_panel.parquet hiện chỉ 1 dòng; AbnAttention=0.0 phải là NaN.

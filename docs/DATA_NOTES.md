# Ghi chú dữ liệu

## Sentiment: title (chính) vs nlp_text (robustness)
Audit 24/10/2026: `nlp_text = title + description` **không đồng nhất giữa nguồn**. Decrypt có description rỗng ~76% (2023–2024 ~99%); CoinDesk/Cointelegraph/CryptoPanic gần như luôn có (~110–133 ký tự). Hệ quả quan sát được: VADER positive Decrypt ~33% vs ~52% nguồn khác; FinBERT neutral giảm theo năm trên một số nguồn.

**Quyết định:** bản chính chạy sentiment trên **`title`** (`sentiment_doc.parquet`, `text_variant='title'`). Giữ `nlp_text` làm **robustness** (`sentiment_doc_fulltext.parquet`, `text_variant='nlp_text'`).

## FRED
Các chuỗi VIXCLS / DGS10 / DTWEXBGS lấy từ `fredgraph.csv` là **current vintage**, không phải point-in-time (ALFRED). Giá trị có thể bị revise sau ngày phát hành. Join vào panel theo `usable_from_utc`, không theo `date`.

- VIXCLS: `usable_from = date + 1 ngày`
- DGS10: `usable_from = date + 2 ngày` (H.15 công bố chiều ngày làm việc kế tiếp ~20–21h UTC)
- DTWEXBGS: thứ Ba sau Friday của tuần quan sát

## Binance daily
`sample.market_end` trong `config/params.yaml` (mặc định `2025-12-31`). Đổi thành 2026 khi bổ sung holdout.

# text_pipeline.py: hướng dẫn chạy

Pipeline chuẩn hoá corpus text cho đề tài 10 (Crypto Narrative & NLP Sentiment). Phần đã kiểm tra: **chuẩn hoá CryptoVision v1**. Phần RSS **chưa kiểm tra trên feed thật** (xem mục 4).

Đọc kèm `data_source_guide_crypto_narrative.md` (quy tắc chống leakage, lý do chọn mẫu từ 2021).

## 1. Cài đặt

```bash
pip install pandas pyarrow feedparser requests python-dateutil
```

Python 3.10 trở lên. File CryptoVision CSV khoảng 900 MB; máy tác giả chạy với khoảng 8 GB RAM (chưa đo mức tối thiểu).

## 2. Dữ liệu đầu vào

Tải **CryptoVision v1** (không phải v2):

- https://data.mendeley.com/datasets/wvjjxr8bxx/1 (DOI 10.17632/wvjjxr8bxx.1)
- File CSV có 17 cột, trong đó có `Full Text`. Bản v2 (`3c3xtxtfb6.2`) không có cột text, không dùng được cho pipeline này.

Giải nén vào một thư mục riêng, ví dụ `data/raw/cryptovision/`. Giữ thư mục này chỉ chứa file dữ liệu, và không đặt script vào đó.

## 3. Chạy

Tạo `config/sources.json`:

```json
{"dataset": {"assume_timezone": "UTC"}, "rss": []}
```

Chuẩn hoá CryptoVision (khoảng 2 phút):

```bash
python text_pipeline.py cryptovision \
  --input data/raw/cryptovision \
  --output data/interim/cryptovision_norm.parquet \
  --config config/sources.json
```

Kiểm tra chất lượng:

```bash
python text_pipeline.py audit \
  --input data/interim/cryptovision_norm.parquet \
  --output-dir outputs/audit
```

**Kết quả đúng phải khớp các số sau** (nếu lệch nhiều, báo lại, đừng dùng tiếp):

| Mục | Giá trị |
|---|---|
| Dòng đọc vào | 188.430 |
| Sau lọc BTC/ETH và dedup | 105.160 |
| Khoảng thời gian | 2017-08-17 → 2025-08-27 |
| `audit_ts_bulk_flag` | 373 |
| `audit_ts_regime_risk` | 28.436 |

File cần gửi lại để kiểm tra: `quality_report.json`, `articles_by_source_year.csv`, `articles_by_week_source.csv`, `daily_density_by_asset_year.csv`.

## 4. RSS (chưa giao)

Lệnh `rss` và `merge` có trong script nhưng:

- Chưa chạy trên feed thật, chỉ thử bằng feed giả.
- Dùng `config/sources.example.json` làm `config/sources.json`. Ba feed (CoinDesk, Cointelegraph, Decrypt) đã kiểm tra ngày 20/10/2026: RSS 2.0 hợp lệ, `pubDate` UTC, chỉ có tóm tắt. Không dùng feed Ethereum blog.
- Cửa sổ feed rất ngắn (25 bài CoinDesk khoảng 2 ngày, 30 bài Cointelegraph khoảng 1,5 ngày), nên poll mỗi 30 phút; không quá 6 giờ.
- Script chạy một lần rồi thoát. Phải tự đặt lịch (cron hoặc task scheduler) chạy mỗi 15–30 phút và để máy bật liên tục. Mỗi lần chạy **cộng dồn** vào file output, dòng có `is_backfill = True` không tính vào holdout live.

## 5. Các cột quan trọng trong output

| Cột | Ý nghĩa |
|---|---|
| `published_at_utc` | Thời điểm xuất bản, UTC. Ngày gán cho tin là ngày UTC của cột này |
| `nlp_text` | `title + description`. **Đưa cột này vào mô hình NLP** |
| `nlp_text_body` | Full text. Chỉ dùng cho kiểm tra độ bền |
| `asset_tags` | JSON list, đã gộp mọi bản sao của cùng một URL |
| `ts_regime_risk` | `True` cho CryptoPanic và CoinDesk trước 2021 (timestamp lệch 6 giờ ở một số tháng) |
| `ts_bulk_flag` | `True` khi một nguồn có từ 8 bài trở lên trùng một giây (nạp hàng loạt) |
| `ts_minute_only` | Timestamp chỉ có độ phân giải phút |
| `sentiment_label_original` | Nhãn của tác giả. **Chỉ để đối chiếu, không làm feature** |
| `collected_at_utc` | Trống với dữ liệu CryptoVision; có giá trị với RSS |

## 6. Quy tắc bắt buộc khi dùng corpus

- Mẫu chính **2021-03-01 → 2025-06-30** (Cointelegraph kết thúc ngày 30/6/2025 trong v1). 2019–2020 chỉ làm robustness.
- Không có cột giá hay `Movement_*` trong output; đừng thêm lại từ file gốc.
- Spec dự báo: tin ngày `t` chỉ dự báo `t+1`.
- Bảng robustness phải có phiên bản bỏ các dòng bị gắn cờ `ts_bulk_flag` và `ts_regime_risk`.

## 7. Chưa nằm trong pipeline

Free Crypto News `archive` (đã hạ xuống tuỳ chọn), CoinDesk HF và Reddit. Giá Binance, FRED và mẫu nhãn tay nằm ở mục 8. Xem guide.

## 8. Các script khác

**`market_data.py`**: giá Binance spot (nến 1h) và FRED. Chạy trên máy có internet (môi trường của tác giả bị chặn nên chỉ kiểm tra bằng máy chủ giả).

```bash
python market_data.py binance-download --output-dir data/raw/binance          # BTC, ETH, 2020-12 → 2025-08
python market_data.py binance-build --raw-dir data/raw/binance --output-dir data/interim/binance
python market_data.py fred --output data/interim/fred.parquet
```

- `binance-download` kiểm tra checksum, bỏ qua file đã có, và thoát với mã 2 nếu thiếu tháng hay sai checksum.
- `binance-build` tự nhận diện timestamp mili giây hay micro giây và có header hay không, rồi tạo bảng ngày UTC (`ret`, `rv`, `log_rv`, `volume`, `abvol`). Ngày thiếu giờ bị đánh dấu `complete_day = False` và `rv` để trống.
- `fred` tải VIX, chỉ số USD diện rộng (`DTWEXBGS`, không phải DXY của ICE vì DXY không có trên FRED) và lợi suất 10 năm. **Join theo cột `usable_from_utc`, không join theo `date`**: VIX và lợi suất có độ trễ một ngày, còn `DTWEXBGS` công bố theo tuần nên chỉ dùng được từ thứ Ba sau thứ Sáu của tuần đó.

**`make_label_sample.py`**: rút mẫu nhãn tay 630 headline (9 ô × 70, phân tầng theo giai đoạn × nhóm nguồn, seed cố định). Thư mục `labeling/` chứa `label_task.csv` (gửi cho người gán, đã ẩn ngày và nguồn), `label_key.csv` (giữ riêng) và `label_guidelines.md`.

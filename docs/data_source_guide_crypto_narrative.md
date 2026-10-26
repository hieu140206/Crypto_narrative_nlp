# Hướng dẫn nguồn dữ liệu — Đề tài 10: Crypto Narrative & NLP Sentiment Intelligence

> Tài liệu tổng hợp: từng nguồn là gì, dùng làm gì, lấy ở đâu, lưu ý gì.
> Tính đến 20/10/2026 (đã chạy cổng kiểm tra dữ liệu trên file CryptoVision v1 đầy đủ). Các giới hạn API và giấy phép thay đổi nhanh, hãy kiểm tra lại trước khi dựa vào.
> Các ước lượng về sức mạnh thống kê trong tài liệu là tính gần đúng, không phải kết quả đã kiểm chứng.

---

## 0. Tóm tắt quyết định

| Hạng mục | Đề xuất |
|---|---|
| Tin chính | **CryptoVision v1** (Mendeley, DOI 10.17632/wvjjxr8bxx.1, có `Full_Text`), tự tính sentiment, bỏ các cột kết quả. Bản v2 (`3c3xtxtfb6.2`) chỉ có 14 cột metadata, không có text |
| Tin đối chiếu | **CoinDesk HF** (1/2020–1/2025, một outlet) cho robustness 2021–2024. Free Crypto News `archive` **hạ xuống tuỳ chọn** (mục 2.2) |
| Tin mới (holdout live) | **RSS tự thu thập**, bật ngay |
| Giá | **Binance Vision** (nến 1h cho volatility) |
| Sentiment | Lexicon (baseline) + **FinBERT** (chính) + **CryptoBERT** (đối chiếu) + **600+ nhãn tay** (xem 2.5) |
| Văn bản đưa vào NLP | **`title + description`** (khớp với nhãn tay trên headline). Body chỉ làm robustness |
| Time range | **2021-03-01 → `T_end` = 2025-06-30.** Dữ liệu thô của CryptoVision v1 dừng 27/8/2025, nhưng Cointelegraph (nguồn lớn nhất của 2024) gần như biến mất từ 1/7/2025 (7 bài trong 7/2025 so với 531 bài trong 6/2025) nên cơ cấu nguồn đứt gãy lần nữa. 2019–2020 chỉ làm robustness vì cơ cấu nguồn đứt gãy và timestamp lệch (mục 5.3) |
| Phạm vi 10–12 tuần | **Lõi:** Binance spot, CryptoVision, RSS, nhãn tay, FRED. **Làm sau khi lõi chạy xong:** CoinDesk HF, Reddit, Binance futures. **Cắt hoặc hạ:** Free Crypto News archive (tuỳ chọn), Wikipedia/Google Trends, GDELT GKG, DLT-Sentiment-News |
| Không dùng | GDELT DOC để bù cuối 2025, CoinGecko bản miễn phí, CryptoPanic free tier |

---

## 1. Logic chung: ba lớp dữ liệu

Đề tài hỏi một câu: **text có chứa thông tin về return, volatility, volume mà biến thị trường chưa có không?** Để trả lời cần ba lớp:

| Lớp | Nhiệm vụ | Nguồn |
|---|---|---|
| 1. Biến kết quả và baseline | Cái cần dự báo, và thông tin thị trường đã có | Binance (spot, futures), FRED |
| 2. Tín hiệu text | Cái cần kiểm tra có giá trị tăng thêm không | Tin tức, Reddit |
| 3. Kiểm chứng và kiểm soát | Chứng minh text signal đo đúng thứ cần đo, loại giải thích thay thế | Nhãn tay, DLT-Sentiment-News, Wikipedia, Google Trends |

Có nhiều nguồn tin không phải để "nhiều dữ liệu hơn", mà vì mỗi nguồn che một điểm yếu của nguồn kia: lịch sử dài, kiểm tra chéo, holdout sạch, góc nhìn nhà đầu tư cá nhân.

---

## 2. Chi tiết từng nguồn

### 🔴 Bắt buộc

#### 2.1 CryptoVision (tin chính)

- **Là gì:** bộ dữ liệu khoảng 188 nghìn tin crypto, 2017–2025, gom từ 6 nguồn (CoinDesk, Cointelegraph, Decrypt, CryptoPanic...), mỗi bài ghép với nến 15 phút của Binance. Nhãn sentiment do tác giả tạo bằng BERT/DistilBERT đã fine-tune.
- **Vai trò:** chuỗi text chính để tính sentiment, attention, narrative theo ngày.
- **Lấy ở đâu:** Mendeley Data. **Dùng v1**: https://data.mendeley.com/datasets/wvjjxr8bxx/1 (DOI 10.17632/wvjjxr8bxx.1, 29/9/2025, 17 cột gồm `Description`, `Full_Text`, `Filtered_Text`). Bản v2 (https://data.mendeley.com/datasets/3c3xtxtfb6/2, 7/11/2025) theo trang Mendeley chỉ có 14 cột và không có cột text; chưa kiểm tra `max(Date_Time)` của v2.
- **Lưu ý:**
  - **Đã đo trên v1:** 188.430 dòng, `Date_Time` từ 2017-08-17 đến 2025-08-27. Tổng số bài theo ngày còn khá đến giữa 8/2025 (rồi giảm mạnh từ 14/8), **nhưng theo từng nguồn thì Cointelegraph kết thúc ngày 30/6/2025**: 531 bài trong 6/2025, 7 bài trong 7/2025, 2 bài trong 8/2025 (sau lọc BTC/ETH và dedup). Cointelegraph là nguồn lớn nhất của 2024 (7.336 bài). Vì vậy `T_end` = **2025-06-30**: sau mốc này cơ cấu nguồn đứt gãy lần nữa, giống chỗ 2020/2021. Nhìn tổng số bài theo ngày sẽ bỏ sót điều này.
  - **Chưa kiểm tra v2:** một người trong nhóm báo CryptoVision có khoảng 66 bài ngày 15/9/2025, không thể đến từ v1 (v1 dừng 27/8). Nếu họ dùng v2 (chỉ có `Title`, không có `Description`), cần chạy `max(Date_Time)` và số bài theo tuần × nguồn của v2, đặc biệt xem Cointelegraph có quay lại không, trước khi kéo `T_end` ra xa hơn.
  - **Mỗi bài bị nhân bản theo từng coin** (40.678 URL xuất hiện nhiều lần, 109.550 dòng). Dedup phải gộp tag asset theo URL trước khi bỏ bản sao, nếu không khoảng 20% bài mất tag BTC hoặc ETH.
  - **Bỏ khỏi feature** các cột `Movement_OpenClose_%`, `Movement_HighLow_%`, `Market_Move`: chúng được tính từ nến sau khi tin ra, dùng sẽ bị leakage.
  - Nhãn sentiment có sẵn chỉ để **đối chiếu**. Hãy tự tính sentiment bằng mô hình của nhóm để so sánh được nhiều mô hình (RQ1).
  - Nhãn sentiment của tác giả chỉ có hai lớp (NEGATIVE 79%, POSITIVE 21% trên bộ đã lọc BTC/ETH, không có neutral) nên là bản đối chiếu yếu. Giữ dưới tên `sentiment_label_original`, không dùng làm feature.
  - Một trong sáu nguồn là CryptoPanic (aggregator), nên dễ trùng với các nguồn gốc. Cần dedup.
  - **Cơ cấu nguồn đứt gãy** (số bài sau lọc BTC/ETH và dedup, bản pipeline gốc; bản patch ít hơn khoảng 0,5%):

    | Năm | cryptopanic | coindesk | cointelegraph | decrypt | blockworks | cryptonews |
    |---|---|---|---|---|---|---|
    | 2019 | 7.232 | 2.867 | 337 | 13 | 0 | 0 |
    | 2020 | 7.629 | 4.803 | 485 | 22 | 1 | 0 |
    | 2021 | 0 | 5.672 | 5.260 | 2.491 | 1.126 | 0 |
    | 2022 | 0 | 5.862 | 5.670 | 2.351 | 1.436 | 0 |
    | 2023 | 0 | 5.247 | 3.381 | 1.671 | 1.919 | 2.859 |
    | 2024 | 0 | 3.758 | 7.336 | 2.708 | 1.575 | 2.794 |
    | 2025 | 0 | 2.462 | 3.942 | 2.041 | 614 | 3.335 |

    CryptoPanic là aggregator, text trung vị khoảng 300 ký tự (các outlet khác khoảng 3.000 ký tự body). Nó biến mất đúng lúc 2020 chuyển sang 2021.
  - **Timestamp 2019–2020 lệch giữa các nguồn.** Ghép 2.701 cặp bài trùng tiêu đề giữa CryptoPanic và outlet gốc: chênh lệch đúng +6 giờ trong 4–11/2019 và 5–11/2020, đúng 0 giờ trong 12/2019–4/2020. Ít nhất một nguồn lệch múi giờ có hệ thống (chưa xác định nguồn nào). Lệch 6 giờ đẩy khoảng một phần tư số bài sang ngày UTC khác. Từ 2021 không còn cách kiểm tra chéo này; riêng CoinDesk khớp ngày trong URL từ 98,4% đến 99,5% mỗi năm 2020–2024, và còn 95,4% năm 2025 (3,8% lệch +1 ngày).
  - **Nạp hàng loạt:** 509 bài của cryptonews.com năm 2023 dùng chung 32 timestamp (một timestamp có 101 bài). Pipeline gắn cờ `ts_bulk_flag`; các bài này nên loại khỏi chuỗi theo ngày hoặc chạy bảng robustness bỏ chúng. CoinDesk và Cointelegraph chỉ có độ phân giải phút.

#### 2.2 Free Crypto News `archive` (hạ xuống tuỳ chọn)

**Cập nhật 21/10/2026.** Không còn là nguồn "bắt buộc có điều kiện":
- Một người trong nhóm test: `/api/archive` trả 404; `/api/news` chỉ phản chiếu RSS hiện tại (Cointelegraph 30 bài khoảng 1,3 ngày, Decrypt 57 bài khoảng 3–4 ngày, CoinDesk báo lỗi hệ thống); ngày 15/9/2025 chỉ có 3 bài. Tôi chưa tái hiện được vì môi trường của tôi bị chặn mạng tới trang này.
- Tài liệu của dự án (`llms.txt`, README trên GitHub) vẫn liệt kê `/api/archive` là endpoint miễn phí và nêu cả hai kiểu tham số (`date=2024-01` và `start_date`/`end_date`), nên lỗi 404 có thể do gọi sai kiểu tham số. Nếu muốn thử lại: `/api/archive?date=2024-01&limit=5` và đọc nội dung lỗi.
- **Phần tiếng Anh của kho này, theo README, là CryptoPanic** (cộng với Odaily tiếng Trung). CryptoPanic cũng là aggregator chiếm gần hết CryptoVision giai đoạn 2017–2020, nên **không độc lập** với CryptoVision như guide từng viết, và có cùng rủi ro timestamp. Chưa rõ kho này phủ gì sau 2020.
- Vai trò đối chiếu chuyển sang **CoinDesk HF** (phủ trọn 2021–2024) và RSS tự thu cho dữ liệu mới.

- **Là gì:** API miễn phí có kho lưu trữ, tự mô tả hơn 662 nghìn bài, 2017–2025.
- **Vai trò (nếu dùng):** nguồn đối chiếu thứ hai. Lưu ý phần tiếng Anh là CryptoPanic nên không độc lập với CryptoVision (xem trên).
- **Lấy ở đâu:** https://cryptocurrency.cv
  - Xem độ phủ: `/api/archive?stats=true`
  - Lấy theo ngày: `/api/archive?start_date=2024-01-01&end_date=2024-01-07` hoặc theo tháng `/api/archive?date=2024-01` (tài liệu ghi cả hai kiểu, chưa kiểm chứng kiểu nào đang chạy)
  - Tài liệu: https://cryptocurrency.cv/llms.txt
- **Lưu ý:**
  - Một bản mô tả ghi nguồn Odaily có khoảng 316 nghìn bài tiếng Trung, tức gần một nửa kho. **Bắt buộc lọc tiếng Anh** trước khi chạy FinBERT/CryptoBERT.
  - Số liệu là tự công bố, các bản mirror ghi khác nhau (150+ hay 450+ endpoint). Tự kiểm tra bằng `stats=true`.
  - Giấy phép: bản quyền thuộc tác giả, API hosted dùng miễn phí. Chỉ dùng cho nghiên cứu, **không đưa dữ liệu thô lên repo công khai**.
  - Dự án cá nhân, chưa rõ giới hạn tốc độ và độ ổn định. Lưu cache từng lần lấy.
  - Chưa chắc phủ cuối 2025.
  - **Nguy cơ timestamp:** trường thời gian của kho lưu trữ có thể là lúc crawler thấy bài lần đầu hoặc lúc nạp hàng loạt, không phải lúc xuất bản. Nếu vậy, bài 2020 có thể mang timestamp 2025 hoặc dồn vào vài thời điểm nạp, làm hỏng việc gán ngày. Phải qua bài kiểm tra ở mục 6 (bước 4) trước khi dùng nguồn này cho chuỗi theo ngày.

#### 2.3 Binance Vision (giá)

- **Là gì:** kho file nến (kline) miễn phí, tải theo tháng hoặc theo ngày, kèm file CHECKSUM.
- **Vai trò:** tạo return, volatility (RV từ nến 1h, hoặc Parkinson/Garman–Klass từ OHLC ngày), volume, và các biến baseline (lagged return, lagged RV).
- **Lấy ở đâu:** https://data.binance.vision, hướng dẫn tại github.com/binance/binance-public-data
  - Mẫu link: `https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-03.zip`
  - Dùng file tháng cho lịch sử, file ngày cho các ngày chưa vào file tháng gần nhất.
- **Lưu ý:**
  - Đơn vị timestamp không nhất quán giữa các giai đoạn (mili giây / micro giây, header khác nhau). Chuẩn hoá về UTC ngay lớp đầu.
  - Volume chỉ của một sàn, không đại diện toàn thị trường.
  - Khi thêm coin: kiểm tra ngày niêm yết, chọn danh sách coin theo thông tin có ở **đầu mẫu**, tránh "top coin hôm nay" (thiên lệch sống sót).

#### 2.4 RSS tự thu thập (holdout live)

- **Là gì:** script đọc RSS của CoinDesk, Cointelegraph, Decrypt..., ghi cả `published_at` lẫn `collected_at`.
- **Vai trò:** dữ liệu mới, sạch, để kiểm tra out-of-sample thật. Mọi nguồn tin lịch sử ở trên đều dừng ở 2025.
- **Lấy ở đâu (đã kiểm tra ngày 20/10/2026, cả ba trả về RSS 2.0 hợp lệ, `pubDate` đều UTC `+0000`):**
  - CoinDesk: `https://www.coindesk.com/arc/outboundfeeds/rss/` (25 bài, trải khoảng 2 ngày)
  - Cointelegraph: `https://cointelegraph.com/rss` (30 bài, khoảng 1,5 ngày)
  - Decrypt: `https://decrypt.co/feed` (38 bài, khoảng 4 ngày)
  - Cả ba chỉ có tóm tắt, `content:encoded` để trống. Cửa sổ feed rất ngắn nên phải poll thường xuyên; 30 phút là đủ an toàn, 6 giờ là giới hạn trên.
  - Trong CoinDesk có một bài mà ngày trong URL (14/10) lệch `pubDate` (18/10): đây là lý do `published_at` phải đối chiếu, không tin tuyệt đối.
  - Danh sách sẵn dùng cho `text_pipeline.py` nằm ở `config/sources.example.json`.
- **Lưu ý:**
  - **Bật càng sớm càng tốt**: dữ liệu chưa thu thì không lấy lại đúng thời điểm.
  - Feed thường chỉ giữ ít bài gần nhất, nên poll thường xuyên (15–30 phút).
  - Thường chỉ có tiêu đề và tóm tắt.

#### 2.5 Gán nhãn tay 600+ headline

- **Vai trò:** thước đo "sự thật" để so lexicon, FinBERT, CryptoBERT (F1, Cohen's kappa).
- **Cỡ mẫu:** 300–500 là không đủ nếu muốn báo cáo theo năm và theo nguồn: 6 năm × 3 nguồn × 3 lớp (pos/neu/neg) chia ra thì mỗi ô chỉ còn khoảng 10 mẫu hoặc ít hơn, F1 trong ô đó vô nghĩa. Dùng **600+ headline** và báo cáo như sau:
  - Chỉ số chính (F1 macro, kappa) trên **toàn mẫu**, kèm **bootstrap 95% CI**.
  - Phân rã theo **nhóm lớn**: 3 giai đoạn (2021–2022, 2023, 2024–2025; thêm nhóm 2019–2020 chỉ nếu chạy robustness) và 2–3 nhóm nguồn, mỗi nhóm khoảng 200 mẫu. Không báo cáo từng năm × từng nguồn.
  - So hai mô hình bằng **McNemar** hoặc bootstrap theo cặp; chênh lệch F1 cỡ 0,03 giữa các mô hình với mẫu này thường không có ý nghĩa thống kê, báo cáo thẳng điều đó.
- **Cách làm:** lấy mẫu phân tầng theo giai đoạn và nguồn (trọng số bằng nhau giữa các nhóm lớn) để lộ ra sự lệch miền. Hai người gán độc lập, tính độ đồng thuận (kappa giữa hai người), giải quyết bất đồng bằng thảo luận và ghi tỷ lệ bất đồng.
- **Vì sao cần:** mọi nhãn có sẵn trong các dataset đều là nhãn tự động hoặc do cộng đồng bình chọn.

### 🟡 Nên có (làm sau khi phần lõi chạy xong)

#### 2.6 CoinDesk News (Hugging Face)

- **Là gì:** tin của một outlet, 1/1/2020 – 1/1/2025, chia sẻ cho mục đích học thuật, cần ghi nguồn, không dùng để phân phối thương mại lại.
- **Vai trò:** chuỗi một nguồn để kiểm tra độ bền. Một outlet duy nhất loại bỏ vấn đề cơ cấu nguồn đổi theo thời gian.
- **Lấy ở đâu:** https://huggingface.co/datasets/maryamfakhari/coindesk-crypto-news-2020-2025
- **Lưu ý:** độ phủ hẹp (một outlet), dừng ở 1/1/2025.

#### 2.7 Reddit qua Arctic Shift

- **Là gì:** kho Reddit lịch sử, miễn phí, phủ từ 2005. API tìm theo từng subreddit; dữ liệu lớn thì tải dump hàng tháng.
- **Vai trò:** tâm lý nhà đầu tư cá nhân (r/Bitcoin, r/CryptoCurrency, r/ethereum...), attention (số bài, số bình luận). Hữu ích cho RQ4 (retail dẫn hay chạy theo giá).
- **Lấy ở đâu:** github.com/ArthurHeitmann/arctic_shift. Ví dụ API: `https://arctic-shift.photon-reddit.com/api/posts/search?q=...&subreddit=...&limit=100`
- **Lưu ý:**
  - Nhiễu cao (spam, giveaway, airdrop).
  - Dump rất lớn (tháng 7/2026 khoảng 80 GB nén): lọc theo subreddit ngay khi đọc.
  - `score`, `num_comments` là số liệu **tại lúc thu thập**, dùng làm trọng số có nguy cơ look-ahead. Nên dùng trọng số bằng nhau nếu chưa chắc.
  - Có cơ chế xử lý yêu cầu gỡ dữ liệu nên nội dung bị xoá không còn.

#### 2.8 Binance futures (funding, premium index)

- **Vai trò:** làm baseline thị trường mạnh hơn, để phép thử "text có giá trị tăng thêm không" khắt khe hơn (RQ5).
- **Lấy ở đâu:** cùng trang data.binance.vision, mục futures (chưa xác minh đường dẫn cụ thể).
- **Lưu ý:** một số loại dữ liệu chỉ có file ngày.

#### 2.9 FRED (VIX, DXY, lãi suất)

- **Vai trò:** biến kiểm soát vĩ mô, tách tác động của narrative khỏi biến động chung của thị trường tài chính.
- **Lấy ở đâu:** https://fred.stlouisfed.org, tải CSV công khai không cần key (`/graph/fredgraph.csv?id=<SERIES>`).
- **Series dùng:** `VIXCLS` (VIX, ngày), `DGS10` (lợi suất 10 năm, ngày), `DTWEXBGS` (chỉ số USD diện rộng, ngày). **DXY của ICE không có trên FRED**, nên dùng `DTWEXBGS`.
- **Độ trễ công bố (point-in-time):** `VIXCLS` và `DGS10` có độ trễ khoảng một ngày. `DTWEXBGS` công bố **theo tuần** (bản H.10: số liệu đến thứ Sáu được công bố vào thứ Hai sau đó; trang FRED ngày 20/10/2026 ghi quan sát cuối là 16/10, cập nhật 19/10). Vì vậy `market_data.py` gán `usable_from_utc` là ngày + 1 cho VIX và DGS10, và thứ Ba sau thứ Sáu của tuần đó cho `DTWEXBGS`. Luôn join theo `usable_from_utc`.
- **Giới hạn:** FRED trả giá trị phiên bản hiện tại, không phải bản đã công bố lúc đó. Chuỗi bị điều chỉnh thì cần ALFRED mới thật sự point-in-time.

### 🟢 Tuỳ chọn (cắt khỏi phạm vi 10–12 tuần; chỉ làm nếu dư thời gian)

#### 2.10 DLT-Sentiment-News (Hugging Face)

- **Là gì:** 23.301 mẫu, 1/2021 – 5/2025, nhãn do cộng đồng CryptoPanic bình chọn.
- **Vai trò:** chỉ để kiểm tra mô hình sentiment, **không** làm chuỗi chính.
- **Lấy ở đâu:** https://huggingface.co/datasets/ExponentialScience/DLT-Sentiment-News
- **Lưu ý:** nhãn dựa trên vote tích luỹ sau khi đăng và phản ánh người dùng một nền tảng.

#### 2.11 Wikipedia pageviews và Google Trends

- **Vai trò:** proxy attention độc lập với tin tức, để kiểm tra "attention từ tin" có phản ánh sự quan tâm thật hay chỉ là số bài đăng.
- **Wikipedia:** API Pageviews của Wikimedia, miễn phí, không cần key, lịch sử theo ngày cho từng bài (tra "Wikimedia Pageviews API", chưa xác minh link ở đây).
- **Google Trends:** `pytrends` đã bị lưu trữ từ 4/2025, API chính thức còn ở bản alpha giới hạn. Giá trị là chỉ số **tương đối** nên phải cố định bộ từ khoá và neo chuẩn hoá cho cả mẫu.

#### 2.12 GDELT GKG trên BigQuery

- **Là gì:** bảng `gdelt-bq.gdeltv2.gkg_partitioned` chứa URL, theme, tone của tài liệu báo chí toàn cầu.
- **Vai trò:** narrative trong báo chí đại chúng, không chỉ báo chuyên crypto.
- **Lưu ý:** truy vấn lớn có thể vượt quota miễn phí hàng tháng; lọc theo cột phân vùng ngày; **không có full text**; lọc từ khoá URL cho nhiều kết quả nhiễu. Thử một tháng trước để ước lượng chi phí.

---

## 3. Không nên dùng (và lý do)

| Nguồn | Lý do |
|---|---|
| **GDELT DOC API** để bù cuối 2025 | Chế độ danh sách bài chỉ xét khoảng 3 tháng gần nhất. Chế độ timeline đi xa hơn nhưng chỉ là chỉ số tổng hợp, không có từng bài |
| **CoinGecko bản Demo** | Chỉ 365 ngày lịch sử, không phủ cửa sổ mẫu |
| **CryptoPanic free tier** | Gói Developer miễn phí được ghi là đã ngừng; feed công khai chỉ lấy tin mới nhất, không có kho cũ |
| **CryptoCompare / CoinDesk Data (miễn phí)** | Một tài liệu kỹ thuật ghi gói miễn phí đã ngừng từ 21/5/2026. Chỉ một nguồn ghi nên cần tự kiểm tra |
| **RSS làm nguồn lịch sử** | Chỉ lấy được tin gần đây, không tạo ra lịch sử 2020–2025 |

---

## 4. Mô hình sentiment

| Mô hình | Vai trò | Lưu ý |
|---|---|---|
| Lexicon (Loughran–McDonald, VADER) | Baseline | Đơn giản, dễ diễn giải |
| **FinBERT** (`ProsusAI/finbert`, Hugging Face) | **Chính**, hợp với tin tức | Có thể lệch miền với jargon crypto |
| **CryptoBERT** (`ElKulako/cryptobert`, Hugging Face) | Đối chiếu | Train trên mạng xã hội (StockTwits, Telegram, Reddit, Twitter) nên có thể lệch miền với headline. Dữ liệu huấn luyện nằm trong khoảng 2018–2022, **chồng lấn với mẫu train**, cần ghi chú rõ |
| Zero-shot NLI | Gán narrative theo taxonomy cố định | Ổn định, dễ diễn giải cho hồi quy |

Báo cáo nên có bảng so sánh F1 và kappa của từng mô hình trên mẫu nhãn tay: toàn mẫu kèm bootstrap CI, phân rã theo nhóm giai đoạn và nhóm nguồn (xem 2.5).

---

## 5. Time range và chia mẫu

### 5.1 So sánh các điểm bắt đầu

| | Từ 2024 | Từ 2023 | Từ 2020 |
|---|---|---|---|
| Số ngày (đến `T_end` = 30/6/2025) | 547 | 912 | 2.008 |
| Chế độ thị trường | Một (kỷ nguyên ETF) | Phục hồi + ETF | Đầy đủ: COVID, bull, bear 2022, ETF |
| Sức mạnh kiểm định | Thấp | Trung bình | Cao |
| RQ2 (bất đối xứng tin xấu/tốt) | Ranh giới | Khá | Tốt (có 2022: Luna, FTX) |
| RQ5 (out-of-sample) | Yếu | Chấp nhận được | Tốt (có test out-of-regime) |
| Rủi ro text không đều | Thấp | Thấp–trung bình | Cao nhất |
| Rủi ro rò rỉ từ mô hình pretrained | Thấp | Trung bình | Cao nhất |
| Công phân tích theo chế độ | Ít | Vừa | Nhiều |

**Phương án chọn là bắt đầu từ 2021-03-01** (1.583 ngày đến 30/6/2025), nằm giữa cột 2023 và cột 2020. Hai tháng đầu 2021 bị bỏ vì thiếu nguồn (xem mục 5.3). Lý do bỏ 2019–2020 ở mục 5.3: cơ cấu nguồn đứt gãy và timestamp lệch, không phải vì thiếu tin BTC.

Ước lượng gần đúng: sai số chuẩn của hệ số tương quan khoảng 1/√n. Với test 731 ngày là ~0,037, với 639 ngày (đến 30/9/2025) là ~0,040, với 547 ngày (đến 30/6/2025) là ~0,043. Với horizon `h = 7`, return chồng lấn nên số quan sát độc lập hiệu dụng chỉ còn khoảng `n/7` và SE tăng lên cỡ 0,10–0,11. Bảng đầy đủ nằm ở mục 5 của file phân tích. **Số đã đo:** `T_end` = 2025-06-30 cho `N_test` = 547, SE ≈ 0,043; với `h = 7` còn khoảng 78 quan sát độc lập, SE ≈ 0,113.

### 5.2 Đề xuất

**2021-03-01 → `T_end` = 2025-06-30.** Mẫu chính bắt đầu từ **2021-03-01** vì hai tháng đầu 2021 thiếu nguồn: Cointelegraph chỉ có 17 và 48 bài (so với 569 trong 3/2021), Decrypt 8 và 100 (so với 299). Từ 3/2021 bốn nguồn lõi (CoinDesk, Cointelegraph, Decrypt, Blockworks) cùng có mặt. `T_end` lấy từ cổng kiểm tra (mục 6): ngày cuối cùng mà cơ cấu nguồn của CryptoVision v1 còn nguyên vẹn (Cointelegraph biến mất sau 30/6/2025). 2019–2020 chỉ dùng làm robustness (ước lượng riêng, có fixed effect theo nguồn), lý do ở mục 5.3.

| Giai đoạn | Thời gian | Số ngày | Mục đích |
|---|---|---|---|
| Train | 2021-03-01 → 2022-12-31 | 671 | Ước lượng, huấn luyện |
| Embargo | 7 ngày cuối mỗi giai đoạn | — | Tránh rò rỉ qua return chồng lấn |
| Validation | 2023-01-01 → 2023-12-31 | 365 | Tuning, chọn mô hình |
| Test | 2024-01-01 → 2025-06-30 | 547 | Out-of-sample, **không động tới** |

- Thêm walk-forward (cửa sổ mở rộng) và khoảng đệm (embargo) giữa các tập và giữa mỗi lần refit, bằng ít nhất horizon dự báo lớn nhất (`h = 7`).
- Với `N_test` = 547, `h = 1` phát hiện được tương quan từ khoảng 0,12 (2,8 lần SE), `h = 7` từ khoảng 0,32. Kết quả không có ý nghĩa thống kê ở `h = 7` chưa đủ để kết luận text không có tác dụng.
- Nếu sau này chỉ còn CoinDesk HF (dừng 1/1/2025, `N_test` ≈ 366), chuyển sang phương án mẫu ngắn bên dưới hoặc dữ liệu giờ cho `h = 1`.
- Phương án thay đổi tối thiểu nếu nhóm muốn mẫu ngắn hơn: bắt đầu 2023, train 2023-01 → 2024-06, validation 2024-07 → 2024-12, test 2025-01 → 2025-06-30.

### 5.3 Vì sao bắt đầu 2021 thay vì 2020, và điểm cần ghi nhớ

Quyết định này dựa trên dữ liệu đo ngày 20/10/2026, không phải vì mật độ BTC thưa (BTC có khoảng 25–30 bài/ngày từ 2019):

- **Cơ cấu nguồn đứt gãy giữa 2020 và 2021** (bảng ở mục 2.1): train 2020–2022 sẽ trộn hai chế độ text khác hẳn nhau. Một cú nhảy trong `N_t` hay sentiment ngay tại 1/1/2021 sẽ đọc thành tín hiệu.
- **Timestamp 2019–2020 lệch 6 giờ giữa hai nguồn chính**, phá quy tắc gán ngày.
- **ETH 2020 chỉ khoảng 10 bài/ngày** (sau khi gộp tag), dưới ngưỡng 20–30 của mục 6. ETH đạt 16–21 bài/ngày từ 2021.
- **Cái giá:** mất đợt COVID crash (3/2020). RQ2 vẫn còn Luna (5/2022) và FTX (11/2022).

Các điểm còn lại:


- **Thay đổi cấu trúc thị trường**: ước lượng theo giai đoạn, kiểm tra điểm gãy (Chow, Bai–Perron), vẽ hệ số trượt.
- **Taxonomy narrative** chốt trước khi nhìn kết quả: "AI tokens", "ETF" chưa tồn tại ở 2020, cần thêm narrative COVID và macro cho 2020, sụp đổ và depeg cho 2022.
- **Ngoại lệ cực đoan** (ví dụ 12–13/3/2020, FTX): log-RV, winsorize, bảng robustness bỏ các ngày này.
- **Chuẩn hoá attention** bằng tổng số bài mỗi ngày và thêm hiệu ứng cố định theo nguồn.

### 5.4 Khoảng trống từ `T_end` đến nay (gồm 2026)

Cả ba nguồn tin lịch sử dừng ở 2025 (CryptoVision v1 có cơ cấu nguồn đầy đủ đến 30/6/2025), nên khoảng trống hiện đã khoảng 16 tháng. Cách xử lý: dừng mẫu chính ở `T_end`. Chỉ kéo dài `T_end` nếu CryptoVision v2 hoặc nguồn khác cho dữ liệu đủ dày và đủ cơ cấu nguồn (kiểm tra theo từng nguồn, không chỉ tổng theo ngày). RSS bật từ nay chỉ cho khoảng 70–85 ngày sau 10–12 tuần, đủ để kiểm tra pipeline và mô tả, không đủ kiểm định RQ5. Reddit (phủ đến hiện tại) lấp được khoảng trống nhưng là loại dữ liệu khác; chạy như phân tích riêng và ghi rõ.

---

## 6. Cổng kiểm tra trước khi chốt

1. `max(Date_Time)` của CryptoVision và số bài mỗi tuần theo nguồn (đặc biệt quý cuối).
2. `GET /api/archive?stats=true` của Free Crypto News: đếm theo tháng, nguồn, **ngôn ngữ**.
3. Lọc tiếng Anh, đếm số bài mỗi ngày mỗi năm. Quy tắc kinh nghiệm: trung bình khoảng 20–30 bài/ngày/asset thì chỉ số ngày ổn định. Giai đoạn thưa thì gom theo tuần hoặc bỏ.
4. Kiểm tra múi giờ, trường thời gian và tỷ lệ trùng giữa các nguồn. **Với archive, kiểm tra cụ thể timestamp là lúc xuất bản hay lúc crawler/nạp:**
   - Ghép theo URL các bài có mặt ở cả archive và CryptoVision (hoặc CoinDesk HF). Tính độ lệch `timestamp_archive − published_at`: trung vị, phân vị 5%/95%, và tỷ lệ độ lệch âm (timestamp sớm hơn lúc xuất bản là dấu hiệu lỗi).
   - Đếm số bài có **cùng một timestamp chính xác** theo từng tháng. Nhiều bài trùng đúng giây hoặc dồn vào vài thời điểm là dấu hiệu nạp hàng loạt.
   - So ngày trong timestamp với ngày nhúng trong URL (nhiều outlet có `/2022/05/12/` trong đường dẫn) cho mẫu 200–300 bài.
   - **Quyết định:** nếu độ lệch trung vị trong vài giờ và gần như không có nạp hàng loạt thì dùng bình thường. Nếu timestamp là lúc nạp, không dùng archive cho chuỗi theo ngày; chỉ dùng cho kiểm tra thô ở tần suất tuần hoặc bỏ, và CryptoVision là nguồn chính duy nhất.
5. Thử gọi GDELT DOC cho tháng cuối 2025 để xác nhận giới hạn.
6. Kiểm tra giấy phép từng nguồn trước khi đưa dữ liệu lên repo.

**Kết quả lần chạy 20/10/2026 trên CryptoVision v1 (bước 1, 3 và một phần bước 4):**

- Bước 1: `max(Date_Time)` = 2025-08-27 17:47 UTC. Theo từng nguồn, Cointelegraph kết thúc ngày 30/6/2025 → `T_end` = 2025-06-30.
- Bước 3 (sau lọc BTC/ETH, gộp tag và dedup; kiểm tra ASCII ≥ 90% gắn `non_en` cho 464 bài, còn lại coi là tiếng Anh): BTC khoảng 25–45 bài/ngày từ 2019, ETH khoảng 10 bài/ngày năm 2020 và 16–21 bài/ngày từ 2021. Phần trăm ngày có tin ≥ 98%.
- Bước 4: kiểm tra chéo timestamp CryptoPanic với CoinDesk cho thấy lệch đúng 6 giờ trong các tháng nêu ở mục 2.1; khớp ngày URL của CoinDesk; cờ nạp hàng loạt ở cryptonews 2023.
- **Chưa làm:** bước 5 (GDELT) và bước 6 (giấy phép). Bước 2 và bước 4 cho archive chỉ cần nếu quyết định dùng archive (đã hạ xuống tuỳ chọn, xem 2.2); môi trường của tôi bị chặn mạng tới archive nên phải do người có internet chạy.

Điều kiện qua cổng: đủ mật độ tin sau khi lọc tiếng Anh ở năm bắt đầu. Nếu không đạt, bắt đầu muộn hơn.
Nếu Free Crypto News không dùng được: bỏ nó, dùng CryptoVision làm nguồn chính và CoinDesk HF làm nguồn đối chiếu (đây là phương án mặc định).

---

## 7. Quy tắc chống leakage

- Với **spec dự báo**: bài xuất bản trong `[t 00:00, t+1 00:00)` UTC chỉ dùng để dự báo return của ngày **t+1** (hoặc lag chỉ số thêm một ngày như phép thử an toàn). Với **spec đồng thời** (mô tả, RQ1 và RQ3, xem mục 2.3 file phân tích), tin ngày `t` được ghép với biến thị trường ngày `t`, ghi nhãn "không phải dự báo" và controls chỉ dùng biến trễ.
- Chuẩn hoá toàn bộ timestamp về UTC, ghi rõ quy tắc gán ngày.
- Loại cột kết quả trong CryptoVision (`Movement_*`, `Market_Move`), cả `Open/High/Low/Close/Volume`. Pipeline đã kiểm tra: output không chứa các cột này.
- Văn bản đưa vào NLP là `nlp_text` = title + description; body ở cột `nlp_text_body`, chỉ cho robustness.
- Giữ các cờ `ts_regime_risk` (CryptoPanic và CoinDesk trước 2021), `ts_bulk_flag`, `ts_minute_only`; bảng robustness bỏ các bài bị gắn cờ.
- RSS: mọi dòng phải có `collected_at_utc`; dòng có `is_backfill = True` (feed trả bài cũ) không được tính vào holdout live.
- Dedup theo URL và độ giống tiêu đề (hash tiêu đề, simhash hoặc cosine trên embedding), giữ bản sớm nhất.
- Không dùng `score`/`num_comments` của Reddit làm trọng số nếu chưa xác nhận chúng là số liệu tại thời điểm đăng.
- Tune mô hình NLP và ML chỉ trên train và validation; test không động tới.
- Kiểm tra khoảng thời gian huấn luyện của mô hình pretrained và ghi rõ phần chồng lấn với mẫu.
- Chọn danh sách coin theo thông tin có ở đầu mẫu.

---

## 8. Cấu trúc dữ liệu đề xuất

```text
data/
  raw/        # dữ liệu gốc, bất biến, kèm hash file
  interim/    # đã dedup, chuẩn hoá thời gian, gắn asset
  processed/  # chỉ số ngày, bảng hồi quy
```

Schema bài báo:

```text
doc_id               # hash của URL chuẩn hoá
published_at_utc     # thời điểm xuất bản
collected_at_utc     # thời điểm bạn thấy bài (dữ liệu thu từ nay)
source, url, headline, body_optional
lang
title_hash, simhash_cluster_id
asset_tags, tag_method
dataset_origin       # cryptovision / archive / rss / reddit ...
```

**Source log** (bảng riêng): nguồn, endpoint/URL, ngày crawl, phiên bản dataset, số bản ghi, khoảng thời gian, giấy phép, ghi chú lỗi.

Khuyến nghị lưu bằng Parquet và truy vấn bằng DuckDB.

---

## 9. Bảng tổng hợp: nguồn → biến → câu hỏi nghiên cứu

| Nguồn | Tạo ra biến nào | RQ chính | Ưu tiên |
|---|---|---|---|
| Binance spot | return, RV, volume, baseline | 2, 3, 4, 5 | Bắt buộc |
| CryptoVision | Sent, NegSent, PosSent, Attention | 1–5 | Bắt buộc |
| Free Crypto News archive | Chuỗi tin đối chiếu chéo | 1, 5 (độ bền) | **Tuỳ chọn**: chỉ dùng nếu qua kiểm tra timestamp (mục 6); không thay CoinDesk HF |
| RSS live | Holdout nhỏ, kiểm tra pipeline | mô tả | Bắt buộc, bật ngay |
| Gán nhãn tay (600+) | Thước đo validity | 1 | Bắt buộc |
| FRED | Biến kiểm soát vĩ mô | 3, 5 | Lõi |
| CoinDesk HF | Chuỗi một nguồn | độ bền | Làm sau khi lõi chạy xong |
| Reddit | Sentiment retail, attention | 3, 4 | Làm sau khi lõi chạy xong |
| Binance futures | funding, premium | 5 | Làm sau khi lõi chạy xong |
| Wikipedia, Google Trends | Proxy attention độc lập | 3 | Cắt khỏi phạm vi |
| DLT-Sentiment-News | Kiểm tra mô hình | 1 | Cắt khỏi phạm vi |
| GDELT GKG | Narrative báo chí đại chúng | 3 | Cắt khỏi phạm vi |

Các câu hỏi nghiên cứu (theo blueprint):
- **RQ1:** Transformer sentiment có tốt hơn dictionary không?
- **RQ2:** Tin xấu có tác động bất đối xứng mạnh hơn tin tốt không?
- **RQ3:** Narrative attention nào liên hệ mạnh nhất với return, volatility, volume?
- **RQ4:** Text dẫn thị trường hay chỉ phản ứng với giá?
- **RQ5:** Sentiment/narrative có cải thiện dự báo out-of-sample ngoài biến thị trường không?

---

## 10. Ghi chú về độ tin cậy của tài liệu

- Số liệu của Free Crypto News (số bài, tỷ lệ tiếng Trung, số endpoint) là **tự công bố của dự án**, chưa kiểm chứng độc lập. Một thành viên test ngày 21/10/2026 gặp `/api/archive` trả 404; tôi chưa tái hiện được. Dùng bước kiểm tra ở mục 6 nếu muốn dùng nguồn này.
- Chưa xác minh: đường dẫn mục futures trên Binance Vision, trang Wikimedia Pageviews, lịch sử công khai của Fear & Greed Index.
- Thông tin về giới hạn của GDELT DOC API lấy từ thông báo của GDELT và các bài tổng hợp, có thể đã thay đổi. Nên tự thử với `startdatetime`/`enddatetime` cho giai đoạn cần.
- Thông tin về CryptoCompare/CoinDesk Data (ngừng gói miễn phí) chỉ có một nguồn ghi.
- Ước lượng sức mạnh thống kê là gần đúng, chỉ để chọn cỡ mẫu, không thay cho phân tích sức mạnh chính thức.
- Các số đo ngày 20/10/2026 (mục 2.1, 5.3, 6) lấy từ file CryptoVision v1 đã tải, tái lập được bằng `text_pipeline.py` bản sửa lệnh `audit`. Chưa xác định nguồn nào trong cặp CryptoPanic/CoinDesk bị lệch 6 giờ. Chưa kiểm tra `max(Date_Time)` của CryptoVision v2, và không rõ v2 có khác v1 về số dòng hay nội dung ngoài danh sách cột trên trang Mendeley.
- Tài liệu này là thiết kế học thuật, không phải khuyến nghị đầu tư.

# Phân tích đề tài 10: Crypto Narrative & NLP Sentiment Intelligence

> Phân tích chi tiết: lý thuyết, mô hình, ý nghĩa kinh tế, dữ liệu, độ khó, khó khăn và giải pháp, hướng mở rộng, repo và tài liệu liên quan.
> Tính đến 20/10/2026. Đây là thiết kế học thuật, không phải khuyến nghị đầu tư.
> Chi tiết từng nguồn dữ liệu nằm ở file riêng: `data_source_guide_crypto_narrative.md`.

---

## 0. Tóm tắt

**Câu hỏi cốt lõi:** văn bản (tin tức, mạng xã hội) có chứa thông tin về return, volatility và volume mà giá chưa phản ánh hết không? Hay text chỉ là "tiếng vọng" của giá?

Đề tài mạnh nhất khi chấp nhận kết quả âm. Đa số tín hiệu text mất sức dự báo khi kiểm định out-of-sample nghiêm túc. Kết luận "sentiment chủ yếu phản ứng với giá, giá trị tăng thêm nhỏ" nhưng đúng point-in-time vẫn tốt hơn một kết quả đẹp có leakage.

---

## 1. Cơ sở lý thuyết

| Lý thuyết | Nội dung | Gắn với RQ |
|---|---|---|
| Behavioral finance (noise trader, investor sentiment) | Nhà đầu tư cá nhân chiếm tỷ trọng lớn trong crypto, cảm xúc và narrative đẩy giá lệch khỏi giá trị cơ bản | RQ1, RQ5 |
| Limits to arbitrage | Crypto không có giá trị nội tại rõ ràng và khó short, nên mispricing do sentiment tồn tại lâu hơn | RQ5 |
| Sequential information arrival / Mixture of distribution hypothesis | Thông tin đến theo chuỗi làm return, volatility, volume cùng biến động. News flow là proxy của information arrival | RQ3 |
| Attention theory (Barber–Odean) | Attention làm tăng volume và volatility, chưa chắc làm tăng return | RQ3 |
| Asymmetric news impact | Tin xấu (hack, ban, depeg) thường gây phản ứng mạnh hơn tin tốt, giống leverage effect | RQ2 |
| Narrative economics (Shiller) | Câu chuyện lan truyền như dịch bệnh, đổi chế độ thị trường | RQ3 |
| Reverse causality | Giá tăng khiến tin tích cực nhiều hơn, vì báo chí viết theo giá | RQ4 |

Các câu hỏi nghiên cứu (theo blueprint):

- **RQ1:** Transformer sentiment có tốt hơn dictionary sentiment trong giải thích crypto outcomes không?
- **RQ2:** Negative sentiment có tác động bất đối xứng mạnh hơn positive sentiment không?
- **RQ3:** Narrative attention nào liên hệ mạnh nhất với return, volatility và volume?
- **RQ4:** Text signals dẫn thị trường hay chủ yếu phản ứng với price movements?
- **RQ5:** Sentiment/narratives có cải thiện out-of-sample forecast ngoài market variables không?

---

## 2. Mô hình chi tiết

### 2.1 Tầng NLP (đo lường)

**Sentiment, ba thế hệ để so sánh (RQ1):**

1. **Lexicon**: Loughran–McDonald, VADER. Đây là benchmark.
2. **FinBERT**: train trên tin tài chính truyền thống, có thể lệch miền với jargon crypto ("rekt", "HODL", "rug pull").
3. **Mô hình chuyên crypto**: CryptoBERT được post-train trên hơn 3,2 triệu bài đăng crypto (StockTwits, Telegram, Reddit, Twitter), nhãn Bearish/Neutral/Bullish. Với tin chuẩn (CoinDesk, Cointelegraph), FinBERT thường hợp hơn. Với mạng xã hội, CryptoBERT hợp hơn.

**Validation:** tự gán nhãn tay 600+ headline (chi tiết ở mục 2.5 file nguồn dữ liệu), tính F1 và Cohen's κ toàn mẫu kèm bootstrap CI, phân rã theo nhóm giai đoạn và nhóm nguồn. Văn bản đưa vào mô hình là `title + description` để khớp với phần được gán nhãn. Đây là điểm cộng lớn vì đa số nhóm bỏ qua.

**Narrative extraction:**

- **LDA** (benchmark): yếu với văn bản ngắn như headline.
- **BERTopic** (sentence embeddings → UMAP → HDBSCAN → c-TF-IDF): chạy dynamic topic modeling để xem narrative đổi theo thời gian.
- **Zero-shot classification** với taxonomy cố định (regulation, ETF, hack/security, stablecoin, DeFi, AI, macro, speculation). Ổn định và dễ diễn giải nhất cho hồi quy vì topic không trôi giữa các giai đoạn. Nên dùng làm chính; BERTopic để khám phá và kiểm tra chéo.

### 2.2 Tầng chỉ số

- `Sent_t = Σ w_d · S_d / Σ w_d`, với `S_d` = P(pos) − P(neg).
- `NegSent_t`, `PosSent_t`: tách riêng để kiểm định bất đối xứng.
- `Attention_{k,t} = N_{k,t} / N_t`.
- Đề xuất thêm: **sentiment dispersion** (độ bất đồng giữa các bài), **negative-news acceleration** (thay đổi `NegSent` so với trung bình 7 ngày), **novelty** (tỷ lệ bài không trùng tin 3 ngày trước).

### 2.3 Tầng kinh tế lượng

Tách rõ hai loại spec vì chúng trả lời hai loại câu hỏi khác nhau:

| Loại | Timing | Trả lời | Dùng để kết luận |
|---|---|---|---|
| **A. Đồng thời (mô tả)** | Tin trong ngày `t` ↔ biến thị trường ngày `t` | RQ1, RQ3: text liên hệ với thị trường thế nào | Chỉ về **mối liên hệ**, không về dự báo |
| **B. Dự báo** | Tin trong ngày `t` → biến thị trường `t+h`, `h ≥ 1` | RQ2, RQ4, RQ5 | Về **giá trị dự báo** |

Lý do cần A: giả thuyết sequential information arrival và mixture of distribution (Chou et al.) dự đoán attention, volatility và volume cùng biến động **trong cùng ngày**. Chỉ chạy B với quy tắc "tin ngày `t` dự báo `t+1`" thì bỏ sót đúng hiệu ứng mà lý thuyết nói tới. Mọi bảng A phải ghi nhãn "đồng thời, không phải dự báo"; controls trong A vẫn chỉ dùng biến trễ (`t−1` trở về trước).

**A. Spec đồng thời:**

- Return: `r_t = α + β·Sent_t + θ'·Attn_t + γ'X_{t−1} + ε_t`
- Volatility: `log RV_t = α + β⁻·NegSent_t + β⁺·PosSent_t + θ'·Attn_t + γ'X_{t−1} + ε_t`
- Volume: `AbVol_t = α + β·Sent_t + θ'·Attn_t + γ'X_{t−1} + ε_t`
- Sai số chuẩn HAC (Newey–West) vì phần dư tự tương quan.
- Với dữ liệu giờ có thể làm thêm A theo khung giờ để thấy tin và thị trường dịch chuyển cùng lúc đến đâu, nhưng đây vẫn là mô tả.

**B. Dự báo return:** `r_{t+h} = α + β·Sent_t + θ'·Attn_t + γ'X_t + ε`

- `h` = 1, 3, 7 ngày.
- Với `h > 1`, return chồng lấn nên bắt buộc dùng **Newey–West (HAC)**, bandwidth ít nhất `h − 1`. Số quan sát độc lập hiệu dụng khoảng `n/h`, nên khi `h = 7` và `n ≈ 600` chỉ còn khoảng 85 quan sát độc lập; đọc p-value theo đó.
- Controls `X_t`: lagged return, log RV, `AbVol`, VIX hoặc DXY nếu có.

**B. Dự báo volatility:** dùng **HAR-RV làm baseline** (RV ngày, tuần, tháng), rồi thêm `NegSent` và `PosSent`.

- Dùng log(RV) vì RV lệch phải mạnh.
- Test `H0: β⁻ = β⁺` bằng Wald test (HAC).
- **RV chính tính từ nến 1 giờ** (khớp với dữ liệu Binance Vision trong file nguồn). Garman–Klass hoặc Parkinson từ OHLC ngày chỉ là phương án dự phòng và làm robustness.

**B. Dự báo volume:** `AbVol_{t+h} = α + β⁻·NegSent_t + β⁺·PosSent_t + θ'·Attn_t + γ'X_t + ε`

- **`AbVol_t = log(Vol_t) − (1/30)·Σ_{j=1..30} log(Vol_{t−j})`**: log-volume trừ trung bình 30 ngày **trước đó** (chỉ dùng quá khứ, không dùng trung bình có chứa `t` hay tương lai). Cách này khử xu hướng dài hạn của volume mà không gây look-ahead.
- Controls bắt buộc có `AbVol_{t−1}` và `log RV_{t−1}`, vì volume và volatility đồng biến mạnh (mixture of distribution); thiếu chúng thì hệ số text chủ yếu bắt lại biến động.
- Volume lấy từ Binance spot nên chỉ đại diện một sàn; ghi rõ trong báo cáo.

**Biến attention (chống đa cộng tuyến và đo đúng thứ cần đo):**

- Nếu mỗi bài chỉ gán **một** narrative thì `Σ_k Attention_{k,t} = 1`, gây đa cộng tuyến hoàn hảo khi đưa tất cả vào cùng hồi quy. Chọn một trong ba cách:
  1. **Multi-label** (zero-shot dùng ngưỡng sigmoid độc lập cho từng narrative): một bài có thể thuộc nhiều narrative, tổng không còn bằng 1. Đây là cách ưu tiên vì tin crypto thường chạm nhiều chủ đề cùng lúc.
  2. **Bỏ một nhóm tham chiếu** (ví dụ `speculation`) và đọc hệ số theo nhóm đó.
  3. **Log-ratio** `log(Attention_k / Attention_ref)`.
- Dù chọn cách nào, các narrative vẫn có thể tương quan với nhau. Kiểm tra VIF; nếu cao thì chạy từng narrative một (kèm hiệu chỉnh đa kiểm định bên dưới) hoặc dùng Elastic Net.
- **Thêm `log(1 + N_t)` làm biến attention riêng** (khối lượng tin tuyệt đối). `Attention_{k,t} = N_{k,t}/N_t` là tỷ trọng, nên khi tổng số bài tăng mà tỷ trọng không đổi thì tỷ trọng không thấy được. Lý thuyết attention (Barber–Odean) nói về lượng chú ý, không chỉ cơ cấu. Chuẩn hoá `N_t` theo hiệu ứng cố định của nguồn vì cơ cấu nguồn đổi theo thời gian.

**Đa kiểm định:** tổ hợp narrative × asset × horizon × biến kết quả sinh ra rất nhiều hệ số. Chốt trước (trước khi nhìn kết quả) danh sách giả thuyết chính, ví dụ `Sent`/`NegSent` cho BTC và ETH, `h = 1` và `h = 7`; phần còn lại là khám phá và hiệu chỉnh bằng Holm hoặc Benjamini–Hochberg.

**Lead-lag (RQ4):** VAR chọn lag theo AIC/BIC, Granger test, impulse response.

- Kiểm tra tính dừng (ADF/KPSS) trước khi chạy.
- Granger chỉ là "dự báo", không phải nhân quả.
- Với dữ liệu intraday, **event study** quanh thời điểm xuất bản trả lời RQ4 sạch hơn VAR ngày.

### 2.4 Tầng ML (RQ5)

- Baseline: Elastic Net, rồi LightGBM/XGBoost, chỉ dùng biến thị trường.
- Augmented: thêm sentiment, attention, và có thể PCA của embedding.
- LSTM/Transformer chỉ nên làm nếu mẫu đủ lớn (ví dụ dữ liệu theo giờ). Với dữ liệu ngày vài trăm đến vài nghìn quan sát, gần như chắc chắn overfit.
- **Đánh giá:** walk-forward (expanding window), có giai đoạn test cuối **không đụng tới** khi tuning. Giữa các tập (train/validation/test) và giữa mỗi lần refit với tập đánh giá đặt **embargo** bằng ít nhất horizon lớn nhất (`h = 7` ngày), vì return chồng lấn làm nhãn của quan sát sát biên chứa thông tin của phía bên kia.
  - Hồi quy: RMSE/MAE, `R²_OOS` (Campbell–Thompson).
  - **So sánh Model_market với Model_market+text là so sánh mô hình lồng nhau**, nên Diebold–Mariano chuẩn không phù hợp: dưới H0 hai mô hình trùng nhau, phương sai của chênh lệch loss bị nén và DM sẽ bác bỏ quá ít. Dùng:
    - **Clark–West (2007)** cho cặp hồi quy tuyến tính lồng nhau (Elastic Net có thể dùng khi mô hình text chỉ thêm biến vào cùng dạng hàm).
    - **Diebold–Mariano có hiệu chỉnh Harvey–Leybourne–Newbold** và sai số HAC (bandwidth ≥ `h − 1`) cho so sánh giữa các mô hình không lồng nhau, ví dụ LightGBM market-only với LightGBM market+text, hoặc Elastic Net với LightGBM.
    - Báo cáo `R²_OOS` kèm p-value của kiểm định tương ứng, không chỉ chênh lệch RMSE.
  - Hướng: hit ratio, so với baseline "luôn dự báo tăng".
  - Volatility: QLIKE là loss function chuẩn.
- **SHAP** để giải thích. Chiến lược giao dịch chỉ là "economic value test" (net of cost).

---

## 3. Ý nghĩa kinh tế phía sau

- Sentiment dự báo return tốt nhưng đảo chiều vài ngày sau → bằng chứng **sentiment-driven overreaction**.
- Sentiment chỉ dự báo volatility và volume → narrative tạo **information arrival và attention**, không tạo alpha. Đây là kết quả hay gặp nhất, vẫn có giá trị cho quản trị rủi ro.
- Tin xấu có hệ số lớn hơn tin tốt → bất đối xứng kiểu leverage effect, cơ sở cho **risk dashboard**.
- Return dẫn sentiment mà không có chiều ngược lại → text index chỉ là chỉ báo muộn.
- Ứng dụng: cảnh báo sớm rủi ro, giám sát narrative, input cho mô hình volatility hoặc VaR.

---

## 4. Dữ liệu (tóm tắt, chi tiết xem file nguồn dữ liệu)

| Lớp | Nguồn | Vai trò |
|---|---|---|
| Giá (bắt buộc) | Binance Vision (nến 1h, BTC/ETH + 5–10 coin) | return, RV, volume, baseline |
| Tin chính | CryptoVision **v1** (Mendeley, có `Full_Text`; v2 không có text) | Sent, Attention, narrative |
| Tin đối chiếu | CoinDesk HF (một outlet, 2020–1/2025); Free Crypto News `archive` chỉ là tuỳ chọn | Kiểm tra độ bền, loại bỏ phụ thuộc một nguồn |
| Tin mới | RSS tự thu thập (bật ngay) | Holdout live |
| Social | Reddit qua Arctic Shift | Tâm lý retail, attention |
| Baseline mạnh | Binance futures (funding, premium index), FRED | Làm phép thử RQ5 khắt khe hơn |
| Kiểm chứng | Nhãn tay, DLT-Sentiment-News, Wikipedia pageviews, Google Trends | Validity, proxy attention |

**Chuẩn hoá bắt buộc:** timestamp UTC; bài xuất bản trong `[t 00:00, t+1 00:00)` UTC chỉ dùng để dự báo return ngày `t+1`; dedup bằng URL và độ giống tiêu đề; asset tagging bằng từ khoá hoặc NER.

**Các giới hạn đã thấy khi tra cứu:**

- GDELT DOC API ở chế độ danh sách bài chỉ xét khoảng 3 tháng gần nhất; muốn lịch sử phải qua GKG trên BigQuery.
- CryptoPanic: gói Developer miễn phí được ghi là đã ngừng, feed công khai không có kho cũ.
- CoinGecko bản Demo chỉ 365 ngày lịch sử.
- CryptoCompare/CoinDesk Data: một nguồn ghi gói miễn phí đã ngừng từ 21/5/2026 (chỉ một nguồn ghi, cần tự kiểm tra).
- Free Crypto News `archive`: gần một nửa là bài tiếng Trung (theo mô tả của dự án), phần tiếng Anh là CryptoPanic nên không độc lập với CryptoVision; một thành viên test ngày 21/10/2026 gặp `/api/archive` trả 404 (chưa tái hiện được). Đã hạ xuống tuỳ chọn.
- CryptoVision v1 (đã đo): Cointelegraph kết thúc ngày 30/6/2025 nên cơ cấu nguồn đầy đủ chỉ đến 30/6/2025; cơ cấu nguồn đứt gãy giữa 2020 và 2021; timestamp CryptoPanic/CoinDesk 2019–2020 lệch đúng 6 giờ ở một số tháng; mỗi bài bị nhân bản theo coin nên phải gộp tag trước khi dedup. Chi tiết ở mục 2.1 và 5.3 file nguồn dữ liệu.

---

## 5. Time range

**Đề xuất: 2021-03-01 → `T_end` = 2025-06-30.** Mẫu chính bắt đầu từ **2021-03-01** vì hai tháng đầu 2021 thiếu nguồn: Cointelegraph chỉ có 17 và 48 bài (so với 569 trong 3/2021), Decrypt 8 và 100 (so với 299). Từ 3/2021 bốn nguồn lõi (CoinDesk, Cointelegraph, Decrypt, Blockworks) cùng có mặt. 2019–2020 chỉ làm robustness (ước lượng riêng, có fixed effect theo nguồn).

| Giai đoạn | Thời gian | Số ngày | Mục đích |
|---|---|---|---|
| Train | 2021-03-01 → 2022-12-31 | 671 | Ước lượng, huấn luyện |
| Embargo | 7 ngày cuối mỗi giai đoạn | — | Tránh rò rỉ qua return chồng lấn (≥ horizon lớn nhất, `h = 7`) |
| Validation | 2023-01-01 → 2023-12-31 | 365 | Tuning, chọn mô hình |
| Test | 2024-01-01 → 2025-06-30 | **547** | Out-of-sample, không động tới |

**`T_end` đã đo, không còn là biến số.** CryptoVision v1 có dữ liệu thô đến 27/8/2025, nhưng **Cointelegraph kết thúc ngày 30/6/2025** (531 bài trong 6/2025, 7 bài trong 7/2025, 2 bài trong 8/2025; sau lọc BTC/ETH và dedup) và đây là nguồn lớn nhất của 2024 (7.336 bài). Tổng số bài theo ngày vẫn còn khá đến giữa 8/2025 nên nhìn tổng sẽ không thấy; sau 30/6 cơ cấu nguồn đứt gãy lần nữa, giống chỗ 2020/2021. Do đó `T_end` = 30/6/2025. Chưa kiểm tra CryptoVision v2 (một thành viên báo có khoảng 66 bài ngày 15/9/2025, không thể đến từ v1).

| Kịch bản | `N_test` | SE tương quan ≈ 1/√n | Quan sát độc lập hiệu dụng khi `h = 7` (≈ n/7) | SE khi `h = 7` |
|---|---|---|---|---|
| **Thực tế: `T_end` = 30/6/2025** | **547** | **0,043** | **~78** | **~0,113** |
| Chỉ còn CoinDesk HF (dừng 31/12/2024) | 366 | 0,052 | ~52 | ~0,139 |

Đây là tính gần đúng, chỉ để chọn cỡ mẫu. Với `N_test` = 547, `h = 1` phát hiện được hệ số tương quan từ khoảng 0,12 và `h = 7` từ khoảng 0,32 (2,8 lần SE, độ mạnh kiểm định khoảng 80%). Tín hiệu text trong crypto thường nhỏ hơn mức đó ở `h = 7`, nên **kết quả "không có ý nghĩa thống kê" ở `h = 7` chưa đủ để kết luận "không có tác dụng"**. Báo cáo phải nói rõ giới hạn sức mạnh này.

**Vì sao bắt đầu 2021 thay vì 2020** (số liệu đo ngày 20/10/2026, chi tiết ở mục 2.1 và 5.3 file nguồn dữ liệu):

- Cơ cấu nguồn đứt gãy đúng tại 1/1/2021: CryptoPanic (aggregator, text ngắn) biến mất, Cointelegraph tăng từ 485 lên 5.260 bài (sau lọc BTC/ETH và dedup), Blockworks và Decrypt xuất hiện.
- Timestamp CryptoPanic và CoinDesk 2019–2020 lệch đúng 6 giờ ở một số tháng, làm hỏng việc gán ngày UTC.
- ETH 2020 chỉ khoảng 10 bài/ngày, dưới ngưỡng 20–30; từ 2021 là 16–21.
- Cái giá là mất đợt COVID crash. RQ2 vẫn còn Luna (5/2022) và FTX (11/2022).

Lý do vẫn giữ 2021–2025: mẫu có đỉnh 2021, đợt gấu 2022 (Luna, FTX), phục hồi 2023 và kỷ nguyên ETF. Điều này giải quyết điểm yếu của RQ2 (cần đủ ngày tin xấu mạnh).

**Rủi ro của mẫu và cách xử lý:**

- **Thay đổi cấu trúc thị trường**: ước lượng theo giai đoạn, kiểm tra điểm gãy (Chow, Bai–Perron), vẽ hệ số trượt.
- **Text năm đầu có thể thưa hoặc không đều**: đếm số bài theo nguồn và năm trước khi chốt; chuẩn hoá attention bằng tổng số bài mỗi ngày, thêm hiệu ứng cố định theo nguồn.
- **Narrative đổi theo thời gian**: chốt taxonomy trước khi nhìn kết quả; "AI tokens", "ETF" chưa tồn tại ở 2020.
- **Rò rỉ từ mô hình pretrained**: CryptoBERT được huấn luyện trên dữ liệu thuộc các khoảng 2018, 11/2020–1/2021, 5/2021–5/2022 và 11/2021–6/2022, chồng lấn với mẫu train. LedgerBERT dùng nhãn 2021–2025, chồng cả test. Dùng FinBERT và zero-shot làm chính, CryptoBERT làm kiểm tra độ bền với ghi chú rõ phần chồng lấn.
- **Ngoại lệ cực đoan** (12–13/3/2020, FTX): log-RV, winsorize, bảng robustness bỏ các ngày này.
- **Thiên lệch sống sót khi chọn coin**: chọn danh sách coin theo thông tin có ở đầu mẫu, kiểm tra ngày niêm yết.

**Khoảng trống từ `T_end` đến nay (gồm toàn bộ 2026):** các nguồn tin lịch sử dừng ở 2025 (CryptoVision xuất bản 7/11/2025, CoinDesk HF dừng 1/1/2025), nên khoảng trống tin tức tính từ 30/6/2025 hiện đã khoảng 16 tháng và vẫn đang dài thêm.

- **Việc đầu tiên:** chạy `max(Date_Time)` và số bài theo tuần × nguồn của CryptoVision v2 (nếu nhóm đang dùng bản đó), xem Cointelegraph có quay lại không. Free Crypto News archive chỉ thử nếu có thời gian (đã hạ xuống tuỳ chọn).
- **RSS bật từ nay chỉ là holdout nhỏ.** 10–12 tuần cho khoảng 70–85 ngày, quá ngắn để kiểm định RQ5 (SE tương quan khoảng 0,11 trở lên). Chỉ dùng để kiểm tra pipeline chạy đúng trên dữ liệu mới và để mô tả, không dùng để kết luận về dự báo. Nên bật ngay, vì dữ liệu chưa thu thì không lấy lại đúng thời điểm.
- **Reddit/GDELT** phủ đến hiện tại nên có thể lấp khoảng trống, nhưng là loại dữ liệu khác (tâm lý nhà đầu tư cá nhân, không phải tin báo chí). Nếu dùng, ghi rõ và chạy như một phân tích riêng, không ghép vào chuỗi tin chính.

---

## 6. Độ khó

| Thành phần | Độ khó | Ghi chú |
|---|---|---|
| Thu thập và làm sạch text | ⭐⭐⭐⭐ | Khó nhất về công sức: bản quyền, dedup, timestamp |
| Sentiment bằng pretrained model | ⭐⭐ | Chạy được trong vài giờ trên GPU |
| Validation sentiment (gán nhãn tay) | ⭐⭐⭐ | Tốn thời gian nhưng đáng làm |
| Topic modeling / zero-shot | ⭐⭐⭐ | BERTopic cần tinh chỉnh |
| Chống leakage, point-in-time | ⭐⭐⭐⭐ | Dễ sai nhất, khó kiểm tra |
| Kinh tế lượng (HAC, VAR, HAR-RV) | ⭐⭐⭐ | Cần hiểu để diễn giải đúng |
| ML + walk-forward + DM test | ⭐⭐⭐⭐ | Dễ overfit, dễ tự lừa mình |
| Dashboard | ⭐⭐ | Streamlit làm nhanh |

Tổng thể: **mức trung bình–cao**. Phần code không quá nặng; phần khó là kỷ luật phương pháp luận.

---

## 7. Khó khăn thực tế và cách giải quyết

| Khó khăn | Giải pháp |
|---|---|
| **Look-ahead bias** (tin ra sau khi giá đã chạy) | Timestamp UTC chính xác; dự báo `r_{t+h}` chỉ bằng bài xuất bản **trước** thời điểm bắt đầu `h`; lag chỉ số thêm một ngày như phép thử an toàn |
| **Reverse causality** (tin viết theo giá) | Lagged return trong controls, VAR, event study intraday; báo cáo cả hai chiều |
| **Dữ liệu text khó tải / hạn chế API** | Dùng dataset có sẵn (CryptoVision), thêm nguồn đối chiếu; chọn corpus nhỏ nhưng sạch |
| **Tin trùng lặp và nhiễu** | Hash tiêu đề, cosine similarity trên embedding, giữ bản sớm nhất |
| **Domain shift của FinBERT** | So với CryptoBERT; validate trên nhãn tay; fine-tune nhẹ chỉ trên giai đoạn train |
| **Overfitting do nhiều biến, ít quan sát** | Elastic Net; PCA cho embedding; ít narrative hơn; hiệu chỉnh đa kiểm định |
| **Structural break** (2021 bull, 2022 FTX/Luna, 2024 ETF) | Chia sub-sample, kiểm tra Chow, rolling coefficient |
| **Sentiment drift (jargon, mô hình train giai đoạn khác)** | Kiểm tra phân phối score theo thời gian; validate lại từng giai đoạn |
| **Kết quả OOS không có ý nghĩa** | Báo cáo thẳng; nêu rõ giá trị nằm ở risk/volatility chứ không ở alpha; nêu rõ giới hạn sức mạnh kiểm định |
| **Giới hạn tính toán** | Colab GPU; tính embedding một lần và cache |

---

## 8. Hướng mở rộng và nâng cao

1. **LLM-based classification**: LLM với prompt có cấu trúc để gán narrative và sentiment theo từng asset, so với FinBERT. Cần lưu ý chi phí và nguy cơ **look-ahead của chính LLM** (đã biết kết quả giá quá khứ); chỉ nên áp dụng cho giai đoạn sau knowledge cutoff của mô hình.
2. **Cross-asset spillover**: narrative của BTC lan sang altcoin thế nào (VAR hoặc Diebold–Yilmaz).
3. **Text + on-chain + derivatives**: kết hợp funding rate, open interest, whale flow, gắn với các đề tài khác (tail-risk, regime detection).
4. **Regime-dependent effect**: sentiment mạnh hơn lúc căng thẳng? Markov-switching hoặc quantile regression.
5. **Dự báo tail risk**: đưa narrative vào VaR/ES hoặc GARCH-X.
6. **Tin tiếng Việt**: đo sentiment từ báo Việt bằng PhoBERT; so với tin quốc tế (cần tự khảo sát nguồn nào có RSS/archive ổn định, chưa xác minh).
7. **Causal inference**: event study với sự kiện ngoại sinh (ETF approval, hack lớn, vụ kiện SEC) và so asset bị ảnh hưởng với asset đối chứng.
8. **Nowcasting / streaming**: dashboard cập nhật theo giờ, kèm cảnh báo.

---

## 9. Repo và tài nguyên liên quan

Phần lớn là **model/dataset trên Hugging Face và bài báo**; rất ít repo end-to-end đúng blueprint. Coi chúng là mảnh ghép để tự lắp pipeline.

**Model sentiment**

- `Kosmosas/BERT-bitcoin-sentiment` (Hugging Face): FinBERT fine-tune với đầu hồi quy tanh, cho điểm liên tục trong [-1, 1], huấn luyện để dự đoán tác động giá ngắn hạn của headline Bitcoin. Điểm số khớp dạng `Sentiment_d ∈ [-1, 1]` của blueprint, **nhưng không dùng làm sentiment chính**: mô hình được huấn luyện bằng chính phản ứng giá, nên điểm của nó đã chứa "tiếng vọng của giá". Đưa vào hồi quy sẽ làm mờ RQ4 (text dẫn hay chạy theo giá) và thổi phồng RQ5. Nếu muốn dùng làm đối chiếu độ bền: (1) kiểm tra khoảng thời gian dữ liệu huấn luyện và nhãn giá, ghi rõ phần chồng lấn với train/validation/test; (2) báo cáo kết quả tách riêng, gắn nhãn "sentiment đã học từ giá"; (3) chỉ chạy nó trên giai đoạn nằm ngoài dữ liệu huấn luyện của chính mô hình.
- `ElKulako/cryptobert` (Hugging Face): post-train trên hơn 3,2 triệu bài đăng crypto, nhãn Bearish/Neutral/Bullish.
- `ExponentialScience/LedgerBERT-Market-Sentiment` (Hugging Face) và GitHub `dlt-science/DLT-Corpus`: fine-tune trên DLT-Sentiment-News.
- `ProsusAI/finbert`, `yiyanghkust/finbert-tone`: các mô hình FinBERT quen thuộc (kiến thức chung, chưa kiểm tra lại trong lần tìm này).

**Topic / narrative**

- `iMahdiGhazavi/bertopic-crypto-topic-modeling` (Hugging Face): 50 topic, train trên 4.000 tài liệu. Quá nhỏ và nhiều nhiễu (có topic airdrop, giveaway), chỉ dùng để học cách load; nên tự train BERTopic trên corpus của bạn.
- Một nghiên cứu dùng BERTopic và dynamic topic modeling để theo dõi chủ đề crypto trên LexisNexis, Web of Science và Reddit.

**Dataset**

- CryptoVision (Mendeley): https://data.mendeley.com/datasets/3c3xtxtfb6/2
- CoinDesk News 2020–2025 (Hugging Face): `maryamfakhari/coindesk-crypto-news-2020-2025`
- DLT-Sentiment-News (Hugging Face): `ExponentialScience/DLT-Sentiment-News` (23.301 mẫu, 1/2021–5/2025)
- `Qokka/crypto-sentiment-data` (GitHub): sentiment 2019 từ Reddit và Telegram, yêu cầu ghi nguồn khi công bố.
- `ElKulako/stocktwits-crypto` (Hugging Face): dùng để kiểm tra sentiment model.

**Pipeline và bài báo tham khảo**

- Một bài báo trình bày pipeline tái lập được: thu tin qua GDELT DOC 2.0 API rồi dùng zero-shot NLI để đánh giá chiều tác động của tin lên từng tài sản.
- arXiv 2204.05781: dùng sentiment từ tin tức, Reddit và tweet cho BTC/ETH, so sánh dự báo return có và không có đặc trưng sentiment, kiểm tra đa cộng tuyến bằng VIF. Gần như là thiết kế RQ5 thu nhỏ.
- Frohmann et al. (2023): BERT fine-tune với trọng số theo số follower để dự báo giá Bitcoin trong ngày.
- BitMood (Facebook + FinBERT): sentiment dự báo tốt volume; Linear Regression cho kết quả tương đương BiLSTM, củng cố nhận định không nên lạm dụng deep learning.
- Cảnh báo: nhiều bài mạnh về kết luận nhưng yếu về phương pháp (ví dụ mẫu chỉ khoảng hai tháng nhưng kết luận có Granger causality). Đừng học theo.

**Công cụ Python** (kiến thức chung): `transformers`, `bertopic`, `sentence-transformers`, `statsmodels` (VAR, Granger, HAC), `arch` (GARCH), `lightgbm`, `shap`, `streamlit`.

---

## 10. Lộ trình gợi ý (10–12 tuần)

| Tuần | Công việc |
|---|---|
| 1 | Cổng kiểm tra dữ liệu (bước 1, 3, 4 đã chạy trên CryptoVision v1 ngày 20/10/2026; archive là tuỳ chọn), bật RSS, chốt taxonomy narrative, dựng repo và data dictionary |
| 2–3 | Thu thập và làm sạch (CryptoVision, CoinDesk HF, Binance, Reddit); dedup; source log |
| 4 | Sentiment (lexicon, FinBERT, CryptoBERT) và validation bằng nhãn tay |
| 5 | Narrative (zero-shot, từ khoá, BERTopic đối chiếu); chỉ số ngày |
| 6–7 | Hồi quy HAC, HAR-RV, VAR/Granger, event study, kiểm tra điểm gãy |
| 8 | ML walk-forward, SHAP, kiểm tra chồng lấn mô hình pretrained |
| 9 | Robustness (đổi horizon, đổi mô hình sentiment, sub-sample) và dashboard |
| 10–12 | Viết báo cáo, kiểm tra tái lập, kiểm tra holdout live nếu RSS đã đủ dài |

**Phạm vi tối thiểu nếu thiếu thời gian:** BTC và ETH, một nguồn tin sạch, FinBERT so với lexicon, 4–5 narrative zero-shot, hồi quy HAC, HAR-RV, Granger, Elastic Net và LightGBM, một dashboard đơn giản.

**Cấu trúc repo đề xuất:**

```text
project/
├── README.md
├── requirements.txt
├── config/
├── data/            # raw / interim / processed
├── src/             # data, features, models, evaluation, utils
├── notebooks/
├── dashboard/
├── tests/
├── outputs/         # tables, figures
└── paper/
```

> **Nguyên tắc chung:** một model đơn giản nhưng point-in-time đúng, có benchmark và kiểm định out-of-sample tốt hơn một model phức tạp nhưng leakage hoặc không thể tái lập.

---

## 11. Ghi chú về độ tin cậy

- Số liệu về các nguồn (số bài, ngôn ngữ, giới hạn API, giấy phép) lấy từ tài liệu của chính nguồn và các bài tổng hợp, có thể đã thay đổi. Hãy kiểm tra trước khi dựa vào.
- Các ước lượng sức mạnh thống kê là gần đúng, chỉ để chọn cỡ mẫu.
- Các mục ghi "kiến thức chung" hoặc "chưa kiểm tra" nên được xác minh trước khi đưa vào báo cáo.

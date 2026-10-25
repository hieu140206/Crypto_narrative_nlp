# PREREG: Đăng ký trước giả thuyết và quy trình kiểm định (Topic 10)

Trạng thái: **Đã đóng băng.** Ngày đóng băng: 25-10-2026. Sau ngày này, mọi thay đổi ghi vào mục "Sai khác so với PREREG" ở cuối, không sửa thân văn bản.

Dấu vân tay cấu hình lúc đóng băng (tính lại và so khớp trước khi chạy mô hình):
- `config/narratives.yaml` (v4): sha256 `78b5a05b559666b32599e246f153858abe355720c8f148b86941707302cd99d9`
- `config/params.yaml`: sha256 `f02edc93e3b7210f871804bd818a232db069c170892ee1e7d5cec97a23a83b8a` (đã gồm `own_n_min`)
- Nếu hai hash này đổi sau khi duyệt, phải ghi lý do ở cuối văn bản. Hash trên là của file ngày soạn; tính lại ngay khi đóng băng.

Nguyên tắc: không commit/push; thời điểm đóng băng được chứng minh bằng hash và ngày sửa file.

## 1. Mẫu và chia tách
- Mẫu chính 2021-03-01 → 2025-06-30. 2020–02/2021 chỉ để kiểm tra độ bền.
- Train 2021-03-01→2022-12-31 (671 ngày); validation 2023; **test 2024-01-01→2025-06-30 (547 ngày), chỉ chạm một lần ở cuối.**
- Embargo 7 ngày giữa các tập, đánh cờ theo từng h (`embargo_drop_h`): bỏ dòng trong 7 ngày trước ranh giới và dòng có cửa sổ target vượt sang tập sau. Mẫu dùng được cho baseline HAR (h=1): train 654, valid 356 mỗi tài sản. Horizon h ∈ {1, 3, 7}. Seed 42.
- Cryptonews loại khỏi mẫu chính. Tài sản: BTC (chính), ETH (phụ).
- Ngày đầu tiên có `AbnAttention` hợp lệ: 2021-03-31 (cần 30 ngày nền).

## 2. Biến
- Sentiment: VADER và FinBERT (pin revision) trên `title`; biến thể `nlp_text` (title + description) chỉ là kiểm tra độ bền. `Sent_t = PosSent − NegSent`; `MeanProb_t = mean(p_pos − p_neg)`.
- Attention: `AbnAttention_t = log(1+N_t) − mean(log(1+N_{t−1..t−30}))`.
- Narrative (đối chiếu tên của nhóm: regulation=regulation; etf và institutional_adoption tách làm hai, nhóm gọi chung là etf; hack=hacks_security; defi=defi; macro=monetary_policy; stablecoin=stablecoins; bảng chính của báo cáo dùng đủ 7 tên): `Att_k = n_k/N_t`, `AbnAtt_k` cùng cách tính. Nhóm chính (7): regulation, etf, institutional_adoption, hacks_security, defi, stablecoins, monetary_policy. Exploratory: ai_general, market_failure, nft_meme_web3, mining_energy, politics_geopolitics, tech_upgrade, cbdc.
- Thị trường: `ret` (log close/open), `rv` (tổng bình phương 24 return giờ), `log_rv`, `abvol`; chỉ dùng ngày `complete_day`.
- **Sentiment/attention chính dùng phạm vi toàn thị trường (`all_*`) cho cả BTC và ETH.** Phạm vi riêng từng tài sản (`own_*`, ngưỡng `own_n_min = 5` bài/ngày, dưới ngưỡng thì NaN) chỉ là kiểm tra độ bền, vì mất khoảng 10% ngày trong train.
- Thành phần HAR (log_rv): `har_d`, `har_w` (cửa sổ 5, tối thiểu 4 ngày hợp lệ), `har_m` (cửa sổ 22, tối thiểu 18 ngày hợp lệ). Trung bình tính trên các ngày `complete_day`, không điền khuyết. Spec A dùng cửa sổ kết thúc ở t−1; Spec B kết thúc ở t.
- Biến kiểm soát (join theo `usable_from_utc`): VIXCLS, DGS10, DTWEXBGS, và ret/log_rv trễ.

## 3. Hai loại đặc tả
- **A (đồng thời, mô tả):** tin ngày t với thị trường ngày t, kiểm soát chỉ ở trễ. Chỉ khẳng định về **liên hệ**.
- **B (dự báo):** tin ngày t dự báo thị trường t+h, h ≥ 1. Chỉ B được nói về **giá trị dự báo**.
- Quy tắc milestone của nhóm (bài trước 00:00 UTC ngày t dùng cho ngày t) tương ứng với **Spec B với h = 1** (tin ngày t−1 → thị trường ngày t). Mọi kết luận về giá trị dự báo và về RQ2, RQ4, RQ5 dựa trên B. Spec A (cùng ngày) chỉ là mô tả liên hệ, ghi rõ như vậy trong báo cáo và không dùng để nói về dự báo.

## 4. Giả thuyết xác nhận (confirmatory)
Mỗi giả thuyết kiểm định hai phía, α = 0,05, **sau hiệu chỉnh Holm trong từng họ**. Sai số chuẩn Newey–West với bandwidth = h + 4 (tối thiểu 5), ghi lại giá trị đã dùng.

Họ 1 (RQ1, spec A, BTC, train+validation):
- H1a: R² điều chỉnh của mô hình với FinBERT `Sent_t` cao hơn mô hình với VADER `Sent_t` khi giải thích `ret_t`. So sánh bằng kiểm định Vuong hoặc bootstrap theo khối trên chênh lệch R².
- H1b: tương tự cho `log_rv`.

Họ 2 (RQ2, spec A và B, BTC):
- H2: hệ số của phần âm lớn hơn về độ lớn so với phần dương: `|β_neg| > |β_pos|` trong `ret_{t+h}`, h ∈ {1,3,7}. Kiểm định bằng Wald trên hiệu `β_pos + β_neg`. Hướng: kỳ vọng bất đối xứng, nhưng kiểm định hai phía.

Họ 3 (RQ3, spec A, BTC): với từng narrative chính k, hệ số của `AbnAtt_k` lên (i) `log_rv`, (ii) `abvol`, (iii) `ret`. 7 narrative × 3 biến = 21 phép thử trong **một** họ, hiệu chỉnh Holm. Kết luận về narrative "mạnh nhất" chỉ dựa trên phép thử qua Holm.

Họ 4 (RQ4): Granger lead-lag giữa `Sent_t` và `ret_t` bằng VAR (lag theo BIC). Hai chiều. Tin chỉ mô tả giá (`price_desc = 1`) được loại ở bản chính của RQ4; bản gồm cả hai là kiểm tra độ bền.
- H4: "text dẫn giá" chỉ được khẳng định nếu Granger Sent→ret có ý nghĩa **và** không biến mất khi loại bài `price_desc`.

Họ 5 (RQ5, spec B, **test 2024-01→2025-06**, chạy một lần):
- Mô hình cơ sở: AR/HAR-RV với biến thị trường và VIX/DGS10/DTWEXBGS.
- Mô hình mở rộng: cộng sentiment và/hoặc narrative.
- H5: mô hình mở rộng có sai số dự báo thấp hơn cơ sở. Kiểm định bằng **Clark–West** (lồng nhau), cùng DM-HLN (không lồng nhau, nếu có), cho h = 1, 3, 7. Báo R²_OOS, QLIKE cho `rv`, kèm p-value.
- Mốc hiệu chỉnh: test có khoảng n/h bậc độc lập (h=7 ≈ 78), nên sai số chuẩn lớn; **kết quả không có ý nghĩa là kết quả hợp lệ**, báo cáo như vậy.

### Giao thức đánh giá dự báo (áp dụng cho mọi mô hình họ 5)
- Rolling-origin (expanding window). Dự báo tại cuối ngày t chỉ dùng các dòng huấn luyện có `target_end_date_h ≤ t` và không bị `embargo_drop_h`.
- Baseline = HAR thị trường (`har_d`, `har_w`, `har_m`, hồi quy OLS). Mô hình "text" = baseline + biến sentiment/narrative. Phép so sánh xác nhận của RQ5 là **text so với HAR**, bằng R²_OOS (so với HAR) và Clark–West một phía (lồng nhau). So với hằng số (B0) hoặc `har_m` đơn (B1) chỉ là kiểm tra baseline, không phải bằng chứng về text.
- Kết quả baseline đã chạy trên validation (2023), chưa có text: R²_OOS vs B1 khoảng 0,03–0,10 (BTC) và 0,03–0,06 (ETH). Con số này đặt mốc: mức cải thiện nhỏ của text là kết quả bình thường.

### Mô hình học máy (thuộc họ 5)
- Elastic Net và XGBoost (hoặc LightGBM), mỗi mô hình chạy hai bản: **chỉ biến thị trường** và **thị trường + văn bản** (sentiment + narrative). Phần "text" chỉ được coi là có giá trị nếu bản thị trường + văn bản tốt hơn bản chỉ thị trường trên test, theo Clark–West (Elastic Net, mô hình lồng nhau) hoặc DM-HLN (XGBoost).
- Chọn siêu tham số bằng CV theo thứ tự thời gian (expanding window, embargo 7 ngày) trên train + validation, không xáo trộn, không dùng test. Lưới tham số ghi trong `config/params.yaml` **trước khi chạy**; không mở rộng lưới sau khi thấy kết quả.
- SHAP chỉ là phân tích diễn giải (khám phá), không dùng làm bằng chứng xác nhận.

### Về RQ6
Bảng phân công của nhóm nhắc "RQ6" nhưng đề tài chỉ có RQ1–RQ5. **RQ6 không tồn tại và không được kiểm định.** Báo cáo cần sửa các chỗ nhắc RQ6 thành RQ1–RQ5.

## 5. Phân tích khám phá (không dùng để khẳng định)
7 narrative exploratory, CryptoBERT trên mẫu con, tương tác sentiment × narrative, cửa sổ chế độ (2022 gấu, kỷ nguyên ETF). Báo cáo riêng, gắn nhãn "khám phá", không hiệu chỉnh vào họ xác nhận.

## 6. Độ bền (đăng ký trước)
Biến thể `nlp_text`; loại Decrypt; loại bài `price_desc`; loại ngày dump nguồn; chia mẫu theo năm; ETH; mốc bắt đầu 2020-01-01; thay Newey–West bằng bootstrap theo khối. Báo toàn bộ, kể cả khi trái với kết quả chính.

## 7. Quy tắc dừng và trung thực
- Không chọn mô hình theo kết quả trên test. Chọn siêu tham số chỉ trên train/validation.
- Số phép thử đã chạy được đếm và công bố (kể cả phép thử không báo cáo).
- Kết quả âm tính được báo cáo đầy đủ như kết quả dương.
- Nhãn tay: 630 dòng, AI gán (hai phiên), người phân xử; báo cáo như "LLM-assisted, human-adjudicated". Dùng để chấm VADER/FinBERT, không dùng làm biến huấn luyện.
- RSS holdout (từ 2026): chỉ để kiểm tra pipeline và đánh giá thăm dò, không dùng để kiểm định chính (quá ngắn).

## 8. Sai khác so với PREREG
(để trống, ghi ngày + lý do + tác động)
Sai khác 1 (2026-10-25). Baseline HAR của họ 5 được bổ sung biến giả ngày trong tuần (UTC) của ngày gốc t. Lý do: trong lúc phát triển pipeline trên validation 2023 (chưa chạm test), biến attention (AbnAttention) cho kết quả có ý nghĩa ở h=3 và h=7 (BTC, Holm p = 0,016 và 0,043), nhưng tín hiệu biến mất khi baseline có biến giả ngày trong tuần (p = 0,08 / 0,73 / 0,40). Nguyên nhân: số bài tin giảm khoảng 1,7 đơn vị log vào cuối tuần, trùng với biến động thấp cuối tuần mà HAR thuần bỏ sót. Baseline HAR+DOW tăng R²_OOS 25% ở h=1 và 17% ở h=3. Tác động: mọi so sánh văn bản-so-với-baseline (RQ5) dùng HAR+DOW; kết quả với HAR thuần vẫn báo cáo như độ bền. 18 phép thử trên HAR thuần đã được ghi trong outputs/tables/test_registry.csv.
Sai khác 2 (2026-10-25). Khối ml: được thêm vào config/params.yaml theo đúng yêu cầu của PREREG (lưới Elastic Net và XGBoost khai báo trước khi chạy). Các tham số khác không đổi. Hash mới của params.yaml: cc7ff1f8…6610. Hash cũ trong phần đầu tài liệu (f02edc93…) là trạng thái trước khi thêm khối ml:. Hai file config/narratives.yaml và config/params.yaml được commit lần đầu lúc ghi nhận sai khác này. Từ điển narrative (v4, đã đóng băng từ 2026-10-25) không đổi so với hash đã ghi.
Sai khác 3 (2026-10-25). Ở lần chạy validation 2023 với lưới ML v1 (commit ef7b09f, params.yaml sha256 cc7ff1f8…6610), siêu tham số chọn bởi CV trên tập train nằm ở biên lưới ở cả 12 ô (Elastic Net alpha = 0,1 là giá trị lớn nhất; XGBoost depth 2, n = 200, lr 0,03 là tổ hợp ít phức tạp nhất). Do đó khối ml_v2 được thêm vào config/params.yaml (sha256 4fe733cf…19b3) với lưới mở rộng; features, CV, refit và seed không đổi. Mở rộng này được quyết định sau khi đã xem kết quả validation v1, vì vậy là sai khác post-hoc. Tập test chưa được chạm. Kết quả validation của cả hai lưới đều được báo cáo; 36 phép thử của lưới v1 vẫn nằm trong registry, và mọi phép thử bổ sung được cộng vào tổng số phép thử công bố. Thêm một phân tích độ nhạy macro_mode=diff (bỏ mức của VIX, DGS10, DTWEXBGS, chỉ giữ biến sai phân 5 ngày), gắn nhãn thăm dò.
Sai khác 4 (2026-10-25). Cấu hình ML dùng cho lần chạy duy nhất trên tập test là: lưới v2 (ml_v2, sha256 params.yaml 4fe733cf…19b3), macro_mode = level, hai mô hình học (Elastic Net, XGBoost), bản market và bản market+text, h ∈ {1,3,7}. Lý do chọn v2 thay v1: lưới v1 chọn ở biên ở cả 12 ô trên CV tập train (xem Sai khác 3). Trên validation 2023, không phép thử nào của họ 5-ML đạt p Holm < 0,05 ở cả ba lượt (v1, v2 level, v2 diff); p Holm nhỏ nhất của v2 level là 0,50. Kết quả v1 và macro_mode = diff chỉ báo cáo ở validation, không chạy lại trên test. Tổng số phép thử text-vs-market trong họ ML đã thực hiện trên validation là 36 (3 lượt × 12); lượt chạy test sẽ thêm 12.
## 9. Kết quả xác nhận trên tập test (chạy một lần, 25-10-2026)

Quy trình: `ALLOW_TEST=1 python -m models.run_test_once`, chạy đúng một lần, khóa `outputs/TEST_LOCK` ghi trước khi đọc dữ liệu test (trạng thái `completed`). Commit mã `fb6e3a6`; HEAD lúc chạy `82ee530`; `params.yaml` sha256 `4fe733cf…19b3`; seed 42; lưới ML v2, `macro_mode = level` (Sai khác 3 và 4). Tập test 2024-01-01 → 2025-06-30, n_oos = 547 cho mọi ô. Kết quả đầy đủ: `outputs/test/`; registry `outputs/tables/test_registry.csv` (433 dòng sau lần chạy). Mọi con số dưới đây lấy nguyên từ file kết quả, không chỉnh sửa.

### 9.1 Họ 5, văn bản so với HAR+DOW (CW một phía, Holm 9 phép mỗi tài sản)

| Tài sản | h | Mô hình | R²_OOS so với HAR+DOW | p thô | p Holm |
|---|---|---|---|---|---|
| BTC | 1 | AbnAttention | 0,0070 | 0,031 | 0,279 |
| BTC | 1 | FinBERT (Sent) | -0,0006 | 0,692 | 1,000 |
| BTC | 1 | VADER (Sent) | 0,0000 | 0,455 | 1,000 |
| BTC | 3 | AbnAttention | 0,0005 | 0,324 | 1,000 |
| BTC | 3 | FinBERT (Sent) | 0,0053 | 0,064 | 0,509 |
| BTC | 3 | VADER (Sent) | -0,0010 | 0,827 | 1,000 |
| BTC | 7 | AbnAttention | -0,0051 | 0,684 | 1,000 |
| BTC | 7 | FinBERT (Sent) | 0,0032 | 0,155 | 1,000 |
| BTC | 7 | VADER (Sent) | -0,0016 | 0,803 | 1,000 |
| ETH | 1 | AbnAttention | 0,0045 | 0,047 | 0,426 |
| ETH | 1 | FinBERT (Sent) | -0,0011 | 0,542 | 1,000 |
| ETH | 1 | VADER (Sent) | -0,0006 | 0,630 | 1,000 |
| ETH | 3 | AbnAttention | -0,0008 | 0,654 | 1,000 |
| ETH | 3 | FinBERT (Sent) | -0,0001 | 0,303 | 1,000 |
| ETH | 3 | VADER (Sent) | -0,0001 | 0,473 | 1,000 |
| ETH | 7 | AbnAttention | -0,0059 | 0,641 | 1,000 |
| ETH | 7 | FinBERT (Sent) | -0,0010 | 0,547 | 1,000 |
| ETH | 7 | VADER (Sent) | -0,0017 | 0,801 | 1,000 |

Kết quả: 0/18 phép thử có p Holm < 0,05; p Holm nhỏ nhất là 0,279 (BTC, h=1, AbnAttention). Không có bằng chứng rằng các đặc trưng văn bản bổ sung dự báo log biến động thực hiện ngoài HAR+DOW.

### 9.2 Họ 5-ML, market+text so với market (Clark–West cho Elastic Net, DM-HLN cho XGBoost; Holm 6 phép mỗi tài sản)

| Tài sản | h | Mô hình | R²_OOS (text so với market) | p thô | p Holm | R²_OOS bản market so với HAR+DOW | R²_OOS bản text so với HAR+DOW |
|---|---|---|---|---|---|---|---|
| BTC | 1 | Elastic Net | 0,0000 | 0,281 | 1,000 | -0,0263 | -0,0262 |
| BTC | 1 | XGBoost | 0,0098 | 0,212 | 1,000 | 0,0100 | 0,0197 |
| BTC | 3 | Elastic Net | -0,0010 | 0,832 | 1,000 | -0,0631 | -0,0641 |
| BTC | 3 | XGBoost | 0,0585 | 4,3e-06 | 2,6e-05 | -0,1358 | -0,0693 |
| BTC | 7 | Elastic Net | 0,0001 | 0,408 | 1,000 | -0,0400 | -0,0399 |
| BTC | 7 | XGBoost | -0,0011 | 0,555 | 1,000 | -0,0573 | -0,0585 |
| ETH | 1 | Elastic Net | -0,0348 | 0,963 | 1,000 | 0,0056 | -0,0290 |
| ETH | 1 | XGBoost | 0,0323 | 0,012 | 0,074 | 0,0274 | 0,0588 |
| ETH | 3 | Elastic Net | 0,0017 | 0,110 | 0,551 | -0,0021 | -0,0003 |
| ETH | 3 | XGBoost | 0,0011 | 0,456 | 1,000 | 0,0168 | 0,0179 |
| ETH | 7 | Elastic Net | -0,0006 | 0,221 | 0,885 | -0,0318 | -0,0324 |
| ETH | 7 | XGBoost | -0,0035 | 0,955 | 1,000 | 0,0134 | 0,0099 |

Kết quả: 1/12 phép thử có p Holm < 0,05: **BTC, h=3, XGBoost** (R²_OOS +5,85%, p thô 4,3×10⁻⁶, p Holm 2,6×10⁻⁵). 11/12 phép thử không có ý nghĩa; ETH h=1 XGBoost có p Holm 0,074.

### 9.3 Giải thích và giới hạn (phần dưới đây là phân tích hậu nghiệm trên file OOS đã lưu, không phải kiểm định đăng ký trước)

- Ở ô BTC h=3 XGBoost, bản text tốt hơn bản market ở cả 6 quý của tập test; 10 ngày có chênh lệch bình phương sai số lớn nhất chiếm 12% tổng chênh lệch tuyệt đối. Chiều của hiệu ứng khớp với validation (+2,3%, p thô 0,32).
- Hiệu ứng **không** xuất hiện ở các mô hình tuyến tính cùng ô: Elastic Net (p thô 0,83) và HAR+text (p thô 0,32).
- Bản XGBoost market là baseline yếu: R²_OOS so với HAR+DOW là −13,6%. Bản text vẫn kém HAR+DOW (−6,9%; chênh lệch không có ý nghĩa, DM = 1,53). Do đó kết quả nói rằng đặc trưng văn bản cải thiện một mô hình ML thị trường yếu ở một ô, không nói rằng văn bản giúp vượt HAR+DOW.
- Tổng số phép thử text-vs-market của họ ML đã thực hiện là 48 (36 trên validation, 12 trên test). p của ô trên vẫn rất nhỏ sau khi chỉnh theo 48 phép thử (≈ 2×10⁻⁴), nên kết luận không phụ thuộc vào cách đếm. Tuy vậy đây là 1 ô trên 12, chỉ ở một mô hình học và một tài sản; báo cáo trình bày nó cùng với 11 ô không có ý nghĩa.
- Họ 1 đến 4 là phân tích mô tả đồng thời, ước lượng trên train+valid; chúng không được ước lượng lại trên tập test. Tập test đã được dùng một lần và không được dùng lại để chọn mô hình.

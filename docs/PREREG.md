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

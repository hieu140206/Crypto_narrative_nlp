# Ghi chú viết báo cáo (TV3/TV4, Topic 10)

Nguồn số: `outputs/tables/` (train+valid), `outputs/test/` (test, chạy một lần), `outputs/tables/test_registry.csv` (457 dòng: 96 dòng ghi CV siêu tham số + 361 dòng phép thử). Mọi con số lấy từ file kết quả; không làm tròn thêm khi trích.

## 1. Thông điệp chính (theo thứ tự nên trình bày)

1. **Văn bản không bổ sung dự báo biến động ngoài HAR+DOW.** Họ 5 trên test: 0/18 phép thử, p Holm nhỏ nhất 0,279. Trên validation: null ở HAR+text và ở cả ba lượt ML (v1, v2 level, v2 diff).
2. **Một kết quả dương, cần trình bày cùng 11 ô null:** XGBoost, BTC, h=3, bản market+text so với market: R²_OOS +5,85%, p thô 4,3×10⁻⁶, p Holm 2,6×10⁻⁵; text tốt hơn ở cả 6 quý. Nhưng bản text vẫn kém HAR+DOW (−6,9%, DM 1,53, không có ý nghĩa), không tái hiện ở Elastic Net hay HAR+text, và không có ở ETH. Cách nói đúng: "đặc trưng văn bản cải thiện một mô hình ML thị trường yếu ở một ô", không phải "văn bản giúp dự báo biến động".
3. **FinBERT có liên hệ đồng thời với lợi suất, VADER và CryptoBERT thì không** (H1a xác nhận; ΔR²adj đầy đủ BTC 0,079, ETH 0,068 so với VADER 0,004, 0,002). Thứ tự này trùng với chất lượng đo theo nhãn (accuracy 62,4% > 49,0% > 46,7%; κ 0,36 > 0,15 > 0,005). Kèm cảnh báo tiếng vọng giá: khoảng 60% lợi thế đồng thời của FinBERT là tiêu đề mô tả biến động giá (price_desc ≈ 30% bài).
4. **Hướng nhân quả đi từ giá sang văn bản:** BTC ret→Sent p Holm 0,0035, Sent→ret 0,473; ETH ret→Sent p Holm 0,057 (biên, không xác nhận).
5. **Không có narrative nào được xác nhận** (RQ3: 0/21 sau Holm cho cả BTC và ETH); H2 (bất đối xứng pos/neg) null.

## 2. Việc phải nói thẳng trong báo cáo

- **Nhãn:** 630 tiêu đề do hai phiên AI gán (không phải người), κ giữa hai bên 0,54; 197 dòng bất đồng do Nam phân xử; 100 dòng độc lập do Nam gán (κ AI–người 0,727; AI thiên về trung tính). Gọi là "LLM-assisted, human-adjudicated".
- **Số phép thử:** công bố 361 dòng phép thử trong registry (ML: 36 validation + 12 test text-vs-market). Báo cáo cả kết quả null.
- **Sai khác so với PREREG:** 4 mục trong PREREG mục 8 (HAR+DOW; khối ml; lưới v2 sau khi thấy chọn ở biên; cấu hình ML cho test) và kết quả test ở mục 9. Lưới v2 và `macro_mode=diff` được quyết định sau khi xem validation v1: post-hoc.
- **Test chỉ dùng một lần** (khóa `outputs/TEST_LOCK`, commit `82ee530`); các phân tích sau đó trên file OOS test là hậu nghiệm.
- **Họ 1–4 là mô tả đồng thời trên train+valid**, không ước lượng lại trên test.
- **Tiếng vọng giá và hiệu ứng ngày trong tuần:** số bài giảm ~1,7 đơn vị log vào cuối tuần, AbnAttention từng bắt chước biến giả cuối tuần; baseline HAR phải có DOW.

## 3. Độ bền và giới hạn (đã làm)

- **CryptoBERT (thăm dò, mẫu con 25% = 10.007 bài, 2021-03→2023-12):** κ với nhãn 0,005; tương quan chuỗi ngày với FinBERT 0,10; ΔR²adj trên lợi suất ≈ 0 (BTC 0,0004, ETH 0,0018); dự báo HAR+DOW null. Kết luận: mô hình huấn luyện trên mạng xã hội không phù hợp tiêu đề báo; không dùng làm kiểm định thay thế. Mẫu con làm FinBERT yếu đi 4–5 lần (BTC 0,079 → 0,018), cho thấy độ phủ bài mỗi ngày quyết định sức mạnh.
- **Nhân bản chéo nguồn (cryptonews, mô tả):** Sent_FinBERT tương quan khoảng 0,20 giữa cryptonews và bốn nguồn chính (CI 95% 0,03–0,36 giai đoạn 2023H2; 0,11–0,29 giai đoạn 2024–2025H1), giảm còn khoảng 0,10 khi loại bài mô tả giá; VADER 0,15 (2024–2025H1). **AbnAttention thô tương quan cao (0,49 và 0,65) nhưng sau khi bỏ hiệu ứng ngày trong tuần chỉ còn 0,26 (2023H2) và −0,01 (2024–2025H1)**: sự đồng thuận chủ yếu là chu kỳ tuần; các cú sốc chú ý riêng lẻ không đồng pha giữa nguồn. Cryptonews trung vị 7 bài mỗi ngày, 20% ngày dưới 3 bài, nên tương quan bị nhiễu đo lường kéo xuống.
- **RSS holdout:** chỉ kiểm tra vận hành. 256 → 213 bài (bỏ 20 backfill, 23 off-topic), 0 lỗi, hash narratives khớp; tỉ lệ nhãn nằm trong khoảng tin cậy của mẫu chính, riêng narrative stablecoins khớp 11,3% (CI 7,7–16,2%) so với 4,5% của mẫu chính (4 ngày, không kết luận được). Không dùng để kiểm định.
- **Giới hạn khác:** ML lưới v2 vẫn chọn depth=1/mcw=20 ở biên cho nhiều ô XGBoost (depth 1 là sàn logic); XGBoost dùng mức của DGS10/DTWEXBGS (không dừng) nên kém HAR+DOW ở hầu hết ô; QLIKE không nhất quán với RMSE ở vài ô (chỉ mô tả).

## 4. Việc còn lại (trước khi nộp)

- Commit các file chưa theo dõi của đợt CryptoBERT/RSS-cryptonews và `docs/PREREG.md`, `docs/REPORT_NOTES.md`.
- Nhờ TV6 sửa "RQ1–RQ6" thành RQ1–RQ5.
- Nếu còn thời gian: loại lần lượt từng nguồn (coindesk, cointelegraph, decrypt, blockworks) và chạy lại HAR+text trên validation; tái hiện độc lập một ô ML.


## PHẦN B. Chuẩn bị thư mục làm việc

### B1. Tạo thư mục (đường dẫn KHÔNG dấu, KHÔNG khoảng trắng)

1. Tải file **`tv1_code.zip`** về, chép vào `D:\du_an`.
2. **Chuột phải** vào `tv1_code.zip` → chọn **Extract All...** → bấm **Extract**.
3. Bạn sẽ có thư mục: `D:\du_an\tv1_code\tv1`. Mở vào trong, bạn thấy các thư mục: `config`, `data`, `logs`, `src` và file `requirements.txt`.

> Từ đây, **thư mục `tv1` này là thư mục gốc của dự án**.

### B3. Bỏ 2 file dữ liệu đã tải vào đúng chỗ

Đi tới: `D:\du_an\tv1_code\tv1\data\raw\_download\` (thư mục này đang trống).

Copy 2 file đã tải vào đây:

| File bạn đã tải | Bỏ vào `_download` và đổi tên thành |
|---|---|
| `CryptoVision_A_Comprehensive_Dataset_for_Crypto_Ne.zip` | **`CryptoVision.zip`** |
| `binance_.zip` | giữ nguyên **`binance_.zip`** |

**Cách đổi tên:** chuột phải vào file → **Rename** (hoặc bấm F2) → gõ tên mới → Enter.

> ⚠️ **KHÔNG giải nén 2 file này.** Code tự đọc thẳng từ file zip.
>
> Nếu đổi tên xong mà Windows hiện thành `CryptoVision.zip.zip`: vào tab **View** của File Explorer → tick **File name extensions** để thấy đuôi file, rồi đổi lại cho đúng.

Kết quả đúng:
```
tv1\data\raw\_download\CryptoVision.zip
tv1\data\raw\_download\binance_.zip
```

---

## PHẦN C. Mở terminal đúng chỗ

### C1. Mở dự án bằng VS Code

1. Mở **VS Code**.
2. Menu trên cùng: **File → Open Folder...**
3. Chọn thư mục **`tv1`** (cái có chứa `config`, `data`, `src`) → bấm **Select Folder**.
4. Nếu hiện hộp thoại "Do you trust the authors...", bấm **Yes, I trust the authors**.

Bên trái bạn sẽ thấy cây thư mục `config`, `data`, `logs`, `src`.

### C2. Mở terminal

Trên menu VS Code: **Terminal → New Terminal** (hoặc bấm phím tắt **Ctrl + `**, phím nằm dưới phím Esc).

Một khung đen/xanh hiện ra ở nửa dưới màn hình. Dòng cuối trông như:

```
PS D:\du_an\tv1_code\tv1>
```

Dòng này nghĩa là: **terminal đang đứng ở thư mục `tv1`**. Mở từ VS Code thì nó tự đứng đúng chỗ, bạn khỏi phải tìm đường.

### C3. Kiểm tra Python đã cài chưa

Gõ lệnh (rồi Enter):

```
python --version
```

- ✅ Thấy `Python 3.11.x` (hoặc 3.12.x...) → tốt, sang bước D.
- ❌ Thấy `python is not recognized...` hoặc nó mở Microsoft Store → bạn quên tick "Add to PATH" ở bước A1. Cách xử lý: **gỡ Python (Settings → Apps)**, cài lại và nhớ tick. Sau khi cài lại, **đóng VS Code mở lại**, rồi thử lại.
- Thử thêm: nếu `python` không chạy được thì gõ `py --version`. Nếu `py` chạy được thì **từ giờ mọi lệnh bên dưới thay chữ `python` bằng `py`** và `pip` bằng `py -m pip`.

---

## PHẦN D. Cài thư viện

Vẫn trong terminal, gõ:

```
pip install -r requirements.txt
```

Bạn sẽ thấy chữ chạy nhiều dòng (Collecting pandas..., Downloading..., Installing...). Chờ 1–3 phút cho tới khi **dòng cuối** là:

```
Successfully installed ...
```

(Nếu nó báo `Requirement already satisfied` thì cũng ổn, tức là đã có sẵn.)

Nếu báo `pip is not recognized`: gõ `python -m pip install -r requirements.txt`.

---

## PHẦN E. Chạy 3 lệnh chính

### E1. Đi vào thư mục chứa code

Gõ:

```
cd src\data
```

Dòng terminal sẽ đổi thành:
```
PS D:\du_an\tv1_code\tv1\src\data>
```

> `cd` = "change directory" = đi vào thư mục. Bắt buộc phải làm bước này, nếu không sẽ báo lỗi `No such file`. Nếu lỡ đi sai chỗ, gõ `cd ..` để lùi ra 1 cấp.

### E2. Bước 1: Xử lý CryptoVision

```
python ingest_cryptovision.py
```

Chờ khoảng 10–30 giây. **Kết quả đúng sẽ giống thế này:**

```
[1] Đọc xong: 188,431 dòng, 14 cột
[2] Đã drop 10 cột leakage. Còn lại: ['url', 'title', 'date_time', 'coin_type']
[3] Dòng hỏng/thiếu giá trị bị loại: 298
[4] Bỏ trùng (url, coin_type): 1,112 dòng
[5] CryptoVision: min=2017-08-17 15:00:33+00:00, MAX=2025-08-27 17:47:34+00:00, rows=187,021, unique_url=119,336
[6] Đã lưu data\raw\cryptovision\cryptovision_clean.parquet
```

Nếu số của bạn giống (188,431 → 187,021, Max = 2025-08-27) thì bạn làm đúng 100%.

### E3. Bước 2: Xử lý Binance

```
python ingest_binance.py
```

Chờ khoảng 20–60 giây. **Kết quả đúng:**

```
[BTCUSDT] 72 file tháng
   52,577 nến | 2020-01-01 00:00:00+00:00 -> 2025-12-31 23:00:00+00:00 | lưu data\raw\binance\BTCUSDT_1h.parquet
[ETHUSDT] 72 file tháng
   52,577 nến | 2020-01-01 00:00:00+00:00 -> 2025-12-31 23:00:00+00:00 | lưu data\raw\binance\ETHUSDT_1h.parquet
```

### E4. Bước 3: Kiểm tra chất lượng

```
python verify_data.py
```

**Kết quả đúng:**

```
OK  Milestone 1: không còn cột leakage; cột = ['url', 'title', 'date_time', 'coin_type']
    Timezone: UTC | Max date: 2025-08-27 17:47:34+00:00
WARN CryptoVision chỉ tới 2025-08-27 (< 31/12/2025) -> báo TV2 lấp bằng Free Crypto News, ghi vào Limitations
OK  Milestone 2 BTCUSDT: 2020-01-01 00:00:00+00:00 -> 2025-12-31 23:00:00+00:00
    Số nến: 52,577 / kỳ vọng 52,608 | thiếu 31 nến
    ...
OK  Milestone 2 ETHUSDT: ...
Xong. Không có FAIL -> bàn giao data/raw/ cho nhóm.
```

- Dòng **`WARN`** là **bình thường**, không phải lỗi. Nó nhắc bạn là CryptoVision thiếu cuối 2025.
- Chỉ cần lo nếu thấy chữ **`FAIL`** hoặc `AssertionError`.

🎉 **Xong phần code của TV1.**

---

## PHẦN F. Xem kết quả và lấy file bàn giao

### F1. File nằm ở đâu?

Trong cây thư mục bên trái VS Code (hoặc File Explorer):

| File | Đường dẫn trong `tv1` |
|---|---|
| Tin tức đã làm sạch | `data\raw\cryptovision\cryptovision_clean.parquet` |
| Giá BTC 1h | `data\raw\binance\BTCUSDT_1h.parquet` |
| Giá ETH 1h | `data\raw\binance\ETHUSDT_1h.parquet` |
| Log Max date | `logs\max_date_log.txt` |
| Source log | `logs\source_log.csv` |

> Nếu thay vì `.parquet` bạn thấy đuôi **`.csv.gz`**: nghĩa là chưa cài được `pyarrow`. Gõ `pip install pyarrow` rồi chạy lại 2 lệnh ở E2 và E3 là được.

### F2. Muốn xem thử nội dung (tùy chọn)

Quay lại thư mục gốc rồi xem 5 dòng đầu của tin tức. Gõ lần lượt:

```
cd ..\..
python -c "import pandas as pd; print(pd.read_parquet('data/raw/cryptovision/cryptovision_clean.parquet').head())"
```

Xem giá BTC:

```
python -c "import pandas as pd; print(pd.read_parquet('data/raw/binance/BTCUSDT_1h.parquet').head())"
```

(`cd ..\..` = lùi ra 2 cấp để về thư mục `tv1`.)

`logs\max_date_log.txt` và `logs\source_log.csv` mở bằng VS Code hoặc Excel xem được luôn.

### F3. Gửi cho nhóm

Nén 5 file ở bảng F1 (hoặc cả thư mục `data\raw` + `logs`) rồi gửi lên Drive / nhóm chat. Nếu nhóm dùng GitHub thì làm theo Phần G.

Tin nhắn mẫu gửi nhóm:

> TV1 xong D1: CryptoVision còn 187,021 dòng (đã drop 10 cột leakage), Max date = 27/08/2025 nên thiếu 28/08–31/12/2025, nhờ TV2 lấp bằng Free Crypto News. Binance BTC/ETH 1h đủ 2020–2025 (52,577 nến/coin, thiếu 31 nến do Binance bảo trì), đã xử lý ms/us. File ở data/raw/, log ở logs/.

---

## PHẦN G. Đưa code lên GitHub (khi nhóm đã có repo)

Làm phần này sau khi TV5 tạo repo và mời bạn. Nếu chưa có, **bỏ qua**, để D2 làm cũng được.

### G1. Cài Git (1 lần)
Vào **https://git-scm.com/download/win**, tải về, bấm Next liên tục tới khi xong. Đóng VS Code và mở lại.

### G2. Khai báo tên (1 lần, thay bằng tên và email thật của bạn)

```
git config --global user.name "Duc Anh"
git config --global user.email "email_cua_ban@gmail.com"
```

### G3. Tải repo của nhóm về (TV5 gửi link, dạng `https://github.com/.../....git`)

Mở **một thư mục khác** (ví dụ `D:\du_an`) trong terminal, gõ:

```
cd D:\du_an
git clone <dán link repo vào đây>
```

Lần đầu nó sẽ hiện cửa sổ đăng nhập GitHub, đăng nhập là xong. Sẽ có thư mục repo mới, hãy mở nó bằng VS Code (File → Open Folder).

### G4. Chép code của mình vào repo

Theo `QUY_UOC_CODE.md` của TV5, chép:

- `tv1\src\data\*.py` (4 file `.py`) → vào `src\data\` của repo
- `tv1\config\paths.yaml` → vào `config\` của repo (nếu TV5 đã có `paths.yaml` thì **gộp** thêm các dòng của mình, đừng ghi đè)
- `tv1\requirements.txt` → gộp vào `requirements.txt` của repo

**Không chép** thư mục `data` và `logs` vào git (nhóm đã cấu hình bỏ qua).

### G5. Tạo nhánh riêng, commit, push

Trong terminal (đang đứng ở thư mục repo):

```
git checkout -b feat/tv1
git add src config requirements.txt
git commit -m "data: add ingest_cryptovision and ingest_binance"
git push origin feat/tv1
```

Giải thích: `checkout -b` = tạo nhánh riêng của bạn; `add` = chọn file; `commit` = đóng gói thay đổi kèm ghi chú; `push` = đẩy lên mạng.

Sau đó báo TV5 để merge vào `main`.

---

## PHẦN H. Bảng lỗi thường gặp

| Bạn thấy | Nghĩa là gì | Cách xử lý |
|---|---|---|
| `python is not recognized` | Chưa tick "Add to PATH" | Thử `py --version`; nếu không được thì cài lại Python và tick ô PATH |
| `pip is not recognized` | Như trên | Dùng `python -m pip install -r requirements.txt` |
| `can't open file '...ingest_cryptovision.py'` | Terminal đang đứng sai thư mục | Gõ `cd src\data` (nếu đang ở `tv1`). Dùng `dir` để xem trong thư mục hiện tại có file nào |
| `FileNotFoundError: ... CryptoVision.zip` | Sai chỗ hoặc sai tên file zip | Kiểm tra lại bước B3, nhất là lỗi `.zip.zip` |
| `ModuleNotFoundError: No module named 'pandas'` (hoặc `yaml`) | Chưa chạy bước D, hoặc chạy xong rồi nhưng đang dùng Python khác | Chạy lại `pip install -r requirements.txt` |
| Chữ `.csv.gz` thay vì `.parquet` | Chưa có pyarrow | `pip install pyarrow` rồi chạy lại E2, E3 |
| `DtypeWarning` | Cảnh báo vàng/đỏ, file gốc có vài dòng vỡ | Bỏ qua được |
| Thấy `FAIL` / `AssertionError` | Dữ liệu có vấn đề thật | Chụp màn hình lỗi, gửi cho mình |

**Mẹo cứu hộ:** nếu bí, **chụp màn hình toàn bộ cửa sổ terminal** gửi cho mình, mình sẽ chỉ chính xác chỗ sai.

---

## Ghi chú cho Mac

- Cài Python từ python.org (hoặc `brew install python`). Trên Mac, dùng **`python3`** thay `python`, và **`pip3`** thay `pip`.
- Đường dẫn dùng dấu **`/`** thay vì `\`. Ví dụ `cd src/data`, `cd ../..`.
- Mở terminal trong VS Code bằng **Terminal → New Terminal**, cách làm các bước còn lại giống hệt.

---

## Checklist cuối cùng (tick hết là xong D1)

- [ ] Cài xong Python (có tick PATH) và VS Code
- [ ] Giải nén `tv1_code.zip`, bỏ 2 file zip dữ liệu vào `data\raw\_download\` đúng tên
- [ ] `pip install -r requirements.txt` báo thành công
- [ ] Chạy `ingest_cryptovision.py`: ra số 187,021 và Max date 2025-08-27
- [ ] Chạy `ingest_binance.py`: ra 52,577 nến mỗi coin
- [ ] Chạy `verify_data.py`: không có `FAIL`
- [ ] Gửi file + log cho nhóm, nhắn TV2 về Max date
- [ ] (Khi có repo) push code lên nhánh `feat/tv1`

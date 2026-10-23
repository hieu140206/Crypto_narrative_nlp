"""clean_merge.py - Làm sạch CryptoVision v1 thành text_clean.parquet (TV2).

Mục đích
    Đọc file CryptoVision (Mendeley) và tạo MỘT bảng văn bản sạch: mỗi dòng là một
    bài, không trùng, giờ UTC, có cột use_date cho biết bài được dùng cho ngày nào.
    Đầu ra giao cho TV3 chạy sentiment. Khung mẫu và quy tắc theo file kế hoạch
    của nhóm trưởng (Excel): cửa sổ 01/01/2024 -> 31/12/2025, train 2024-01 ->
    2025-06, test 2025-07 -> 2025-12.

Cách chạy (cần: pip install pandas pyarrow rapidfuzz). Không có rapidfuzz vẫn chạy
được nhưng bước so tiêu đề gần giống chậm hơn nhiều (có thể hàng chục phút):
    python clean_merge.py                          # dùng đường dẫn mặc định
    python clean_merge.py duong_dan_file.csv       # chỉ định file CryptoVision

Quy định của nhóm:
    - KHÔNG đưa RSS holdout vào file này (RSS có clean_rss.py riêng).
    - Không import vào master_pipeline.ipynb.
    - Đường dẫn lấy từ config/paths.yaml (khóa: cryptovision_path, text_clean_path);
      thiếu thì dùng mặc định bên dưới. Seed 42.
    - Không bao giờ đọc 10 cột rò rỉ (Open, High, Low, Close, Volume,
      Movement_OpenClose_%, Movement_HighLow_%, Market_Move, sentiment_label,
      sentiment_score): script chỉ nạp các cột cần dùng và kiểm tra lại ở cuối
      (milestone 1 của file Excel).

Các bước (theo thứ tự):
    1. Đọc theo từng khối, chỉ lấy cột cần dùng, đổi giờ về UTC, ghi Max date,
       cắt theo khung mẫu.
    2. Lọc coin BTC/ETH (theo cột Coin Type).
    3. Gộp tag asset theo URL (mỗi bài bị nhân bản theo coin), giữ bản sớm nhất.
    4. Gắn cờ và lọc tiếng Anh (tỷ lệ ký tự ASCII), bỏ thông cáo báo chí.
    5. Dedup lớp 2: tiêu đề gần giống trong cửa sổ 48 giờ (không gộp nhầm các
       tiêu đề chỉ khác con số, ví dụ "Price analysis 5/12" và "5/19").
    6. Gắn cờ timestamp (ts_regime_risk, ts_bulk_flag, ts_minute_only).
    7. Tạo nlp_text, doc_id, use_date theo quy tắc chống look-ahead; xuất parquet.

Quy tắc gán ngày (look-ahead):
    Bài đăng trước 00:00 UTC của ngày t thì dùng cho ngày t; bài đăng từ 00:00 UTC
    trở đi trong ngày t thì dùng cho ngày t+1. Nghĩa là use_date = (ngày UTC của
    published_at_utc) + 1 ngày.
"""
import difflib
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

random.seed(42)

# ---- Tham số (nhóm trưởng chốt; có thể chuyển sang config/params.yaml) ------
# Khung mẫu theo file Excel của nhóm trưởng: 01/01/2024 -> 31/12/2025.
# CryptoVision chỉ có dữ liệu thật đến cuối 08/2025, nên theo dòng rủi ro trong Excel
# ("cắt mẫu ở ngày cuối cùng có dữ liệu thật, ghi rõ trong Limitations") T_END đặt
# ở 2025-08-17: đây là tuần đầy đủ cuối cùng theo hướng dẫn nguồn dữ liệu (tuần
# 18-24/08 chỉ có 41 bài, 25-27/08 chỉ có 12 bài). Muốn giữ đến hết dữ liệu thì
# đổi thành "2025-12-31"; bộ lọc tự dừng ở ngày cuối có trong file (Max date).
SAMPLE_START = "2024-01-01"
T_END = "2025-08-17"                 # tính hết ngày này (đến 23:59:59 UTC)
ROBUSTNESS_START = None              # ví dụ "2021-01-01" để giữ thêm các năm trước làm robustness
COIN_KEEP = {"bitcoin", "ethereum", "btc", "eth"}  # so khớp chữ thường, cột Coin Type
ASCII_MIN_RATIO = 0.90               # dưới mức này gắn non_en
DROP_NON_EN = True
DROP_PRESS_RELEASE = True            # bỏ bài có /press-releases/ trong URL
DEDUP_WINDOW_HOURS = 48
TITLE_SIMILARITY = 0.92
BULK_TS_MIN_ROWS = 10                # từ ngần này bài trở lên cùng giây/cùng nguồn = nạp hàng loạt
MINUTE_ONLY_SOURCES = {"coindesk", "cointelegraph"}
REGIME_RISK_SOURCES = {"cryptopanic", "coindesk"}   # trước 2021 timestamp có thể lệch
CHUNK_ROWS = 50_000

DEFAULT_CSV_CANDIDATES = ["data/raw/cryptovision/cryptovision_v1.csv", "cryptovision_v1.csv"]
DEFAULT_OUT = "data/interim/text_clean.parquet"

# Chỉ nạp các cột này. Các cột rò rỉ không bao giờ được đọc.
# Excel yêu cầu tối thiểu URL, Title, Date_Time, Coin_Type. Giữ thêm Description và
# Full Text (thành nlp_text_body) để TV3 tự chọn dùng tiêu đề hay cả bài.
USE_COLS = ["URL", "Title", "Description", "Full Text", "Date Time", "Coin Type"]
LEAK_COLS = {"Open", "High", "Low", "Close", "Volume", "Movement_OpenClose_%",
             "Movement_HighLow_%", "Market_Move", "sentiment_label", "sentiment_score",
             "Sentiment_Label"}


def project_root():
    """Thư mục gốc dự án: gần nhất phía trên có config/; không có thì dùng thư mục file này."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_paths(root, cli_csv=None):
    """Xác định file vào và file ra: tham số dòng lệnh > paths.yaml > mặc định."""
    csv_rel, out_rel = None, DEFAULT_OUT
    cfg = root / "config" / "paths.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            csv_rel = data.get("cryptovision_path")
            out_rel = data.get("text_clean_path", out_rel)
        except Exception:
            pass

    def absolute(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    if cli_csv:
        csv_path = Path(cli_csv)
    else:
        candidates = ([csv_rel] if csv_rel else []) + DEFAULT_CSV_CANDIDATES
        csv_path = next((absolute(c) for c in candidates if absolute(c).exists()), None)
    if csv_path is None or not Path(csv_path).exists():
        sys.exit("Không tìm thấy file CryptoVision. Chạy: python clean_merge.py <đường_dẫn_file.csv>")
    return Path(csv_path), absolute(out_rel)


def normalize_url(url):
    """Chuẩn hóa URL: chữ thường, bỏ tham số theo dõi (utm_...), bỏ #, bỏ dấu / cuối."""
    url = (url or "").strip().lower().split("#")[0]
    if "?" in url:
        base, query = url.split("?", 1)
        keep = [q for q in query.split("&") if q and not q.startswith(("utm_", "ref=", "fbclid"))]
        url = base + ("?" + "&".join(keep) if keep else "")
    return url.rstrip("/")


def source_of(url):
    """Lấy tên nguồn từ tên miền, ví dụ cryptopanic.com -> cryptopanic."""
    host = urlparse(url or "").netloc.lower().replace("www.", "")
    return host.split(".")[0] if host else "unknown"


def normalize_title(title):
    t = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def clean_text_cell(value):
    """Ô trống hoặc chữ 'nan' (do file lưu giá trị thiếu) -> chuỗi rỗng."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() == "nan" else s


def ascii_ratio(text):
    return sum(1 for c in text if ord(c) < 128) / len(text) if text else 1.0


def parse_utc(series):
    """Chuyển cột giờ sang datetime UTC (chấp nhận nhiều dạng ghi)."""
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # pandas cũ không có format="mixed"
        return pd.to_datetime(series, utc=True, errors="coerce")


def step_read(csv_path, log):
    """Bước 1 và 2: đọc theo khối, đổi UTC, cắt khung mẫu, lọc coin."""
    start = pd.Timestamp(ROBUSTNESS_START or SAMPLE_START, tz="UTC")
    end = pd.Timestamp(T_END, tz="UTC") + pd.Timedelta(days=1)  # hết ngày T_END
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in USE_COLS if c not in header]
    if missing:
        sys.exit("File thiếu cột: %s. Cột hiện có: %s" % (missing, header))
    parts, total, coin_counts, max_ts, min_ts = [], 0, {}, None, None
    for chunk in pd.read_csv(csv_path, usecols=USE_COLS, dtype=str, chunksize=CHUNK_ROWS):
        total += len(chunk)
        chunk["published_at_utc"] = parse_utc(chunk["Date Time"])
        c_max, c_min = chunk["published_at_utc"].max(), chunk["published_at_utc"].min()
        max_ts = c_max if max_ts is None or c_max > max_ts else max_ts
        min_ts = c_min if min_ts is None or c_min < min_ts else min_ts
        chunk = chunk[(chunk["published_at_utc"] >= start) & (chunk["published_at_utc"] < end)]
        for k, v in chunk["Coin Type"].fillna("(trống)").value_counts().items():
            coin_counts[k] = coin_counts.get(k, 0) + int(v)
        chunk = chunk[chunk["Coin Type"].fillna("").str.strip().str.lower().isin(COIN_KEEP)]
        parts.append(chunk.drop(columns=["Date Time"]))
    log["rows_in_file"] = total
    log["max_date_in_file"] = str(max_ts)   # điền vào source log (Max date)
    log["min_date_in_file"] = str(min_ts)
    log["coin_type_counts_in_window"] = dict(sorted(coin_counts.items(), key=lambda kv: -kv[1])[:15])
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=USE_COLS)
    log["rows_after_window_and_coin"] = len(df)
    return df


def step_merge_tags(df, log):
    """Bước 3: gộp tag asset theo URL, giữ bản sớm nhất của mỗi URL."""
    df["_url_norm"] = df["URL"].map(normalize_url)
    df["_asset"] = df["Coin Type"].str.strip().str.lower().map(
        {"bitcoin": "BTC", "btc": "BTC", "ethereum": "ETH", "eth": "ETH"})
    tags = df.groupby("_url_norm")["_asset"].agg(lambda s: sorted(set(s.dropna())))
    df = df.sort_values("published_at_utc").drop_duplicates("_url_norm", keep="first")
    df = df.reset_index(drop=True)
    df["currencies"] = df["_url_norm"].map(tags)
    log["rows_after_merge_tags_by_url"] = len(df)
    return df


def title_digits(title_norm):
    return "".join(re.findall(r"\d+", title_norm))


def step_dedup_title(df, log):
    """Bước 5: tiêu đề gần giống trong cửa sổ 48 giờ; giữ bài sớm hơn."""
    try:
        from rapidfuzz.fuzz import ratio as fast_ratio
        similar = lambda a, b: fast_ratio(a, b) / 100 >= TITLE_SIMILARITY
    except ImportError:  # không có rapidfuzz thì dùng difflib (chậm hơn)
        def similar(a, b):
            m = difflib.SequenceMatcher(None, a, b)
            return (m.real_quick_ratio() >= TITLE_SIMILARITY and m.quick_ratio() >= TITLE_SIMILARITY
                    and m.ratio() >= TITLE_SIMILARITY)
    df = df.sort_values("published_at_utc").reset_index(drop=True)
    norm = df["Title"].map(normalize_title).tolist()
    pubs = df["published_at_utc"].tolist()
    window = pd.Timedelta(hours=DEDUP_WINDOW_HOURS)
    keep, recent = [], []  # recent: (giờ đăng, tiêu đề chuẩn hóa, chuỗi số trong tiêu đề)
    for pub, title in zip(pubs, norm):
        recent = [r for r in recent if pub - r[0] <= window]
        digits = title_digits(title)
        dup = any(r[2] == digits and similar(title, r[1]) for r in recent)
        keep.append(not dup)
        if not dup:
            recent.append((pub, title, digits))
    log["dropped_dup_title"] = keep.count(False)
    return df[keep].reset_index(drop=True)


def clean(csv_path):
    log = {}
    df = step_read(csv_path, log)
    if df.empty:
        sys.exit("Không còn dòng nào sau khi cắt khung mẫu/lọc coin. Xem coin_type_counts trong log.")
    df = step_merge_tags(df, log)

    # Làm sạch ô văn bản, tạo nguồn
    df["title"] = df["Title"].map(clean_text_cell)
    df["description"] = df["Description"].map(clean_text_cell)
    df["source"] = df["URL"].map(source_of)
    df["nlp_text"] = df["title"]   # theo Excel: sentiment chạy trên headline (title)
    df["nlp_text_body"] = df["Full Text"].map(clean_text_cell)  # nội dung đầy đủ, TV3 tự chọn dùng

    # Bước 4: tiếng Anh và thông cáo báo chí
    df["non_en"] = df["nlp_text"].map(ascii_ratio) < ASCII_MIN_RATIO
    log["non_en_flagged"] = int(df["non_en"].sum())
    if DROP_NON_EN:
        df = df[~df["non_en"]]
    is_pr = df["URL"].str.contains("/press-releases/", case=False, na=False)
    log["press_release_flagged"] = int(is_pr.sum())
    if DROP_PRESS_RELEASE:
        df = df[~is_pr]
    df = df[df["title"] != ""].reset_index(drop=True)
    log["rows_after_language_and_press"] = len(df)

    df = step_dedup_title(df, log)

    # Bước 6: cờ timestamp
    year = df["published_at_utc"].dt.year
    df["ts_regime_risk"] = df["source"].isin(REGIME_RISK_SOURCES) & (year < 2021)
    same_ts = df.groupby(["source", "published_at_utc"])["URL"].transform("size")
    df["ts_bulk_flag"] = same_ts >= BULK_TS_MIN_ROWS
    ts = df["published_at_utc"]
    df["ts_minute_only"] = df["source"].isin(MINUTE_ONLY_SOURCES) & (ts.dt.second == 0)

    # Bước 7: id, ngày dùng, cột đầu ra
    df["doc_id"] = df["_url_norm"].map(lambda u: hashlib.sha1(u.encode("utf-8")).hexdigest())
    df["use_date"] = (ts.dt.normalize() + pd.Timedelta(days=1)).dt.tz_localize(None)
    df["in_main_sample"] = ts >= pd.Timestamp(SAMPLE_START, tz="UTC")
    df["dataset_origin"] = "cryptovision"
    df = df.rename(columns={"URL": "url"})
    cols = ["doc_id", "url", "source", "published_at_utc", "use_date", "title", "description",
            "nlp_text", "nlp_text_body", "currencies", "non_en", "ts_regime_risk",
            "ts_bulk_flag", "ts_minute_only", "in_main_sample", "dataset_origin"]
    df = df[cols].sort_values("published_at_utc").reset_index(drop=True)

    # Kiểm tra cuối: không có cột rò rỉ, giờ không rỗng, URL không trùng
    leaked = LEAK_COLS & set(df.columns)
    assert not leaked, "Còn cột rò rỉ: %s" % leaked
    assert df["published_at_utc"].notna().all(), "Có published_at_utc rỗng"
    assert df["doc_id"].is_unique, "Còn doc_id trùng"
    log["rows_final"] = len(df)
    log["first_published"] = str(df["published_at_utc"].min())
    log["last_published"] = str(df["published_at_utc"].max())
    log["rows_by_source"] = df["source"].value_counts().to_dict()
    log["rows_by_year"] = df["published_at_utc"].dt.year.value_counts().sort_index().to_dict()
    log["ts_flags"] = {c: int(df[c].sum()) for c in ("ts_regime_risk", "ts_bulk_flag", "ts_minute_only")}
    log["sample"] = {"start": ROBUSTNESS_START or SAMPLE_START, "main_start": SAMPLE_START, "t_end": T_END}
    return df, log


def main():
    cli_csv = sys.argv[1] if len(sys.argv) > 1 else None
    root = project_root()
    csv_path, out_path = load_paths(root, cli_csv)
    print("Doc file: %s" % csv_path)
    df, log = clean(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("Da luu %d dong vao %s" % (len(df), out_path))
    print("Log so lieu: %s" % log_path)
    for k in ("rows_in_file", "rows_after_window_and_coin", "rows_after_merge_tags_by_url",
              "non_en_flagged", "press_release_flagged", "dropped_dup_title", "rows_final"):
        print("  %-32s %s" % (k, log[k]))


if __name__ == "__main__":
    main()
"""clean_merge.py - Làm sạch CryptoVision v1 thành text_clean.parquet (TV2).

Mục đích
    Đọc file CryptoVision (Mendeley) và tạo MỘT bảng văn bản sạch: mỗi dòng là một
    bài, không trùng, giờ UTC, có news_date và use_date. Đầu ra giao cho TV3.
    Mốc thời gian lấy từ config/params.yaml (không hard-code): đọc từ
    robustness_start → t_end; in_main_sample chỉ True trong [sample.start, t_end],
    trừ nguồn exclude_sources_main và ngày dump.

Cách chạy (cần: pip install pandas pyarrow pyyaml rapidfuzz):
    python clean_merge.py                          # dùng đường dẫn mặc định
    python clean_merge.py duong_dan_file.csv       # chỉ định file CryptoVision

Quy định của nhóm:
    - KHÔNG đưa RSS holdout vào file này (RSS có clean_rss.py riêng).
    - Đường dẫn: config/paths.yaml; tham số: config/params.yaml. Seed 42.
    - Không bao giờ đọc cột giá / Market_Move. Sentiment sẵn có chỉ dưới tiền tố cv_ref_.
    - nlp_text = title + ". " + description (cùng công thức với RSS).
    - news_date = ngày UTC của published_at_utc; use_date = news_date + 1.
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

# ---- Tham số mặc định (bị ghi đè bởi config/params.yaml khi có) -------------
SAMPLE_START = "2021-03-01"
T_END = "2025-06-30"
ROBUSTNESS_START = "2020-01-01"
EXCLUDE_SOURCES_MAIN = ["cryptonews"]
DUMP_MAX_PER_SOURCE_DAY = 300
MIN_TITLE_CHARS = 20
JUNK_TITLES = ["x icon", "error 500", "page not found", "just a moment"]
COIN_KEEP = {"bitcoin", "ethereum", "btc", "eth"}
ASCII_MIN_RATIO = 0.90
DROP_NON_EN = True
DROP_PRESS_RELEASE = True
DEDUP_WINDOW_HOURS = 48
TITLE_SIMILARITY = 0.92
BULK_TS_MIN_ROWS = 10
MINUTE_ONLY_SOURCES = {"coindesk", "cointelegraph"}
REGIME_RISK_SOURCES = {"cryptopanic", "coindesk"}
CHUNK_ROWS = 50_000

DEFAULT_CSV_CANDIDATES = ["data/raw/cryptovision/cryptovision_v1.csv", "cryptovision_v1.csv"]
DEFAULT_OUT = "data/interim/text_clean.parquet"

# Chỉ nạp các cột này. Các cột rò rỉ không bao giờ được đọc.
# Excel yêu cầu tối thiểu URL, Title, Date_Time, Coin_Type. Giữ thêm Description và
# Full Text (thành nlp_text_body) để TV3 tự chọn dùng tiêu đề hay cả bài.
USE_COLS = ["URL", "Title", "Description", "Full Text", "Date Time", "Coin Type"]
# Hai cột sentiment có sẵn: giữ với tiền tố cv_ref_ chỉ để đối chiếu, KHÔNG làm đặc trưng
REF_COLS = ["sentiment_label", "sentiment_score"]
REF_RENAME = {"sentiment_label": "cv_ref_sentiment_label",
              "sentiment_score": "cv_ref_sentiment_score"}
REF_OUT_COLS = ["cv_ref_sentiment_label", "cv_ref_sentiment_score", "cv_ref_coin_type"]
LEAK_COLS ={"Open", "High", "Low", "Close", "Volume", "Movement_OpenClose_%",
             "Movement_HighLow_%", "Market_Move", "sentiment_label", "sentiment_score",
             "Sentiment_Label"}


def project_root():
    """Thư mục gốc dự án: gần nhất phía trên có config/; không có thì dùng thư mục file này."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_params(root):
    """Đọc mốc thời gian / ngưỡng từ config/params.yaml vào biến module."""
    global SAMPLE_START, T_END, ROBUSTNESS_START
    global EXCLUDE_SOURCES_MAIN, DUMP_MAX_PER_SOURCE_DAY, MIN_TITLE_CHARS, JUNK_TITLES
    cfg = root / "config" / "params.yaml"
    if not cfg.exists():
        return
    try:
        import yaml
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    sample = data.get("sample") or {}
    if sample.get("start"):
        SAMPLE_START = sample["start"]
    if sample.get("t_end"):
        T_END = sample["t_end"]
    if sample.get("robustness_start"):
        ROBUSTNESS_START = sample["robustness_start"]
    if "exclude_sources_main" in data:
        EXCLUDE_SOURCES_MAIN = list(data["exclude_sources_main"])
    if "dump_max_per_source_day" in data:
        DUMP_MAX_PER_SOURCE_DAY = int(data["dump_max_per_source_day"])
    if "min_title_chars" in data:
        MIN_TITLE_CHARS = int(data["min_title_chars"])
    if "junk_titles" in data:
        JUNK_TITLES = [str(j).lower() for j in data["junk_titles"]]
    seed = data.get("seed", 42)
    random.seed(seed)


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
    """Bước 1 và 2: đọc theo khối, đổi UTC, cắt khung (robustness→t_end), lọc coin."""
    start = pd.Timestamp(ROBUSTNESS_START or SAMPLE_START, tz="UTC")
    end = pd.Timestamp(T_END, tz="UTC") + pd.Timedelta(days=1)  # hết ngày T_END
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [c for c in USE_COLS if c not in header]
    if missing:
        sys.exit("File thiếu cột: %s. Cột hiện có: %s" % (missing, header))
    ref_cols = [c for c in REF_COLS if c in header]  # thiếu cột sentiment thì bỏ qua, không dừng
    log["has_reference_sentiment_cols"] = ref_cols
    parts, total, coin_counts, max_ts, min_ts = [], 0, {}, None, None
    for chunk in pd.read_csv(csv_path, usecols=USE_COLS + ref_cols, dtype=str, chunksize=CHUNK_ROWS):
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
    log["read_window"] = {"start": str(start), "end_exclusive": str(end),
                          "main_start": SAMPLE_START, "t_end": T_END}
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=USE_COLS + ref_cols)
    log["rows_after_window_and_coin"] = len(df)
    return df


def step_filter_titles(df, log):
    """Bỏ tiêu đề rác / quá ngắn; bỏ trùng tiêu đề chuẩn hóa (giữ bản sớm nhất)."""
    low = df["title"].str.strip().str.lower()
    junk = low.isin(JUNK_TITLES) | low.str.contains(r"server error", na=False)
    log["dropped_junk_titles"] = int(junk.sum())
    df = df[~junk].reset_index(drop=True)
    short = df["title"].str.len() < MIN_TITLE_CHARS
    log["dropped_short_titles"] = int(short.sum())
    df = df[~short].reset_index(drop=True)
    # Trùng tiêu đề chuẩn hóa toàn cục: giữ bài đăng sớm nhất
    df = df.sort_values("published_at_utc").reset_index(drop=True)
    before = len(df)
    df["_title_norm"] = df["title"].map(normalize_title)
    df = df.drop_duplicates(subset="_title_norm", keep="first").drop(columns=["_title_norm"])
    log["dropped_exact_dup_title"] = before - len(df)
    return df.reset_index(drop=True)


def step_flag_main_sample(df, log):
    """Gắn dump_day và in_main_sample theo params (cửa sổ chính, exclude, dump)."""
    main_start = pd.Timestamp(SAMPLE_START, tz="UTC")
    main_end = pd.Timestamp(T_END, tz="UTC") + pd.Timedelta(days=1)
    df = df.copy()
    df["day_utc"] = df["published_at_utc"].dt.tz_convert("UTC").dt.normalize()
    df["articles_source_day"] = df.groupby(["source", "day_utc"])["source"].transform("size")
    df["dump_day"] = df["articles_source_day"] > DUMP_MAX_PER_SOURCE_DAY
    exclude = {s.lower() for s in EXCLUDE_SOURCES_MAIN}
    in_window = (df["published_at_utc"] >= main_start) & (df["published_at_utc"] < main_end)
    df["in_main_sample"] = (
        in_window
        & ~df["source"].str.lower().isin(exclude)
        & ~df["dump_day"]
    )
    dump_keys = df.loc[df["dump_day"], ["source", "day_utc"]].drop_duplicates()
    log["dump_days_flagged"] = int(df["dump_day"].sum())
    log["dump_day_keys"] = [
        [r.source, str(r.day_utc.date())] for r in dump_keys.itertuples(index=False)
    ]
    log["in_main_sample_true"] = int(df["in_main_sample"].sum())
    log["in_main_sample_false"] = int((~df["in_main_sample"]).sum())
    # Bỏ hẳn bài ngày dump khỏi text_clean (tránh AbnAttention giả; test_no_dump_days)
    n_dump = int(df["dump_day"].sum())
    df = df[~df["dump_day"]].reset_index(drop=True)
    log["dropped_dump_day_rows"] = n_dump
    return df.drop(columns=["day_utc", "articles_source_day", "dump_day"])


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

    # Làm sạch ô văn bản, tạo nguồn; nlp_text thống nhất với RSS
    df["title"] = df["Title"].map(clean_text_cell)
    df["description"] = df["Description"].map(clean_text_cell)
    df["source"] = df["URL"].map(source_of)
    df["nlp_text"] = (df["title"] + ". " + df["description"]).str.strip().str.rstrip(".")
    # nếu description rỗng thì chỉ còn title (không để ". " thừa)
    empty_desc = df["description"] == ""
    df.loc[empty_desc, "nlp_text"] = df.loc[empty_desc, "title"]
    df["nlp_text_body"] = df["Full Text"].map(clean_text_cell)

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

    df = step_filter_titles(df, log)
    df = step_dedup_title(df, log)
    df = step_flag_main_sample(df, log)

    # Bước 6: cờ timestamp
    year = df["published_at_utc"].dt.year
    df["ts_regime_risk"] = df["source"].isin(REGIME_RISK_SOURCES) & (year < 2021)
    same_ts = df.groupby(["source", "published_at_utc"])["URL"].transform("size")
    df["ts_bulk_flag"] = same_ts >= BULK_TS_MIN_ROWS
    ts = df["published_at_utc"]
    df["ts_minute_only"] = df["source"].isin(MINUTE_ONLY_SOURCES) & (ts.dt.second == 0)

    # Bước 7: id, news_date, use_date, cột đầu ra
    df["doc_id"] = df["_url_norm"].map(lambda u: hashlib.sha1(u.encode("utf-8")).hexdigest())
    present = [c for c in REF_COLS if c in df.columns]
    if present:
        df = df.rename(columns=REF_RENAME)
        df["cv_ref_coin_type"] = df["Coin Type"]
        if "cv_ref_sentiment_score" in df.columns:
            df["cv_ref_sentiment_score"] = pd.to_numeric(df["cv_ref_sentiment_score"], errors="coerce")
    ref_out = [c for c in REF_OUT_COLS if c in df.columns]
    # news_date = ngày UTC (naive) của published_at_utc; use_date = news_date + 1
    news_date = ts.dt.tz_convert("UTC").dt.tz_localize(None).dt.normalize()
    df["news_date"] = news_date
    df["use_date"] = news_date + pd.Timedelta(days=1)
    df["dataset_origin"] = "cryptovision"
    df = df.rename(columns={"URL": "url"})
    cols = ["doc_id", "url", "source", "published_at_utc", "news_date", "use_date", "title",
            "description", "nlp_text", "nlp_text_body", "currencies", "non_en",
            "ts_regime_risk", "ts_bulk_flag", "ts_minute_only",
            "in_main_sample", "dataset_origin"] + ref_out
    df = df[cols].sort_values("published_at_utc").reset_index(drop=True)

    leaked = LEAK_COLS & set(df.columns)
    assert not leaked, "Còn cột rò rỉ: %s" % leaked
    bad_sent = [c for c in df.columns if "sentiment" in c.lower() and not c.startswith("cv_ref_")]
    assert not bad_sent, "Cột sentiment phải có tiền tố cv_ref_: %s" % bad_sent
    assert df["published_at_utc"].notna().all(), "Có published_at_utc rỗng"
    assert df["doc_id"].is_unique, "Còn doc_id trùng"
    assert df["published_at_utc"].max() < pd.Timestamp(T_END, tz="UTC") + pd.Timedelta(days=1)
    assert df["in_main_sample"].dtype == bool
    log["rows_final"] = len(df)
    log["first_published"] = str(df["published_at_utc"].min())
    log["last_published"] = str(df["published_at_utc"].max())
    log["rows_by_source"] = df["source"].value_counts().to_dict()
    log["rows_by_year"] = df["published_at_utc"].dt.year.value_counts().sort_index().to_dict()
    log["ts_flags"] = {c: int(df[c].sum()) for c in ("ts_regime_risk", "ts_bulk_flag", "ts_minute_only")}
    log["sample"] = {
        "robustness_start": ROBUSTNESS_START or SAMPLE_START,
        "main_start": SAMPLE_START,
        "t_end": T_END,
        "exclude_sources_main": EXCLUDE_SOURCES_MAIN,
        "dump_max_per_source_day": DUMP_MAX_PER_SOURCE_DAY,
    }
    if "cv_ref_sentiment_label" in df.columns:
        log["reference_label_counts"] = df["cv_ref_sentiment_label"].value_counts(dropna=False).to_dict()
    return df, log


def main():
    cli_csv = sys.argv[1] if len(sys.argv) > 1 else None
    root = project_root()
    load_params(root)
    csv_path, out_path = load_paths(root, cli_csv)
    print("Doc file: %s" % csv_path)
    print("Khung: robustness=%s main=%s..%s" % (ROBUSTNESS_START, SAMPLE_START, T_END))
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
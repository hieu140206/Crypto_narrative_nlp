"""clean_rss.py - Làm sạch RSS holdout (TV2).

Mục đích
    Gộp các file data/raw/rss_holdout/YYYY-MM-DD.jsonl (do rss_holdout.py thu)
    thành MỘT bảng sạch: data/interim/rss_holdout.parquet.
    Chạy một lần sau khi thu xong (D7). Có thể chạy thử bất cứ lúc nào để xem
    số liệu; chạy lại luôn ra cùng kết quả vì đọc từ file thô, không sửa file thô.

Cách chạy (cần pandas và pyarrow: pip install pandas pyarrow):
    python clean_rss.py

Quy định của nhóm:
    - File này TÁCH RIÊNG với clean_merge.py (CryptoVision). KHÔNG đưa RSS vào
      text_clean.parquet.
    - Không import vào master_pipeline.ipynb.
    - Đường dẫn lấy từ config/paths.yaml (khóa: rss_holdout_dir, rss_clean_path);
      nếu thiếu thì dùng mặc định bên dưới. Seed 42.

Các bước (theo thứ tự):
    1. Gộp file .jsonl, bỏ dòng trùng khóa.
    2. Chuẩn hóa tên và kiểu cột (collected_at -> collected_at_utc, giờ UTC).
    3. Bỏ bài is_backfill = true (quá cũ hoặc giờ đăng nằm sau giờ thu).
    4. Đối chiếu ngày trong URL với published_at, gắn cờ url_date_mismatch.
    5. Lọc chủ đề: gắn cờ crypto_relevant; chỉ bỏ bài ngoài chủ đề ở nguồn Decrypt.
    6. Dedup hai lớp: URL đã chuẩn hóa, rồi tiêu đề gần giống trong 48 giờ.
    7. Tính available_at và use_date (quy tắc chống look-ahead), tạo nlp_text.
    8. Lưu parquet và file log số liệu.

Quy tắc gán ngày (look-ahead):
    available_at = max(published_at, collected_at_utc).
    Bài có available_at trước 00:00 UTC của ngày t thì dùng cho ngày t; bài từ
    00:00 UTC trở đi trong ngày t thì dùng cho ngày t+1. Nghĩa là use_date =
    (ngày UTC của available_at) + 1 ngày.
"""
import difflib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

random.seed(42)

# ---- Tham số (có thể chuyển sang config/params.yaml sau) -------------------
DEFAULT_RAW_DIR = "data/raw/rss_holdout"
DEFAULT_OUT_PATH = "data/interim/rss_holdout.parquet"
DEDUP_WINDOW_HOURS = 48       # cửa sổ so sánh tiêu đề gần giống
TITLE_SIMILARITY = 0.92       # tỷ lệ giống nhau tối thiểu để coi là trùng tiêu đề
URL_DATE_TOLERANCE_DAYS = 1   # lệch từ mức này trở lên thì gắn cờ
DROP_URL_MISMATCH = False     # True: bỏ luôn bài lệch ngày URL (nhóm quyết định)
# Nguồn chuyên crypto (CoinDesk, Cointelegraph) chỉ gắn cờ crypto_relevant, không bỏ;
# chỉ bỏ bài ngoài chủ đề ở các nguồn đăng cả tin công nghệ chung (Decrypt).
OFF_TOPIC_FILTER_SOURCES = {"decrypt"}

# Từ khóa nhận biết bài về crypto (so khớp không phân biệt hoa thường)
CRYPTO_PATTERN = re.compile(
    r"crypto|bitcoin|\bbtc\b|ethereum|\beth\b|\bether\b|blockchain|stablecoin|"
    r"\bdefi\b|\bnft|token|altcoin|solana|\bsol\b|\bxrp\b|ripple|dogecoin|binance|"
    r"coinbase|\bsec\b|\bcftc\b|etf|wallet|mining|miner|halving|web3|memecoin|"
    r"airdrop|ledger|tether|\busdt\b|\busdc\b|exchange|on-chain|onchain|"
    r"digital asset|satoshi|cardano|polkadot|avalanche|chainlink|layer[- ]?2|"
    r"tokeniz|treasury|market cap|bull run|bear market|liquidat|perpetual|"
    r"saylor|vitalik|hoskinson|fairshake|super pac|quantum|monero|zcash|"
    r"mt\.? ?gox|kalshi|polymarket|prediction market|proof-of-human|worldcoin|"
    r"privacy coin|\bdao\b|staking|validator|rollup|metamask|gensler|"
    r"\bbank|payments?|\bfed\b|interest rate|\bcpi\b|inflation",
    re.IGNORECASE,
)
URL_DATE_PATTERN = re.compile(r"/(20\d\d)/(\d\d)/(\d\d)/")


def project_root():
    """Thư mục gốc dự án: gần nhất phía trên có config/; không có thì dùng thư mục file này."""
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_paths(root):
    """Đọc đường dẫn từ config/paths.yaml nếu có PyYAML, không thì dùng mặc định."""
    raw, out = DEFAULT_RAW_DIR, DEFAULT_OUT_PATH
    cfg = root / "config" / "paths.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            raw = data.get("rss_holdout_dir", raw)
            out = data.get("rss_clean_path", out)
        except Exception:
            pass
    raw_p, out_p = Path(raw), Path(out)
    return (raw_p if raw_p.is_absolute() else root / raw_p,
            out_p if out_p.is_absolute() else root / out_p)


def normalize_url(url):
    """Chuẩn hóa URL để so trùng: bỏ tham số theo dõi (utm_...), dấu / cuối, chữ hoa."""
    url = (url or "").strip().lower()
    url = url.split("#")[0]
    if "?" in url:
        base, query = url.split("?", 1)
        keep = [q for q in query.split("&") if q and not q.startswith(("utm_", "ref=", "fbclid"))]
        url = base + ("?" + "&".join(keep) if keep else "")
    return url.rstrip("/")


def normalize_title(title):
    """Chuẩn hóa tiêu đề để so gần giống: chữ thường, bỏ dấu câu, gom khoảng trắng."""
    t = re.sub(r"[^\w\s]", " ", (title or "").lower())
    return re.sub(r"\s+", " ", t).strip()


def step_load(raw_dir):
    """Bước 1: gộp file .jsonl, bỏ dòng trùng khóa."""
    files = sorted(raw_dir.glob("2*.jsonl"))
    if not files:
        sys.exit("Không tìm thấy file .jsonl trong %s" % raw_dir)
    frames = [pd.read_json(f, lines=True, dtype=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset="key", keep="first").reset_index(drop=True), len(files)


def step_normalize(df):
    """Bước 2: đổi tên và kiểu cột, giờ về UTC."""
    df = df.rename(columns={"collected_at": "collected_at_utc"})
    for col in ("published_at", "collected_at_utc"):
        df[col] = pd.to_datetime(df[col], utc=True)
    for col in ("title", "summary"):
        df[col] = df[col].fillna("").astype(str)
    return df


def url_date_flag(row):
    """Bước 4: True nếu ngày trong URL lệch published_at từ mức cho phép trở lên."""
    m = URL_DATE_PATTERN.search(row["link"] or "")
    if not m:
        return False  # không có ngày trong URL thì không đối chiếu được
    url_date = pd.Timestamp("%s-%s-%s" % m.groups(), tz="UTC")
    pub_date = row["published_at"].normalize()
    return abs((pub_date - url_date).days) >= URL_DATE_TOLERANCE_DAYS


def step_dedup(df):
    """Bước 6: dedup hai lớp. Trả về (bảng đã lọc, số bài bỏ lớp 1, số bài bỏ lớp 2)."""
    # Lớp 1: trùng URL đã chuẩn hóa (giữ bài thu sớm nhất)
    df = df.sort_values("collected_at_utc").reset_index(drop=True)
    df["_url_norm"] = df["link"].map(normalize_url)
    before = len(df)
    df = df.drop_duplicates(subset="_url_norm", keep="first").reset_index(drop=True)
    dropped_url = before - len(df)

    # Lớp 2: tiêu đề gần giống nhau trong cửa sổ 48 giờ (giữ bài đăng sớm hơn)
    df = df.sort_values("published_at").reset_index(drop=True)
    df["_title_norm"] = df["title"].map(normalize_title)
    window = pd.Timedelta(hours=DEDUP_WINDOW_HOURS)
    keep, recent = [], []  # recent: (giờ đăng, tiêu đề chuẩn hóa) của các bài đã giữ
    for pub, title in zip(df["published_at"], df["_title_norm"]):
        recent = [(t, s) for t, s in recent if pub - t <= window]
        is_dup = any(
            difflib.SequenceMatcher(None, title, s).ratio() >= TITLE_SIMILARITY
            for _, s in recent
        )
        keep.append(not is_dup)
        if not is_dup:
            recent.append((pub, title))
    dropped_title = keep.count(False)
    df = df[keep].drop(columns=["_url_norm", "_title_norm"]).reset_index(drop=True)
    return df, dropped_url, dropped_title


def clean(raw_dir):
    """Chạy toàn bộ các bước, trả về (bảng sạch, log số liệu)."""
    log = {}
    df, n_files = step_load(raw_dir)
    log["files"], log["rows_loaded"] = n_files, len(df)
    df = step_normalize(df)

    # Bước 3: bỏ bài backfill
    n = len(df)
    df = df[~df["is_backfill"].astype(bool)].reset_index(drop=True)
    log["dropped_backfill"] = n - len(df)

    # Bước 4: gắn cờ lệch ngày URL
    df["url_date_mismatch"] = df.apply(url_date_flag, axis=1)
    log["url_date_mismatch_flagged"] = int(df["url_date_mismatch"].sum())
    if DROP_URL_MISMATCH:
        df = df[~df["url_date_mismatch"]].reset_index(drop=True)

    # Bước 5: lọc chủ đề crypto
    text = df["title"] + " " + df["summary"] + " " + df["categories"].map(
        lambda c: " ".join(c) if isinstance(c, list) else str(c))
    df["crypto_relevant"] = text.map(lambda s: bool(CRYPTO_PATTERN.search(s)))
    log["off_topic_flagged"] = int((~df["crypto_relevant"]).sum())
    log["off_topic_by_source"] = df.loc[~df["crypto_relevant"], "source"].value_counts().to_dict()
    log["off_topic_titles"] = df.loc[~df["crypto_relevant"], "title"].tolist()  # để đọc tay kiểm tra
    drop_mask = (~df["crypto_relevant"]) & df["source"].isin(OFF_TOPIC_FILTER_SOURCES)
    log["off_topic_dropped"] = int(drop_mask.sum())
    df = df[~drop_mask].reset_index(drop=True)

    # Bước 6: dedup
    df, log["dropped_dup_url"], log["dropped_dup_title"] = step_dedup(df)

    # Bước 7: available_at, use_date, nlp_text
    df["available_at"] = df[["published_at", "collected_at_utc"]].max(axis=1)
    df["use_date"] = df["available_at"].dt.normalize() + pd.Timedelta(days=1)
    df["use_date"] = df["use_date"].dt.tz_localize(None)  # lưu dạng ngày, không kèm múi giờ
    df["nlp_text"] = (df["title"] + ". " + df["summary"]).str.strip()

    cols = ["source", "title", "summary", "nlp_text", "link", "categories",
            "published_at", "collected_at_utc", "available_at", "use_date",
            "lag_minutes", "published_missing", "url_date_mismatch", "crypto_relevant"]
    df = df[cols].sort_values("available_at").reset_index(drop=True)
    log["rows_final"] = len(df)
    log["rows_by_source"] = df["source"].value_counts().to_dict()
    log["first_use_date"] = str(df["use_date"].min().date())
    log["last_use_date"] = str(df["use_date"].max().date())
    return df, log


def main():
    root = project_root()
    raw_dir, out_path = load_paths(root)
    df, log = clean(raw_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log_path = out_path.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Da luu %d dong vao %s" % (len(df), out_path))
    print("Log so lieu: %s" % log_path)
    for k in ("files", "rows_loaded", "dropped_backfill", "url_date_mismatch_flagged",
              "off_topic_flagged", "off_topic_dropped", "dropped_dup_url", "dropped_dup_title", "rows_final"):
        print("  %-28s %s" % (k, log[k]))


if __name__ == "__main__":
    main()

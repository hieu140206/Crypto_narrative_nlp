"""rss_holdout.py - Thu RSS holdout (TV2).

Mục đích
    Thu tin trực tiếp từ 3 feed RSS gốc (CoinDesk, Cointelegraph, Decrypt) để làm
    tập holdout chống look-ahead. Mỗi tin ghi published_at (giờ đăng, UTC) và
    collected_at (giờ script chạy, UTC) do CHÍNH script này đặt.

Cách chạy (mỗi 30 phút, qua Task Scheduler / cron):
    python -m src.data.rss_holdout            # trong thư mục gốc dự án
    python rss_holdout.py                     # hoặc chạy thẳng file

Quy định của nhóm:
    - KHÔNG import file này vào master_pipeline.ipynb.
    - Đường dẫn lấy từ config/paths.yaml (khóa: rss_holdout_dir); nếu thiếu thì
      dùng data/raw/rss_holdout.
    - Seed 42.
    - Chỉ thêm vào cuối file (append-only), không sửa tin đã ghi.

Quy tắc giờ:
    - published_at lấy từ RSS (pubDate), đổi về UTC. Thiếu thì dùng collected_at và
      đặt published_missing = true.
    - lag_minutes = collected_at - published_at (đơn vị: phút).
    - is_backfill = true nếu lag > 7 ngày (mục cũ lẫn vào feed) hoặc giờ đăng nằm
      sau giờ thu (lỗi đồng hồ của feed). Loại các dòng này khỏi tập holdout.
"""
import hashlib
import html
import json
import random
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

random.seed(42)
try:  # numpy không bắt buộc
    import numpy as np
    np.random.seed(42)
except ImportError:
    pass

# Ba feed gốc; không đi qua cryptocurrency.cv vì đó chỉ là bản sao RSS
FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cointelegraph": "https://cointelegraph.com/rss",
    "decrypt": "https://decrypt.co/feed",
}
USER_AGENT = "Mozilla/5.0 (research project; rss_holdout)"
TIMEOUT = 30            # số giây chờ tối đa cho mỗi feed
BACKFILL_DAYS = 7       # chênh lệch lớn hơn mức này thì gắn is_backfill
DEFAULT_DIR = "data/raw/rss_holdout"

# Namespace (không gian tên XML) dùng khi đọc RSS/Atom
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def project_root():
    """Tìm thư mục gốc dự án: thư mục gần nhất phía trên có chứa config/.

    Nếu không thấy config/ thì dùng thư mục chứa file này (không dùng thư mục
    đang chạy, vì Task Scheduler chạy ở System32 và sẽ ghi dữ liệu sai chỗ).
    """
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def load_out_dir(root):
    """Đọc đường dẫn từ config/paths.yaml nếu có PyYAML, không thì dùng mặc định."""
    cfg = root / "config" / "paths.yaml"
    rel = DEFAULT_DIR
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            rel = data.get("rss_holdout_dir", DEFAULT_DIR)
        except Exception:
            pass
    out = Path(rel)
    return out if out.is_absolute() else root / out


def now_utc():
    return datetime.now(timezone.utc)


def iso(dt):
    """Định dạng thời gian chuẩn ISO 8601, múi giờ UTC."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_html(text):
    """Bỏ thẻ HTML và ký tự đặc biệt, gom khoảng trắng."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def parse_date(raw):
    """Trả về datetime UTC, hoặc None nếu không đọc được."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:  # không có múi giờ: coi là UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def text_of(el, path):
    found = el.find(path, NS)
    return (found.text or "").strip() if found is not None and found.text else ""


def parse_feed(xml_bytes):
    """Đọc RSS 2.0 hoặc Atom, trả về danh sách dict thô."""
    root = ET.fromstring(xml_bytes)
    items = []
    for it in root.iter("item"):
        items.append({
            "title": text_of(it, "title"),
            "link": text_of(it, "link"),
            "guid": text_of(it, "guid"),
            "summary": text_of(it, "description") or text_of(it, "content:encoded"),
            "published_raw": text_of(it, "pubDate") or text_of(it, "dc:date"),
            "categories": [c.text.strip() for c in it.findall("category") if c.text],
        })
    if not items:  # không có <item> thì thử định dạng Atom
        for en in root.iter("{%s}entry" % NS["atom"]):
            link_el = en.find("atom:link", NS)
            items.append({
                "title": text_of(en, "atom:title"),
                "link": link_el.get("href", "") if link_el is not None else "",
                "guid": text_of(en, "atom:id"),
                "summary": text_of(en, "atom:summary") or text_of(en, "atom:content"),
                "published_raw": text_of(en, "atom:published") or text_of(en, "atom:updated"),
                "categories": [c.get("term", "") for c in en.findall("atom:category", NS)],
            })
    return items


def make_record(source, raw, collected):
    """Tạo một bản ghi hoàn chỉnh; trả về (khóa chống trùng, bản ghi)."""
    link = raw["link"] or raw["guid"]
    pub = parse_date(raw["published_raw"])
    missing = pub is None
    if missing:  # thiếu giờ đăng: dùng giờ thu và gắn cờ cảnh báo
        pub = collected
    lag = round((collected - pub).total_seconds() / 60, 1)
    # Cũ quá 7 ngày, hoặc giờ đăng nằm sau giờ thu hơn 5 phút (đồng hồ feed lệch)
    is_backfill = lag > BACKFILL_DAYS * 24 * 60 or lag < -5
    key = hashlib.sha1(link.strip().encode("utf-8")).hexdigest()
    return key, {
        "key": key,
        "source": source,
        "title": strip_html(raw["title"]),
        "link": link,
        "summary": strip_html(raw["summary"]),
        "categories": raw["categories"],
        "published_at": iso(pub),
        "published_missing": missing,
        "collected_at": iso(collected),
        "lag_minutes": lag,
        "is_backfill": is_backfill,
    }


def fetch(url):
    """Tải nội dung một feed."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def load_seen(path):
    """Đọc danh sách khóa các tin đã ghi để không ghi trùng."""
    if not path.exists():
        return set()
    return set(path.read_text(encoding="utf-8").split())


def append_lines(path, lines):
    """Chỉ thêm vào cuối file (append-only), không sửa dữ liệu cũ."""
    if lines:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run(feeds=None, out_dir=None, fetcher=fetch):
    """Chạy một lượt thu. Feed nào lỗi thì ghi lỗi, không chặn feed khác."""
    feeds = feeds or FEEDS
    out_dir = Path(out_dir) if out_dir else load_out_dir(project_root())
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_path = out_dir / "_seen.txt"
    seen = load_seen(seen_path)
    collected = now_utc()
    # File theo ngày thu (UTC), mỗi ngày một file
    day_file = out_dir / (collected.strftime("%Y-%m-%d") + ".jsonl")
    run_log = {"run_at": iso(collected), "feeds": {}}
    new_lines, new_keys = [], []

    for name, url in feeds.items():
        info = {"ok": False, "items": 0, "new": 0, "error": None}
        try:
            raws = parse_feed(fetcher(url))
            info["items"] = len(raws)
            for raw in raws:
                if not (raw["link"] or raw["guid"]):
                    continue
                key, rec = make_record(name, raw, collected)
                if key in seen:
                    continue
                seen.add(key)
                new_keys.append(key)
                new_lines.append(json.dumps(rec, ensure_ascii=False))
                info["new"] += 1
            info["ok"] = True
        except Exception as exc:  # noqa: BLE001 - ghi lỗi rồi đi tiếp feed khác
            info["error"] = "%s: %s" % (type(exc).__name__, exc)
        run_log["feeds"][name] = info

    append_lines(day_file, new_lines)
    append_lines(seen_path, new_keys)
    append_lines(out_dir / "_runs.jsonl", [json.dumps(run_log, ensure_ascii=False)])
    return run_log


def main():
    log = run()
    for name, info in log["feeds"].items():
        status = "OK " if info["ok"] else "LOI"  # in không dấu để tránh lỗi font console Windows
        print("[%s] %-14s items=%-3d moi=%-3d %s"
              % (status, name, info["items"], info["new"], info["error"] or ""))
    # Mã thoát 1 nếu cả 3 feed đều lỗi (để Task Scheduler nhận biết)
    sys.exit(0 if any(i["ok"] for i in log["feeds"].values()) else 1)


if __name__ == "__main__":
    main()
# Independent read-only RSS news widget; no writes to historical research data.
from __future__ import annotations
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlparse, urljoin, quote
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen
import re
import xml.etree.ElementTree as ET
import streamlit as st

FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
    "Cointelegraph ETH": "https://cointelegraph.com/rss/tag/ethereum",
    "Cointelegraph SOL": "https://cointelegraph.com/rss/tag/solana",
    "Decrypt ETH": "https://decrypt.co/feed?tag=ethereum",
    "Decrypt SOL": "https://decrypt.co/feed?tag=solana",
}
# Independent topic-specific RSS queries. These do not touch historical datasets.
TOPIC_FEEDS = {
    "BTC": "https://news.google.com/rss/search?q=" + quote('Bitcoin OR BTC cryptocurrency when:7d') + "&hl=en-US&gl=US&ceid=US:en",
    "ETH": "https://news.google.com/rss/search?q=" + quote('Ethereum OR Ether OR ETH crypto when:7d') + "&hl=en-US&gl=US&ceid=US:en",
    "SOL": "https://news.google.com/rss/search?q=" + quote('Solana OR SOL cryptocurrency when:7d') + "&hl=en-US&gl=US&ceid=US:en",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CryptoVisionRSSReader/1.0)"}
MEDIA = "{http://search.yahoo.com/mrss/}"
CONTENT = "{http://purl.org/rss/1.0/modules/content/}"

def _safe_url(url):
    url = (url or "").strip()
    parsed = urlparse(url)
    return url if parsed.scheme in ("http", "https") and parsed.netloc else ""

class _ImageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.image = ""
    def handle_starttag(self, tag, attrs):
        data = dict(attrs)
        if tag.lower() == "img" and not self.image:
            self.image = data.get("data-src") or data.get("src") or ""
            return
        if tag.lower() != "meta":
            return
        if data.get("property", "").lower() in ("og:image", "og:image:url", "twitter:image") or data.get("name", "").lower() in ("twitter:image", "twitter:image:src"):
            self.image = self.image or data.get("content", "")

def _article_image(url):
    # Best effort: publisher may block metadata requests. Never save images.
    try:
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=3) as response:
            if "text/html" not in response.headers.get("Content-Type", ""):
                return ""
            page = response.read(180_000).decode("utf-8", errors="ignore")
        parser = _ImageParser()
        parser.feed(page)
        return _safe_url(urljoin(url, parser.image)) if parser.image else ""
    except Exception:
        return ""

def _rss_image(node, link):
    for tag in (MEDIA + "content", MEDIA + "thumbnail", "enclosure", "image"):
        for element in node.findall(tag):
            candidate = element.get("url", "") or element.findtext("url") or ""
            mime = element.get("type", "")
            if candidate and (tag != "enclosure" or mime.startswith("image/")):
                return _safe_url(urljoin(link, candidate))
    # Some publishers embed an image in the RSS description.
    for field in ("description", CONTENT + "encoded", "summary"):
        body = node.findtext(field) or ""
        match = re.search(r'<img\b[^>]*\b(?:src|data-src)=["\']([^"\']+)', body, re.I)
        if match:
            return _safe_url(urljoin(link, match.group(1)))
    return ""

def _fetch_feed(source, url):
    req = Request(url, headers={**HEADERS, "Accept": "application/rss+xml, application/xml, text/xml"})
    with urlopen(req, timeout=7) as response:
        root = ET.fromstring(response.read(2_000_000))
    items = []
    for node in root.findall(".//item")[:45]:
        title = (node.findtext("title") or "").strip()
        link = _safe_url(node.findtext("link"))
        if not title or not link:
            continue
        try:
            dt = parsedate_to_datetime(node.findtext("pubDate") or "").astimezone(timezone.utc)
        except Exception:
            dt = datetime.min.replace(tzinfo=timezone.utc)
        items.append({"source": source.replace(" ETH", "").replace(" SOL", ""), "title": title, "link": link,
                      "published": dt, "image": _rss_image(node, link)})
    return items

@st.cache_data(ttl=55, show_spinner=False)
def load_news():
    news, errors = [], []
    for source, url in {**FEEDS, **TOPIC_FEEDS}.items():
        try:
            news.extend(_fetch_feed(source, url))
        except Exception as exc:
            errors.append(f"{source}: {type(exc).__name__}")
    # Prefer direct publisher items over Google News syndication for cover images.
    unique = {item["link"]: item for item in news}
    news = sorted(unique.values(), key=lambda item: item["published"], reverse=True)
    # Keep independent pools so an abundance of BTC/general news cannot crowd out ETH/SOL.
    def topics(item):
        title = item["title"].lower()
        matches = []
        for label, pattern in (("btc", r"\b(bitcoin|btc)\b"),
                               ("eth", r"\b(ethereum|ether|eth)\b"),
                               ("sol", r"\b(solana|sol)\b")):
            if re.search(pattern, title):
                matches.append(label)
        return matches
    selected = []
    for topic in ("btc", "eth", "sol"):
        selected.extend(item for item in news if topic in topics(item))
    selected.extend(news[:30])
    news = list({item["link"]: item for item in selected}.values())
    news.sort(key=lambda item: item["published"], reverse=True)
    # Only enrich the first few visible stories when RSS omits images.
    # This is read-only and never modifies historical datasets or model results.
    # Enrich images across ALL tabs, not only the first eight global headlines.
    # RSS media remains the preferred source. Metadata lookups are best-effort.
    candidates = []
    for label in ("btc", "eth", "sol"):
        candidates.extend(item for item in news if label in topics(item) and not item["image"])
    candidates.extend(item for item in news[:8] if not item["image"])
    targets = list({item["link"]: item for item in candidates if "news.google.com" not in urlparse(item["link"]).netloc}.values())[:36]
    if targets:
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = list(pool.map(lambda item: _article_image(item["link"]), targets))
        for item, image in zip(targets, images):
            if image:
                item["image"] = image
    # Display direct publisher articles with real images first in each coin tab.
    # No invented thumbnails: articles without an available image keep the coin icon.
    news.sort(key=lambda item: (bool(item.get("image")), item["published"]), reverse=True)
    return news, errors, datetime.now(timezone.utc)

def make_news_markup(items, checked, errors):
    now = datetime.now(timezone.utc)
    rows = []
    for item in items:
        title = escape(item["title"])
        link = escape(item["link"], quote=True)
        source = escape(item["source"])
        delta = max(0, int((now - item["published"]).total_seconds()))
        ago = ("Time unavailable" if item["published"].year < 1900 else
               f"{max(1,delta//60)} min ago" if delta < 3600 else
               f"{delta//3600} hours ago" if delta < 86400 else
               f"{delta//86400} days ago")
        low = item["title"].lower()
        cats = []
        if re.search(r'\b(bitcoin|btc)\b', low): cats.append("btc")
        if re.search(r'\b(ethereum|ether|eth)\b', low): cats.append("eth")
        if re.search(r'\b(solana|sol)\b', low): cats.append("sol")
        coin = next((c.upper() for c in cats), "NEWS")
        symbol = {"BTC":"₿", "ETH":"◆", "SOL":"◎", "NEWS":"↗"}[coin]
        image_url = _safe_url(item.get("image", ""))
        if image_url:
            thumb = (f'<span class="live-thumb"><img src="{escape(image_url, quote=True)}" '
                     f'alt="" loading="lazy" referrerpolicy="no-referrer" '
                     f'onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\'">'
                     f'<span class="live-fallback {coin.lower()}" style="display:none">{symbol}</span></span>')
        else:
            thumb = f'<span class="live-thumb"><span class="live-fallback {coin.lower()}">{symbol}</span></span>'
        rows.append(
            f'<a class="live-row" data-coin="{" ".join(cats)}" href="{link}" target="_blank" rel="noopener noreferrer">'
            f'{thumb}<span class="live-story"><strong>{title}</strong><small>{source} · {ago}</small></span>'
            f'<span class="live-tag">{coin}</span></a>'
        )
    status = ("Some sources unavailable · " if errors else "") + "Updated " + checked.astimezone().strftime("%H:%M:%S")
    content = "".join(rows) if rows else '<div class="live-empty">Live articles temporarily unavailable.</div>'
    return f"""<div class="live-card" id="news-live">
      <div class="live-heading"><h2><span class="ico">▣</span> Latest Crypto News <span class="live-dot"></span></h2>
      <button class="live-view" onclick="toggleNews(this)">View all →</button></div>
      <div class="live-meta">LIVE RSS · refresh every 60 seconds · {escape(status)}</div>
      <div class="live-tabs">
        <button class="sel" onclick="filterNews(this,'all')">All</button>
        <button onclick="filterNews(this,'btc')">BTC</button>
        <button onclick="filterNews(this,'eth')">ETH</button>
        <button onclick="filterNews(this,'sol')">SOL</button>
      </div>
      <div class="live-list">{content}</div><div class="live-filter-empty" style="display:none;padding:15px;color:#7185a5;font-size:12px">No matching recent headlines from the available RSS sources. Try Refresh or All.</div>
      <div class="live-foot">Publisher cover images where available · Live news separate from historical research</div>
    </div>
<style>
*{{box-sizing:border-box}}
.live-card{{border:1px solid #dbe8ff;border-radius:16px;background:#fff;padding:17px;box-shadow:0 8px 26px #315caa0d;margin-top:8px;font-family:Arial,sans-serif}}
.live-heading{{display:flex;justify-content:space-between;align-items:center;gap:10px}}
.live-heading h2{{font-size:16px!important;color:#10265b;margin:0;font-weight:800}}
.live-dot{{display:inline-block;width:7px;height:7px;background:#00c6a8;border-radius:50%;vertical-align:middle}}
.live-view{{border:0;background:transparent;color:#2679ff;font-size:12px;font-weight:700;cursor:pointer;white-space:nowrap}}
.live-meta{{font-size:10px;color:#8193b5;margin:7px 0 12px}}
.live-tabs{{display:flex;gap:3px;background:#edf3ff;border-radius:12px;padding:4px;margin-bottom:9px;max-width:460px}}
.live-tabs button{{flex:1;border:0;background:transparent;border-radius:9px;padding:8px;color:#5a7098;font-size:11px;cursor:pointer}}
.live-tabs button.sel{{background:#fff;color:#205bdb;font-weight:800;box-shadow:0 2px 7px #3854a81a}}
.live-row{{display:flex;align-items:center;gap:12px;padding:10px 2px;border-bottom:1px solid #edf1fb;text-decoration:none;color:#112657;min-height:86px;transition:background .15s}}
.live-row:hover{{background:#f5f8ff}}
.live-thumb{{flex:0 0 96px;width:96px;height:66px;border-radius:11px;overflow:hidden;background:#e8effc;display:block}}
.live-thumb img{{width:100%;height:100%;object-fit:cover;display:block}}
.live-fallback{{width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:27px;color:white;background:linear-gradient(135deg,#9cb5e7,#587bc4)}}
.live-fallback.btc{{background:linear-gradient(135deg,#ffc769,#e7851b)}}
.live-fallback.eth{{background:linear-gradient(135deg,#a8a6fa,#5958be)}}
.live-fallback.sol{{background:linear-gradient(135deg,#65d6d2,#6958c9)}}
.live-story{{min-width:0;flex:1}}
.live-story strong{{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;font-size:12px;line-height:1.4;color:#13234f}}
.live-story small{{display:block;margin-top:6px;font-size:10px;color:#8393b3}}
.live-tag{{font-size:10px;background:#e9f3ff;border-radius:15px;padding:5px 8px;color:#3867b5;font-weight:700;white-space:nowrap}}
.live-foot{{font-size:9px;color:#8797b4;padding-top:11px}}
.live-empty{{font-size:12px;padding:18px;color:#6b7a9c}}
@media(max-width:530px){{.live-thumb{{flex-basis:75px;width:75px;height:60px}}.live-tag{{display:none}}.live-card{{padding:12px}}}}
</style>
<script>
let selectedCoin='all', expanded=false;
function refreshRows(){{
  const rows=[...document.querySelectorAll('#news-live .live-row')];
  const matched=rows.filter(x=>selectedCoin==='all'||x.dataset.coin.split(' ').includes(selectedCoin));
  rows.forEach(x=>x.style.display='none');
  matched.slice(0,expanded?30:4).forEach(x=>x.style.display='flex');
}}
function filterNews(btn,coin){{
  selectedCoin=coin;
  document.querySelectorAll('#news-live .live-tabs button').forEach(x=>x.classList.remove('sel'));
  btn.classList.add('sel');
  refreshRows();
  const empty=document.querySelector('#news-live .live-filter-empty');
  if(empty) empty.style.display=document.querySelectorAll('#news-live .live-row[style*="flex"]').length?'none':'block';
}}
function toggleNews(btn){{
  expanded=!expanded;
  btn.textContent=expanded?'Show less ↑':'View all →';
  refreshRows();
}}
refreshRows();
</script>"""

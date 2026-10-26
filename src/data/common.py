"""Hàm dùng chung: đọc config, set seed, ghi source log, lưu/đọc bảng."""
import csv, random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_paths() -> dict:
    with open(ROOT / "config" / "paths.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str) -> Path:
    """Đổi đường dẫn tương đối trong config thành đường dẫn tuyệt đối."""
    return ROOT / rel


def save_df(df: pd.DataFrame, path: Path) -> Path:
    """Lưu parquet (chuẩn của nhóm). Nếu máy chưa cài pyarrow -> cảnh báo và lưu tạm csv.gz."""
    path = Path(path)
    try:
        df.to_parquet(path, index=False)
        return path
    except ImportError:
        alt = path.with_suffix(".csv.gz")
        print("  !! Chưa cài pyarrow (pip install pyarrow) -> lưu tạm", alt.name)
        df.to_csv(alt, index=False)
        return alt


def load_df(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        return pd.read_parquet(path)
    alt = path.with_suffix(".csv.gz")
    return pd.read_csv(alt, parse_dates=[c for c in ["date_time", "open_time", "close_time"]
                                          if c in pd.read_csv(alt, nrows=0).columns])


def write_source_log(source, n_rows, start, end, note=""):
    """Ghi thêm 1 dòng vào source log (ngày crawl, số bản ghi, khoảng thời gian)."""
    path = p(load_paths()["logs"]["source_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["collected_at_utc", "source", "n_rows", "start", "end", "note"])
        w.writerow([datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), source, n_rows, start, end, note])

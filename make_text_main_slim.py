"""make_text_main_slim.py - Cắt text_clean -> text_main_slim (chỉ main sample).

Giữ doc_id, title, nlp_text cho in_main_sample=True.
Bản chính sentiment dùng title; nlp_text (title+description) = robustness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def main() -> None:
    import yaml
    root = project_root()
    paths = yaml.safe_load((root / "config" / "paths.yaml").read_text(encoding="utf-8")) or {}
    src = root / paths.get("text_clean_path", "data/interim/text_clean.parquet")
    out = root / paths.get("text_main_slim_path", "data/interim/text_main_slim.parquet")
    if not src.exists():
        sys.exit("Không thấy %s" % src)

    cols = ["doc_id", "title", "nlp_text", "in_main_sample"]
    df = pd.read_parquet(src, columns=cols)
    slim = df.loc[df["in_main_sample"], ["doc_id", "title", "nlp_text"]].reset_index(drop=True)
    assert len(slim) > 0
    out.parent.mkdir(parents=True, exist_ok=True)
    slim.to_parquet(out, index=False)

    log = {
        "src": str(src),
        "out": str(out),
        "rows_src": len(df),
        "rows_slim": len(slim),
        "src_mb": round(src.stat().st_size / 1e6, 2),
        "out_mb": round(out.stat().st_size / 1e6, 2),
        "cols": list(slim.columns),
        "note": "title = bản chính sentiment; nlp_text = robustness (title+description)",
    }
    log_path = out.with_suffix(".log.json")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Da luu %d dong -> %s (%.1f MB; src %.1f MB) cols=%s" % (
        len(slim), out, log["out_mb"], log["src_mb"], log["cols"]))


if __name__ == "__main__":
    main()

"""Kiểm sentiment_doc / sentiment_doc_fulltext (skip nếu chưa có file)."""
from pathlib import Path

import pandas as pd
import pytest

REQUIRED = {"doc_id", "model", "label", "p_pos", "p_neg", "p_neu", "text_variant"}


def _n_main(root: Path, paths: dict) -> int:
    text = pd.read_parquet(root / paths["text_clean_path"], columns=["in_main_sample"])
    return int(text["in_main_sample"].sum())


def _check_file(path: Path, expect_variant: str, n_main: int | None):
    df = pd.read_parquet(path)
    assert REQUIRED <= set(df.columns), set(df.columns)
    assert (df["text_variant"] == expect_variant).all(), df["text_variant"].unique().tolist()
    sums = df["p_pos"] + df["p_neg"] + df["p_neu"]
    assert (sums - 1.0).abs().max() < 1e-4
    for (m, v), g in df.groupby(["model", "text_variant"]):
        assert g["doc_id"].is_unique, (m, v)
        if n_main is not None:
            assert len(g) == n_main, (m, v, len(g), n_main)


def test_sentiment_title_main(paths):
    """Bản chính (title) — skip nếu Colab chưa trả file."""
    root = Path(__file__).resolve().parents[1]
    p = root / paths["sentiment_doc_path"]
    if not p.exists():
        pytest.skip(f"chưa có {p} (chạy Colab --text-col title)")
    n = _n_main(root, paths)
    _check_file(p, "title", n)


def test_sentiment_fulltext_robustness(paths):
    """Robustness nlp_text — file đổi tên từ Colab cũ."""
    root = Path(__file__).resolve().parents[1]
    key = "sentiment_doc_fulltext_path"
    if key not in paths:
        pytest.skip("paths thiếu sentiment_doc_fulltext_path")
    p = root / paths[key]
    if not p.exists():
        pytest.skip(f"chưa có {p}")
    n = _n_main(root, paths)
    _check_file(p, "nlp_text", n)


def test_sentiment_smoke_or_trial_ok(paths):
    """Smoke/trial có text_variant nếu tồn tại."""
    root = Path(__file__).resolve().parents[1]
    main = root / paths["sentiment_doc_path"]
    candidates = [
        main.with_name(main.stem + "_smoke" + main.suffix),
        root / "data/interim/sentiment_doc_smoke.parquet",
        root / "data/interim/sentiment_doc_trial500.parquet",
    ]
    found = [c for c in candidates if c.exists()]
    if not found:
        pytest.skip("không có smoke/trial")
    df = pd.read_parquet(found[0])
    if "text_variant" in df.columns:
        assert df["text_variant"].isin(["title", "nlp_text"]).all()
    sums = df["p_pos"] + df["p_neg"] + df["p_neu"]
    assert (sums - 1.0).abs().max() < 1e-4

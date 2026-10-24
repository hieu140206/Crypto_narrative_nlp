"""modeltable_io.py — tải bảng mô hình Spec A/B với chốt test.

Chỉ được đọc split='test' khi ALLOW_TEST=1.
Mọi script mô hình sau này phải dùng load_model_table (không đọc parquet trực tiếp).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


class TestSplitLockedError(PermissionError):
    """Raise khi gọi load_model_table(..., split='test') mà chưa ALLOW_TEST=1."""

    __test__ = False  # tránh pytest thu thập class tên Test*


def project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config").exists():
            return p
    return here.parent


def _load_paths(root: Path | None = None) -> dict:
    import yaml
    root = root or project_root()
    return yaml.safe_load((root / "config" / "paths.yaml").read_text(encoding="utf-8")) or {}


def model_table_path(
    spec: str,
    *,
    panel: str = "title",
    root: Path | None = None,
    paths: dict | None = None,
) -> Path:
    """Đường dẫn parquet theo spec ∈ {A,B} và panel ∈ {title, fulltext/nlp_text, noprice}."""
    root = root or project_root()
    paths = paths or _load_paths(root)
    spec = spec.upper()
    is_full = panel in ("fulltext", "nlp_text")
    is_noprice = panel == "noprice"
    if spec == "A":
        if is_noprice:
            key = "model_table_A_noprice_path"
        elif is_full:
            key = "model_table_A_fulltext_path"
        else:
            key = "model_table_A_path"
    elif spec == "B":
        if is_noprice:
            key = "model_table_B_noprice_path"
        elif is_full:
            key = "model_table_B_fulltext_path"
        else:
            key = "model_table_B_path"
    else:
        raise ValueError("spec phải là 'A' hoặc 'B', nhận %r" % spec)
    rel = paths.get(key)
    if not rel:
        raise KeyError("paths.yaml thiếu %s" % key)
    p = Path(rel)
    return p if p.is_absolute() else root / p


def load_model_table(
    spec: str,
    split: str,
    *,
    panel: str = "title",
    horizon: int | None = None,
    drop_embargo: bool = True,
    root: Path | None = None,
    paths: dict | None = None,
) -> pd.DataFrame:
    """Tải model_table_{spec} lọc theo split.

    - split='test' → raise TestSplitLockedError trừ khi os.environ['ALLOW_TEST']=='1'.
    - Spec B: nếu drop_embargo và horizon cho trước → bỏ hàng embargo_drop_{h}==True;
      nếu horizon=None → bỏ hàng nếu bất kỳ embargo_drop_* True.
    """
    split = str(split).lower()
    if split == "test" and os.environ.get("ALLOW_TEST") != "1":
        raise TestSplitLockedError(
            "split='test' bị khoá. Đặt ALLOW_TEST=1 chỉ khi chạy bước cuối (PREREG)."
        )
    if split not in ("train", "valid", "validation", "test"):
        raise ValueError("split phải ∈ {train, valid, test}, nhận %r" % split)
    if split == "validation":
        split = "valid"

    path = model_table_path(spec, panel=panel, root=root, paths=paths)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    out = df.loc[df["split"] == split].copy()

    if spec.upper() == "B" and drop_embargo:
        if horizon is not None:
            col = f"embargo_drop_{int(horizon)}"
            if col not in out.columns:
                raise KeyError("Thiếu cột %s trong %s" % (col, path))
            out = out.loc[~out[col].astype(bool)].copy()
        else:
            cols = [c for c in out.columns if c.startswith("embargo_drop_")]
            if cols:
                mask = out[cols].astype(bool).any(axis=1)
                out = out.loc[~mask].copy()
    return out.reset_index(drop=True)

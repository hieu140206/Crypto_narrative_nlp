"""modeltable_io.py — tải bảng mô hình Spec A/B với chốt test.

Chỉ được đọc split='test' khi ALLOW_TEST=1.
Mọi script mô hình sau này phải dùng load_model_table (không đọc parquet trực tiếp).

``split_override``: remap cửa sổ theo ngày (pseudo-test) mà không đọc split test thật.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


class TestSplitLockedError(PermissionError):
    """Raise khi gọi load_model_table(..., split='test') mà chưa ALLOW_TEST=1."""

    __test__ = False  # tránh pytest thu thập class tên Test*


# Pseudo-test: train / prior / eval dịch lùi; cấm mọi ngày ≥ HARD_DATE_CAP
PSEUDO_SPLIT_OVERRIDE: dict[str, tuple[str, str]] = {
    "train": ("2021-03-01", "2021-12-31"),
    "prior": ("2022-01-01", "2022-12-31"),
    "eval": ("2023-01-01", "2023-12-31"),
}
HARD_DATE_CAP = "2024-01-01"


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
    table_path: Path | str | None = None,
) -> pd.DataFrame:
    """Tải model_table_{spec} lọc theo split.

    - split='test' → raise TestSplitLockedError trừ khi os.environ['ALLOW_TEST']=='1'.
    - Spec B: nếu drop_embargo và horizon cho trước → bỏ hàng embargo_drop_{h}==True;
      nếu horizon=None → bỏ hàng nếu bất kỳ embargo_drop_* True.
    - table_path: override đường dẫn parquet (vd. LOSO / temp check); không đụng defaults.
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

    if table_path is not None:
        path = Path(table_path)
        if not path.is_absolute():
            path = (root or project_root()) / path
    else:
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


def _normalize_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.tz_localize(None).dt.normalize()


def assert_no_hard_cap_dates(df: pd.DataFrame, *, cap: str = HARD_DATE_CAP) -> None:
    """Raise nếu có date ≥ cap (mặc định 2024-01-01)."""
    if df is None or df.empty or "date" not in df.columns:
        return
    d = _normalize_dates(df["date"])
    cap_ts = pd.Timestamp(cap)
    bad = int((d >= cap_ts).sum())
    if bad:
        raise AssertionError(
            "Có %d dòng date ≥ %s (min_bad=%s max=%s) — cấm trong split_override/pseudo-test"
            % (bad, cap, d[d >= cap_ts].min(), d.max())
        )


def _slice_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    d = _normalize_dates(df["date"])
    m = (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return df.loc[m].copy()


def load_splits_with_override(
    spec: str,
    *,
    panel: str = "title",
    horizon: int | None = None,
    drop_embargo: bool = True,
    root: Path | None = None,
    paths: dict | None = None,
    split_override: dict[str, tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Trả ``{"train", "valid", "test"}``.

    - ``split_override is None``: hành vi thường (test cần ALLOW_TEST).
    - Có override: chỉ đọc train+valid thật, remap theo cửa sổ ngày thành
      train / valid(=prior) / test(=eval). Không bao giờ gọi split='test'.
      Assert không có date ≥ HARD_DATE_CAP.
    """
    if split_override is None:
        train = load_model_table(
            spec, "train", panel=panel, horizon=horizon,
            drop_embargo=drop_embargo, root=root, paths=paths,
        )
        valid = load_model_table(
            spec, "valid", panel=panel, horizon=horizon,
            drop_embargo=drop_embargo, root=root, paths=paths,
        )
        test = load_model_table(
            spec, "test", panel=panel, horizon=horizon,
            drop_embargo=drop_embargo, root=root, paths=paths,
        )
        return {"train": train, "valid": valid, "test": test}

    need = {"train", "prior", "eval"}
    if need - set(split_override):
        raise ValueError("split_override cần keys %s, nhận %s" % (need, sorted(split_override)))

    # Chỉ train + valid — không đụng test
    train_raw = load_model_table(
        spec, "train", panel=panel, horizon=horizon,
        drop_embargo=drop_embargo, root=root, paths=paths,
    )
    valid_raw = load_model_table(
        spec, "valid", panel=panel, horizon=horizon,
        drop_embargo=drop_embargo, root=root, paths=paths,
    )
    pool = pd.concat([train_raw, valid_raw], ignore_index=True)
    assert_no_hard_cap_dates(pool)

    train = _slice_window(pool, *split_override["train"]).assign(split="train")
    prior = _slice_window(pool, *split_override["prior"]).assign(split="valid")
    eval_ = _slice_window(pool, *split_override["eval"]).assign(split="test")
    for part in (train, prior, eval_):
        assert_no_hard_cap_dates(part)
        if part.empty:
            raise ValueError("Cửa sổ split_override rỗng sau lọc ngày")
    return {"train": train.reset_index(drop=True),
            "valid": prior.reset_index(drop=True),
            "test": eval_.reset_index(drop=True)}

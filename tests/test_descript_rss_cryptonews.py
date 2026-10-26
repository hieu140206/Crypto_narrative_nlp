"""Unit tests for descriptive RSS/cryptonews helpers (no confirmatory registry)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.descript_rss_cryptonews import (  # noqa: E402
    FROZEN_NARR_LF,
    corr_with_block_boot,
    narratives_yaml_hashes,
    residualize_dow,
    wilson_ci,
)


def test_wilson_ci_bounds_and_center():
    p, lo, hi = wilson_ci(50, 100)
    assert abs(p - 0.5) < 1e-12
    assert 0.0 <= lo < p < hi <= 1.0
    p0, lo0, hi0 = wilson_ci(0, 20)
    assert p0 == 0.0 and lo0 == 0.0 and hi0 > 0
    p1, lo1, hi1 = wilson_ci(20, 20)
    assert p1 == 1.0 and hi1 == 1.0 and lo1 < 1.0
    assert all(np.isnan(x) for x in wilson_ci(0, 0))


def test_narratives_hash_lf_matches_frozen():
    info = narratives_yaml_hashes(ROOT)
    assert info["sha256_lf"] == FROZEN_NARR_LF
    assert info["lf_matches_frozen"] is True
    assert len(info["sha256_raw"]) == 64
    assert len(info["sha256_crlf"]) == 64


def test_residualize_dow_orthogonal_to_dummies():
    dates = pd.date_range("2023-07-01", periods=70, freq="D")
    # construct y = 1 + 2*Tue + noise
    dow = dates.dayofweek
    y = 1.0 + 2.0 * (dow == 1).astype(float) + 0.01 * np.arange(len(dates))
    resid = residualize_dow(pd.Series(y), pd.Series(dates))
    assert resid.notna().sum() == len(dates)
    # mean residual ≈ 0
    assert abs(float(resid.mean())) < 1e-8


def test_corr_block_boot_reproducible():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    y = 0.6 * x + rng.normal(size=80) * 0.5
    a = corr_with_block_boot(x, y, method="pearson", n_boot=200, seed=42)
    b = corr_with_block_boot(x, y, method="pearson", n_boot=200, seed=42)
    assert a["corr"] == pytest.approx(b["corr"])
    assert a["ci_lo"] == pytest.approx(b["ci_lo"])
    assert a["ci_hi"] == pytest.approx(b["ci_hi"])
    assert a["n"] == 80
    s = corr_with_block_boot(x, y, method="spearman", n_boot=200, seed=42)
    assert -1 <= s["corr"] <= 1
    assert s["ci_lo"] <= s["corr"] <= s["ci_hi"] or (
        np.isnan(s["ci_lo"])  # degenerate
    )

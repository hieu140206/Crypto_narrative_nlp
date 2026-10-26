from pathlib import Path
import pandas as pd, pytest, yaml

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def params():
    return yaml.safe_load((ROOT / "config/params.yaml").read_text(encoding="utf-8"))

@pytest.fixture(scope="session")
def paths():
    return yaml.safe_load((ROOT / "config/paths.yaml").read_text(encoding="utf-8"))

def _load(paths, key, columns=None):
    p = ROOT / paths[key]
    if not p.exists():
        pytest.skip(f"chưa có {p}")
    return pd.read_parquet(p, columns=columns)

@pytest.fixture(scope="session")
def text(paths):
    return _load(paths, "text_clean_path")

@pytest.fixture(scope="session")
def panel(paths):
    return _load(paths, "daily_panel_path")

@pytest.fixture(scope="session")
def market(paths):
    return _load(paths, "market_daily_path")

@pytest.fixture(scope="session")
def fred(paths):
    return _load(paths, "fred_path")

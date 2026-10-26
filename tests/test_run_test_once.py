"""Tests for models/run_test_once.py — lock order, dry-run / pseudo isolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import run_test_once as rto  # noqa: E402
from modeltable_io import (  # noqa: E402
    HARD_DATE_CAP,
    PSEUDO_SPLIT_OVERRIDE,
    TestSplitLockedError,
    assert_no_hard_cap_dates,
)


def _fake_summary(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"asset": ["BTC"] * n, "h": list(range(1, n + 1))})


def test_dry_run_no_registry_no_lock(tmp_path, monkeypatch):
    """dry-run does not write main registry and does not create TEST_LOCK."""
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "tables").mkdir(parents=True)
    reg = tmp_path / "outputs" / "tables" / "test_registry.csv"
    reg.write_text("timestamp,script\n2020,old\n", encoding="utf-8")
    before = reg.read_text(encoding="utf-8")

    calls = {"har_register": None, "ml_register": None, "har_stage": None, "ml_stage": None}

    def fake_har(**kwargs):
        calls["har_register"] = kwargs.get("register")
        calls["har_stage"] = kwargs.get("stage")
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "har_text_summary.csv").write_text("asset,h\nBTC,1\n", encoding="utf-8")
        return _fake_summary()

    def fake_ml(**kwargs):
        calls["ml_register"] = kwargs.get("register")
        calls["ml_stage"] = kwargs.get("stage")
        out = kwargs["out_dir"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "ml_summary_valid_gv2_mlevel.csv").write_text(
            "asset,h\nBTC,1\n", encoding="utf-8"
        )
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", fake_har)
    monkeypatch.setattr(rto, "run_family5_ml", fake_ml)

    result = rto.run_pipeline(dry_run=True, root=tmp_path)

    assert result["dry_run"] is True
    assert result["stage"] == "valid"
    assert result["lock_written"] is False
    assert calls["har_register"] is False
    assert calls["ml_register"] is False
    assert calls["har_stage"] == "valid"
    assert calls["ml_stage"] == "valid"
    assert not (tmp_path / "outputs" / "TEST_LOCK").exists()
    assert reg.read_text(encoding="utf-8") == before


def test_lock_blocks_second_real_run(tmp_path, monkeypatch):
    """existing TEST_LOCK → exit code 2; no bypass flag."""
    monkeypatch.setenv("ALLOW_TEST", "1")
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    lock = tmp_path / "outputs" / "TEST_LOCK"
    lock.parent.mkdir(parents=True)
    lock.write_text('{"status": "started"}\n', encoding="utf-8")

    called = {"n": 0}

    def boom(**kwargs):
        called["n"] += 1
        raise AssertionError("should not run models when locked")

    monkeypatch.setattr(rto, "run_family5_har_text", boom)
    monkeypatch.setattr(rto, "run_family5_ml", boom)

    with pytest.raises(SystemExit) as ei:
        rto.run_pipeline(dry_run=False, root=tmp_path)
    assert ei.value.code == rto.EXIT_LOCKED == 2
    assert called["n"] == 0


def test_lock_started_before_test_load(tmp_path, monkeypatch):
    """(a) status=started tồn tại TRƯỚC khi hàm nạp/chạy đọc test được gọi."""
    monkeypatch.setenv("ALLOW_TEST", "1")
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    monkeypatch.setattr(rto, "params_sha256", lambda root=None: "abc123")
    monkeypatch.setattr(rto, "git_head", lambda root=None: "deadbeef")

    order: list[str] = []

    def fake_har(**kwargs):
        order.append("har")
        lock = tmp_path / "outputs" / "TEST_LOCK"
        assert lock.exists(), "lock must exist before HAR runs"
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["status"] == "started"
        assert kwargs["stage"] == "test"
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["out_dir"] / "har_out.csv").write_text("x\n", encoding="utf-8")
        return _fake_summary()

    def fake_ml(**kwargs):
        order.append("ml")
        lock = tmp_path / "outputs" / "TEST_LOCK"
        assert lock.exists()
        assert json.loads(lock.read_text(encoding="utf-8"))["status"] in ("started", "completed")
        (kwargs["out_dir"] / "ml_out.csv").write_text("y\n", encoding="utf-8")
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", fake_har)
    monkeypatch.setattr(rto, "run_family5_ml", fake_ml)

    result = rto.run_pipeline(dry_run=False, root=tmp_path)
    assert order == ["har", "ml"]
    assert result["lock_written"] is True
    payload = json.loads((tmp_path / "outputs" / "TEST_LOCK").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["params_sha256"] == "abc123"
    assert "har_out.csv" in payload["output_files"]


def test_mid_failure_leaves_started_lock(tmp_path, monkeypatch):
    """(b) lỗi giữa chừng → khóa status=started; lần 2 exit 2."""
    monkeypatch.setenv("ALLOW_TEST", "1")
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    monkeypatch.setattr(rto, "params_sha256", lambda root=None: "abc")
    monkeypatch.setattr(rto, "git_head", lambda root=None: "g")

    def fail_har(**kwargs):
        lock = tmp_path / "outputs" / "TEST_LOCK"
        assert lock.exists()
        assert json.loads(lock.read_text(encoding="utf-8"))["status"] == "started"
        raise RuntimeError("boom mid-run")

    monkeypatch.setattr(rto, "run_family5_har_text", fail_har)
    monkeypatch.setattr(rto, "run_family5_ml", lambda **k: _fake_summary())

    with pytest.raises(RuntimeError, match="boom"):
        rto.run_pipeline(dry_run=False, root=tmp_path)

    lock = tmp_path / "outputs" / "TEST_LOCK"
    assert lock.exists()
    assert json.loads(lock.read_text(encoding="utf-8"))["status"] == "started"

    with pytest.raises(SystemExit) as ei:
        rto.run_pipeline(dry_run=False, root=tmp_path)
    assert ei.value.code == rto.EXIT_LOCKED


def test_real_run_writes_lock(tmp_path, monkeypatch):
    """Successful real run writes TEST_LOCK completed; out_dir=outputs/test."""
    monkeypatch.setenv("ALLOW_TEST", "1")
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    monkeypatch.setattr(rto, "params_sha256", lambda root=None: "abc123")
    monkeypatch.setattr(rto, "git_head", lambda root=None: "deadbeef")
    expected_out = (tmp_path / "outputs" / "test").resolve()
    expected_reg = (tmp_path / "outputs" / "tables" / "test_registry.csv").resolve()

    def fake_har(**kwargs):
        assert kwargs["stage"] == "test"
        assert kwargs["register"] is True
        assert Path(kwargs["out_dir"]).resolve() == expected_out
        assert Path(kwargs["registry_path"]).resolve() == expected_reg
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["out_dir"] / "har_out.csv").write_text("x\n", encoding="utf-8")
        return _fake_summary()

    def fake_ml(**kwargs):
        assert kwargs["stage"] == "test"
        assert Path(kwargs["out_dir"]).resolve() == expected_out
        assert Path(kwargs["registry_path"]).resolve() == expected_reg
        (kwargs["out_dir"] / "ml_out.csv").write_text("y\n", encoding="utf-8")
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", fake_har)
    monkeypatch.setattr(rto, "run_family5_ml", fake_ml)

    result = rto.run_pipeline(dry_run=False, root=tmp_path)
    assert result["lock_written"] is True
    assert Path(result["out_dir"]).resolve() == expected_out
    payload = json.loads((tmp_path / "outputs" / "TEST_LOCK").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["stage"] == "test"


def test_real_mode_out_dir_is_outputs_test(tmp_path, monkeypatch):
    """(a) mode real chọn out_dir = outputs/test (không phải tables)."""
    monkeypatch.setenv("ALLOW_TEST", "1")
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    monkeypatch.setattr(rto, "params_sha256", lambda root=None: "x")
    monkeypatch.setattr(rto, "git_head", lambda root=None: "y")
    seen = {}

    def capture(**kwargs):
        seen["out_dir"] = Path(kwargs["out_dir"]).resolve()
        seen["registry_path"] = Path(kwargs["registry_path"]).resolve()
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["out_dir"] / "f.csv").write_text("1\n", encoding="utf-8")
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", capture)
    monkeypatch.setattr(rto, "run_family5_ml", capture)
    rto.run_pipeline(mode="real", root=tmp_path)
    assert seen["out_dir"] == (tmp_path / "outputs" / "test").resolve()
    assert seen["registry_path"] == (tmp_path / "outputs" / "tables" / "test_registry.csv").resolve()
    assert "tables" not in seen["out_dir"].parts or seen["out_dir"].name == "test"
    assert seen["out_dir"].name == "test"


def test_output_writes_use_out_dir_not_hardcoded_tables():
    """(b) har_text / ml_forecast: mọi to_csv/write_text kết quả dùng out_dir (không hardcode tables)."""
    import ast

    for rel in ("models/har_text.py", "models/ml_forecast.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # tìm chuỗi literal chứa outputs/tables trong Call to_csv / write_text
        bad = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name not in ("to_csv", "write_text", "write_bytes"):
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        if "outputs/tables" in sub.value.replace("\\", "/"):
                            bad.append((rel, name, sub.value, getattr(node, "lineno", "?")))
        assert not bad, "hardcoded outputs/tables in write calls: %s" % bad

    # đảm bảo đường ghi kết quả gắn với biến out_dir (mẫu phổ biến trong code)
    for rel, needles in (
        ("models/har_text.py", ["out_dir / f\"har_text_", "out_dir / \"har_text_summary"]),
        ("models/ml_forecast.py", ["out_dir / f\"ml_detail_", "out_dir / f\"ml_summary_", "out_dir / f\"ml_"]),
    ):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for n in needles:
            assert n in src, "%s missing pattern %r" % (rel, n)


def test_dry_and_pseudo_do_not_touch_tables_summaries(tmp_path, monkeypatch):
    """(c) dry-run / pseudo-test không đụng outputs/tables summary files."""
    import hashlib
    import time

    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    tables = tmp_path / "outputs" / "tables"
    tables.mkdir(parents=True)
    har = tables / "har_text_summary.csv"
    ml = tables / "ml_summary_valid_gv2_mlevel.csv"
    har.write_text("asset,h\nBTC,1\n", encoding="utf-8")
    ml.write_text("asset,h\nBTC,1\n", encoding="utf-8")
    time.sleep(0.05)

    def _snap(p: Path):
        raw = p.read_bytes()
        return p.stat().st_mtime_ns, hashlib.sha256(raw).hexdigest()

    before = {p.name: _snap(p) for p in (har, ml)}

    def stub(**kwargs):
        assert kwargs["register"] is False
        assert kwargs.get("registry_path") is None
        out = Path(kwargs["out_dir"])
        assert out.resolve() != tables.resolve()
        assert "tables" not in out.resolve().parts or out.name in ("dryrun", "pseudotest")
        out.mkdir(parents=True, exist_ok=True)
        (out / "local.csv").write_text("ok\n", encoding="utf-8")
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", stub)
    monkeypatch.setattr(rto, "run_family5_ml", stub)

    rto.run_pipeline(mode="dry_run", root=tmp_path)
    rto.run_pipeline(mode="pseudo_test", root=tmp_path)

    after = {p.name: _snap(p) for p in (har, ml)}
    assert before == after
    assert not (tmp_path / "outputs" / "TEST_LOCK").exists()


def test_dry_run_never_loads_test_split(tmp_path, monkeypatch):
    """dry-run only loads train+valid; never split=test."""
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)

    loaded_splits: list[str] = []

    def fake_load(spec, split, **kwargs):
        loaded_splits.append(split)
        if split == "test":
            raise TestSplitLockedError("test locked")
        n = 8
        return pd.DataFrame({
            "date": pd.date_range("2021-06-01", periods=n, freq="D"),
            "asset": "BTC",
            "split": split,
        })

    import models.har_text as ht
    import models.ml_forecast as mf

    monkeypatch.setattr(ht, "load_model_table", fake_load)
    monkeypatch.setattr(mf, "load_model_table", fake_load)

    def stub_har(**kwargs):
        assert kwargs["stage"] == "valid"
        assert kwargs["register"] is False
        assert kwargs.get("split_override") is None
        ht.load_model_table("B", "train", horizon=1, drop_embargo=True, root=tmp_path)
        ht.load_model_table("B", "valid", horizon=1, drop_embargo=True, root=tmp_path)
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        return _fake_summary()

    def stub_ml(**kwargs):
        assert kwargs["stage"] == "valid"
        assert kwargs["register"] is False
        mf.load_model_table("B", "train", horizon=1, drop_embargo=True, root=tmp_path)
        mf.load_model_table("B", "valid", horizon=1, drop_embargo=True, root=tmp_path)
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", stub_har)
    monkeypatch.setattr(rto, "run_family5_ml", stub_ml)

    rto.run_pipeline(dry_run=True, root=tmp_path)

    assert "test" not in loaded_splits
    assert set(loaded_splits) <= {"train", "valid"}
    assert not (tmp_path / "outputs" / "TEST_LOCK").exists()


def test_pseudo_test_no_lock_no_registry_no_hard_cap(tmp_path, monkeypatch):
    """(c) --pseudo-test: không khóa/registry; không nạp date ≥ 2024; nhánh test+override."""
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    (tmp_path / "outputs" / "tables").mkdir(parents=True)
    reg = tmp_path / "outputs" / "tables" / "test_registry.csv"
    reg.write_text("timestamp,script\n2020,old\n", encoding="utf-8")
    before = reg.read_text(encoding="utf-8")

    max_dates: list[pd.Timestamp] = []

    def fake_har(**kwargs):
        assert kwargs["stage"] == "test"
        assert kwargs["register"] is False
        assert kwargs["split_override"] == PSEUDO_SPLIT_OVERRIDE
        # giả lập dữ liệu đã nạp: chỉ ≤ 2023
        df = pd.DataFrame({
            "date": pd.to_datetime(["2021-06-01", "2022-06-01", "2023-06-01"]),
        })
        assert_no_hard_cap_dates(df)
        max_dates.append(df["date"].max())
        kwargs["out_dir"].mkdir(parents=True, exist_ok=True)
        return _fake_summary()

    def fake_ml(**kwargs):
        assert kwargs["stage"] == "test"
        assert kwargs["register"] is False
        assert kwargs["split_override"] == PSEUDO_SPLIT_OVERRIDE
        df = pd.DataFrame({"date": pd.to_datetime(["2023-12-31"])})
        assert_no_hard_cap_dates(df)
        max_dates.append(df["date"].max())
        return _fake_summary()

    monkeypatch.setattr(rto, "run_family5_har_text", fake_har)
    monkeypatch.setattr(rto, "run_family5_ml", fake_ml)

    result = rto.run_pipeline(mode="pseudo_test", root=tmp_path)
    assert result["pseudo_test"] is True
    assert result["lock_written"] is False
    assert not (tmp_path / "outputs" / "TEST_LOCK").exists()
    assert reg.read_text(encoding="utf-8") == before
    assert all(d < pd.Timestamp(HARD_DATE_CAP) for d in max_dates)


def test_assert_no_hard_cap_dates():
    ok = pd.DataFrame({"date": pd.to_datetime(["2023-12-31"])})
    assert_no_hard_cap_dates(ok)
    bad = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    with pytest.raises(AssertionError):
        assert_no_hard_cap_dates(bad)


def test_dry_run_cli_rejects_non_valid():
    with pytest.raises(SystemExit):
        rto.main(["--dry-run", "test"])


def test_real_run_requires_allow_test(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_TEST", raising=False)
    monkeypatch.setattr(rto, "project_root", lambda: tmp_path)
    with pytest.raises(SystemExit) as ei:
        rto.run_pipeline(dry_run=False, root=tmp_path)
    assert ei.value.code != rto.EXIT_LOCKED

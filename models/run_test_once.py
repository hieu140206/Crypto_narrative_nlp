"""Chạy một lần khoá trên tập test: Family 5 (HAR+text) + Family 5-ML.

Dry-run: ``--dry-run valid`` — pipeline stage='valid', ghi ``outputs/dryrun/``,
không tạo TEST_LOCK, không ghi registry.

Pseudo-test: ``--pseudo-test`` — đúng nhánh stage='test' (prior + CV train+prior)
nhưng cửa sổ dịch lùi (train 2021 / prior 2022 / eval 2023). Không đọc date≥2024,
không registry, không khóa. Ghi ``outputs/pseudotest/``.

Lần thật (cần ALLOW_TEST=1; KHÔNG có cờ bypass):
  - Từ chối nếu ``outputs/TEST_LOCK`` đã tồn tại (exit 2).
  - Ghi khóa status=started TRƯỚC khi đọc bất kỳ dòng test nào.
  - Thành công → status=completed + danh sách output.
  - Lỗi giữa chừng → giữ status=started (không xoá khoá).

Seed 42.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modeltable_io import PSEUDO_SPLIT_OVERRIDE  # noqa: E402
from models.har_baseline import project_root  # noqa: E402
from models.har_text import TEXT_MODELS, run_all as run_har_text  # noqa: E402
from models.ml_forecast import params_sha256, run_evaluation  # noqa: E402

SEED = 42
LOCK_NAME = "TEST_LOCK"
GRID = "v2"
MACRO_MODE = "level"
EXIT_LOCKED = 2


def lock_path(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "outputs" / LOCK_NAME


def git_head(root: Path | None = None) -> str:
    root = root or project_root()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "UNKNOWN"


def list_output_files(out_dir: Path) -> list[str]:
    if not out_dir.exists():
        return []
    return sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file()
    )


def _rel_out(root: Path, out_dir: Path) -> str:
    try:
        return str(out_dir.relative_to(root).as_posix())
    except ValueError:
        return str(out_dir)


def write_lock_started(*, root: Path, out_dir: Path) -> Path:
    """Ghi TEST_LOCK status=started — gọi TRƯỚC khi đọc test."""
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "started",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params_sha256": params_sha256(root),
        "git_head": git_head(root),
        "grid": GRID,
        "macro_mode": MACRO_MODE,
        "seed": SEED,
        "out_dir": _rel_out(root, out_dir),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_lock_completed(*, root: Path, out_dir: Path, stage: str = "test") -> Path:
    """Cập nhật TEST_LOCK → completed + danh sách file."""
    path = lock_path(root)
    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
    completed = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload.update({
        "status": "completed",
        "completed_utc": completed,
        "params_sha256": params_sha256(root),
        "git_head": git_head(root),
        "grid": GRID,
        "macro_mode": MACRO_MODE,
        "seed": SEED,
        "stage": stage,
        "out_dir": _rel_out(root, out_dir),
        "output_files": list_output_files(out_dir),
    })
    if "timestamp_utc" not in payload:
        payload["timestamp_utc"] = completed
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_lock(*, root: Path, out_dir: Path, stage: str = "test") -> Path:
    """Tương thích cũ: ghi completed (giả sử started đã có hoặc ghi mới)."""
    return write_lock_completed(root=root, out_dir=out_dir, stage=stage)


def main_registry_path(root: Path | None = None) -> Path:
    """Registry chính luôn nằm ở outputs/tables/test_registry.csv."""
    root = root or project_root()
    return root / "outputs" / "tables" / "test_registry.csv"


def run_family5_har_text(
    *,
    stage: str,
    out_dir: Path,
    root: Path,
    register: bool,
    panel: str = "title",
    split_override: dict[str, tuple[str, str]] | None = None,
    registry_path: Path | None = None,
) -> object:
    """9 phép PREREG / asset (không gồm M_att_des_dow)."""
    return run_har_text(
        root=root,
        out_dir=out_dir,
        panel=panel,
        seed=SEED,
        models=dict(TEXT_MODELS),
        include_deseason_robust=False,
        register=register,
        stage=stage,
        split_override=split_override,
        registry_path=registry_path,
    )


def run_family5_ml(
    *,
    stage: str,
    out_dir: Path,
    root: Path,
    register: bool,
    panel: str = "title",
    split_override: dict[str, tuple[str, str]] | None = None,
    registry_path: Path | None = None,
) -> object:
    return run_evaluation(
        stage=stage,
        root=root,
        out_dir=out_dir,
        panel=panel,
        register=register,
        grid=GRID,
        macro_mode=MACRO_MODE,
        split_override=split_override,
        registry_path=registry_path,
    )


def run_pipeline(
    *,
    dry_run: bool | None = None,
    mode: str | None = None,
    root: Path | None = None,
    panel: str = "title",
) -> dict:
    """Chạy pipeline. Truyền ``dry_run=`` (tương thích) hoặc ``mode=``.

    mode ∈ {'real','dry_run','pseudo_test'}.
    """
    root = root or project_root()
    if mode is None:
        if dry_run is None:
            raise TypeError("cần dry_run= hoặc mode=")
        mode = "dry_run" if dry_run else "real"
    mode = str(mode).lower()

    if mode == "dry_run":
        stage = "valid"
        out_dir = root / "outputs" / "dryrun"
        register = False
        split_override = None
        if os.environ.get("ALLOW_TEST") == "1":
            print("NOTE: dry-run ignores ALLOW_TEST; stage stays 'valid'", flush=True)
    elif mode == "pseudo_test":
        stage = "test"
        out_dir = root / "outputs" / "pseudotest"
        register = False
        split_override = dict(PSEUDO_SPLIT_OVERRIDE)
    elif mode == "real":
        stage = "test"
        # Kết quả test → outputs/test; registry vẫn append vào tables/
        out_dir = root / "outputs" / "test"
        register = True
        split_override = None
        if os.environ.get("ALLOW_TEST") != "1":
            raise SystemExit(
                "Real test run requires ALLOW_TEST=1 (and no existing TEST_LOCK)."
            )
        lp = lock_path(root)
        if lp.exists():
            print(
                "REFUSE: %s already exists — test already run / in progress." % lp,
                file=sys.stderr,
            )
            raise SystemExit(EXIT_LOCKED)
    else:
        raise ValueError("mode ∈ {real,dry_run,pseudo_test}, nhận %r" % mode)

    # Registry chính chỉ khi register=True (real); dry/pseudo không đụng tables/
    reg_path = main_registry_path(root) if register else None

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "=== run_test_once | mode=%s stage=%s out_dir=%s register=%s override=%s ==="
        % (mode, stage, out_dir, register, bool(split_override)),
        flush=True,
    )

    lock_started = False
    if mode == "real":
        write_lock_started(root=root, out_dir=out_dir)
        lock_started = True
        print("TEST_LOCK started ->", lock_path(root), flush=True)

    try:
        har_sum = run_family5_har_text(
            stage=stage, out_dir=out_dir, root=root, register=register,
            panel=panel, split_override=split_override, registry_path=reg_path,
        )
        ml_sum = run_family5_ml(
            stage=stage, out_dir=out_dir, root=root, register=register,
            panel=panel, split_override=split_override, registry_path=reg_path,
        )
    except Exception:
        if lock_started:
            print(
                "ERROR: pipeline failed after TEST_LOCK status=started. "
                "Khóa KHÔNG bị xoá — cần quyết định thủ công "
                "(giữ khóa / xoá sau khi audit).",
                file=sys.stderr,
            )
            traceback.print_exc()
        raise

    result = {
        "mode": mode,
        "dry_run": mode == "dry_run",
        "pseudo_test": mode == "pseudo_test",
        "stage": stage,
        "out_dir": str(out_dir),
        "register": register,
        "har_n_rows": int(len(har_sum)) if har_sum is not None else 0,
        "ml_n_rows": int(len(ml_sum)) if ml_sum is not None else 0,
        "lock_written": False,
        "lock_status": None,
    }

    if mode == "real":
        write_lock_completed(root=root, out_dir=out_dir, stage=stage)
        result["lock_written"] = True
        result["lock_status"] = "completed"
        print("TEST_LOCK completed ->", lock_path(root), flush=True)
    else:
        print("%s complete; no TEST_LOCK, register=False" % mode, flush=True)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Locked one-shot Family 5 + ML test / dry-run / pseudo-test"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--dry-run",
        nargs="?",
        const="valid",
        default=None,
        metavar="STAGE",
        help="stage=valid → outputs/dryrun/; no lock; no registry.",
    )
    g.add_argument(
        "--pseudo-test",
        action="store_true",
        help="Nhánh stage=test với split_override dịch lùi → outputs/pseudotest/; "
             "no lock; no registry; không đọc date≥2024.",
    )
    p.add_argument("--panel", default="title", choices=["title", "fulltext", "nlp_text"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.pseudo_test:
        run_pipeline(mode="pseudo_test", panel=args.panel)
        return
    if args.dry_run is not None:
        if str(args.dry_run).lower() != "valid":
            raise SystemExit("--dry-run only supports 'valid' (got %r)" % args.dry_run)
        run_pipeline(mode="dry_run", panel=args.panel)
        return
    run_pipeline(mode="real", panel=args.panel)


if __name__ == "__main__":
    main()

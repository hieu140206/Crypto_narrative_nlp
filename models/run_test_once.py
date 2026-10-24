"""Chạy một lần khoá trên tập test: Family 5 (HAR+text) + Family 5-ML.

Dry-run: ``--dry-run valid`` — pipeline đầy đủ với stage='valid', ghi
``outputs/dryrun/``, không tạo TEST_LOCK, không ghi registry.

Lần thật (KHÔNG chạy trừ khi ALLOW_TEST=1 và người dùng chủ động):
  - Từ chối nếu ``outputs/TEST_LOCK`` đã tồn tại (exit 2).
  - Sau khi thành công: ghi TEST_LOCK (timestamp, sha256 params, git HEAD, files).
  - Không có cờ bypass khoá.

Seed 42. Không đọc test khi dry-run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main_registry_path(root: Path | None = None) -> Path:
    root = root or project_root()
    return root / "outputs" / "tables" / "test_registry.csv"


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
    files = sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file()
    )
    return files


def write_lock(
    *,
    root: Path,
    out_dir: Path,
    stage: str,
) -> Path:
    path = lock_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "params_sha256": params_sha256(root),
        "git_head": git_head(root),
        "stage": stage,
        "grid": GRID,
        "macro_mode": MACRO_MODE,
        "seed": SEED,
        "out_dir": (
            str(out_dir.relative_to(root).as_posix())
            if root in out_dir.parents or out_dir == root
            else str(out_dir)
        ),
        "output_files": list_output_files(out_dir),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def run_family5_har_text(
    *,
    stage: str,
    out_dir: Path,
    root: Path,
    register: bool,
    panel: str = "title",
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
    )


def run_family5_ml(
    *,
    stage: str,
    out_dir: Path,
    root: Path,
    register: bool,
    panel: str = "title",
) -> object:
    return run_evaluation(
        stage=stage,
        root=root,
        out_dir=out_dir,
        panel=panel,
        register=register,
        grid=GRID,
        macro_mode=MACRO_MODE,
    )


def run_pipeline(
    *,
    dry_run: bool,
    root: Path | None = None,
    panel: str = "title",
) -> dict:
    root = root or project_root()
    if dry_run:
        stage = "valid"
        out_dir = root / "outputs" / "dryrun"
        register = False
        # Dry-run không bao giờ đụng TEST_LOCK / registry chính
        if os.environ.get("ALLOW_TEST") == "1":
            # vẫn an toàn: stage=valid không đọc test; cảnh báo nhẹ
            print("NOTE: dry-run ignores ALLOW_TEST; stage stays 'valid'", flush=True)
    else:
        stage = "test"
        out_dir = root / "outputs" / "tables"
        register = True
        if os.environ.get("ALLOW_TEST") != "1":
            raise SystemExit(
                "Real test run requires ALLOW_TEST=1 (and no existing TEST_LOCK)."
            )
        lp = lock_path(root)
        if lp.exists():
            print("REFUSE: %s already exists — test already run." % lp, file=sys.stderr)
            raise SystemExit(EXIT_LOCKED)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "=== run_test_once | dry_run=%s stage=%s out_dir=%s register=%s ==="
        % (dry_run, stage, out_dir, register),
        flush=True,
    )

    har_sum = run_family5_har_text(
        stage=stage, out_dir=out_dir, root=root, register=register, panel=panel
    )
    ml_sum = run_family5_ml(
        stage=stage, out_dir=out_dir, root=root, register=register, panel=panel
    )

    result = {
        "dry_run": dry_run,
        "stage": stage,
        "out_dir": str(out_dir),
        "register": register,
        "har_n_rows": int(len(har_sum)) if har_sum is not None else 0,
        "ml_n_rows": int(len(ml_sum)) if ml_sum is not None else 0,
        "lock_written": False,
    }

    if not dry_run:
        write_lock(root=root, out_dir=out_dir, stage=stage)
        result["lock_written"] = True
        print("TEST_LOCK written ->", lock_path(root), flush=True)
    else:
        print("dry-run complete; no TEST_LOCK, register=False", flush=True)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Locked one-shot Family 5 + Family 5-ML test (or --dry-run valid)"
    )
    p.add_argument(
        "--dry-run",
        nargs="?",
        const="valid",
        default=None,
        metavar="STAGE",
        help="Run pipeline with stage=valid (only 'valid' supported). "
             "Writes outputs/dryrun/; no TEST_LOCK; no registry.",
    )
    p.add_argument("--panel", default="title", choices=["title", "fulltext", "nlp_text"])
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run is not None:
        if str(args.dry_run).lower() != "valid":
            raise SystemExit("--dry-run only supports 'valid' (got %r)" % args.dry_run)
        run_pipeline(dry_run=True, panel=args.panel)
        return
    run_pipeline(dry_run=False, panel=args.panel)


if __name__ == "__main__":
    main()

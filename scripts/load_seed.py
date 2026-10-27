#!/usr/bin/env python3
"""Load the anonymised demo dataset into a storage volume.

The repository ships ``seed/demo.db`` and ``seed/evidence/``: three completed
jobs whose plate numbers are synthetic and whose evidence images have the plate
pixels blurred. This script installs that dataset into the application storage
directory so the dashboard has something to show without exposing real data.

**Stop the application first.** SQLite runs in WAL mode, and replacing the
database under a live process can corrupt the volume:

    docker compose stop app

Then, from the repository root:

    docker run --rm \
        -v "$PWD/seed:/seed:ro" \
        -v traffic-alpr_app_data:/data \
        --entrypoint python \
        traffic-alpr:gpu /app/scripts/load_seed.py --seed /seed --force

The image runs as uid 10001, the same user that owns the volume, so no root
step is needed. On a bare-metal install point ``--storage`` at the directory
that contains ``violations.db``.

The loader refuses to overwrite an existing database unless ``--force`` is
given, deletes stale ``-wal``/``-shm`` files, copies the redacted evidence
images, and prints a short verification summary.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import NoReturn


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _verify(database: Path, evidence_dir: Path) -> None:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        jobs = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        violations = connection.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        orphans = connection.execute(
            "SELECT COUNT(*) FROM violations v "
            "LEFT JOIN jobs j ON v.job_id = j.id WHERE j.id IS NULL"
        ).fetchone()[0]
        missing = 0
        for (relative,) in connection.execute(
            "SELECT proof_image_path FROM violations WHERE proof_image_path IS NOT NULL"
        ):
            if relative and not (evidence_dir / Path(relative).name).exists():
                missing += 1
    finally:
        connection.close()

    print(f"  jobs       : {jobs}")
    print(f"  violations : {violations}")
    print(f"  orphans    : {orphans}")
    print(f"  missing evidence files: {missing}")
    if jobs == 0 or violations == 0 or orphans or missing:
        _fail("seed verification failed")


def main() -> None:
    default_seed = Path(__file__).resolve().parent.parent / "seed"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=default_seed,
                        help="directory containing demo.db and evidence/")
    parser.add_argument("--storage", type=Path,
                        default=Path(os.getenv("STORAGE_DIR", "/data")),
                        help="application storage directory (default: $STORAGE_DIR or /data)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing violations.db")
    args = parser.parse_args()

    source_db = args.seed / "demo.db"
    source_evidence = args.seed / "evidence"
    if not source_db.is_file():
        _fail(f"{source_db} not found")
    if not source_evidence.is_dir():
        _fail(f"{source_evidence} not found")

    args.storage.mkdir(parents=True, exist_ok=True)
    target_db = args.storage / "violations.db"
    target_evidence = args.storage / "evidence"
    target_evidence.mkdir(parents=True, exist_ok=True)

    if target_db.exists() and not args.force:
        _fail(
            f"{target_db} already exists. Stop the app and re-run with --force "
            "if you really want to replace it."
        )

    # Replace atomically so a partial copy never becomes the live database.
    temporary = target_db.with_suffix(".db.loading")
    shutil.copy2(source_db, temporary)
    for suffix in ("-wal", "-shm"):
        Path(f"{target_db}{suffix}").unlink(missing_ok=True)
    os.replace(temporary, target_db)

    copied = 0
    for image in sorted(source_evidence.glob("*.jpg")):
        shutil.copy2(image, target_evidence / image.name)
        copied += 1

    print(f"seed installed into {args.storage}")
    print(f"  database  : {target_db}")
    print(f"  evidence  : {copied} images")
    _verify(target_db, target_evidence)
    print("\nStart the app again and open the dashboard:")
    print("  docker compose start app")


if __name__ == "__main__":
    main()

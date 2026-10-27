"""Backup database SQLite + evidence dengan aman (WAL-friendly).

Jalankan di dalam container:

    docker compose exec app python scripts/backup_db.py

Hasil default: /data/backups/<timestamp>/violations.db + evidence.tar.gz
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from config import settings


def _sqlite_path() -> Path:
    """Ambil path file SQLite dari DATABASE_URL."""
    url = settings.database_url
    if not url.startswith("sqlite"):
        raise SystemExit(
            f"Script ini hanya untuk SQLite, DATABASE_URL saat ini: {url}"
        )
    raw = url.split("///", 1)[-1]
    path = Path(unquote(raw))
    if not path.is_absolute():
        path = settings.base_dir / path
    return path


def backup_database(source: Path, target: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Database tidak ditemukan: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3.Connection.backup() memakai backup API resmi: konsisten
    # meskipun ada writer lain (mode WAL), tidak seperti `cp`.
    with sqlite3.connect(str(source), timeout=30) as src:
        with sqlite3.connect(str(target)) as dst:
            src.backup(dst)
    print(f"[ok] database  -> {target} ({target.stat().st_size / 1024:.0f} KB)")


def backup_evidence(evidence_dir: Path, target: Path) -> None:
    if not evidence_dir.is_dir():
        print(f"[skip] folder evidence tidak ada: {evidence_dir}")
        return

    files = [path for path in evidence_dir.rglob("*") if path.is_file()]
    if not files:
        print(f"[skip] folder evidence kosong: {evidence_dir}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        archive.add(evidence_dir, arcname="evidence")
    print(
        f"[ok] evidence  -> {target} "
        f"({len(files)} file, {target.stat().st_size / (1024 * 1024):.1f} MB)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.storage_dir / "backups",
        help="Folder tujuan backup (default: <storage>/backups).",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Hanya backup database, lewati arsip evidence.",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = args.output / stamp

    print(f"Storage : {settings.storage_dir}")
    print(f"Tujuan  : {destination}")

    backup_database(_sqlite_path(), destination / "violations.db")
    if not args.no_evidence:
        backup_evidence(settings.evidence_dir, destination / "evidence.tar.gz")

    print("Backup selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

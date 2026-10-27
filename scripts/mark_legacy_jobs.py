"""Tandai job lama (era MOCK_OCR) sebagai legacy/simulasi tanpa menghapus data.

Aturan:
- Job yang belum punya snapshot mode dianggap `legacy`.
- Pelanggaran dengan plat tidak kosong pada job legacy -> `plate_source=simulated`.
- Pelanggaran tanpa plat (`TIDAK TERBACA`) -> `plate_source=unreadable`.

Jalankan:
    .\\.venv\\Scripts\\python.exe scripts\\mark_legacy_jobs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from alpr import UNREADABLE  # noqa: E402
from database import SessionLocal, init_db  # noqa: E402
from models import Job, Violation  # noqa: E402


def main() -> None:
    init_db()
    session = SessionLocal()
    try:
        jobs = session.scalars(select(Job).order_by(Job.created_at)).all()
        marked_jobs = 0
        simulated = 0
        unreadable = 0

        for job in jobs:
            if job.mode and job.mode != "legacy":
                # Job baru sudah punya snapshot mode; jangan diubah.
                continue
            job.mode = "legacy"
            marked_jobs += 1
            for violation in job.violations:
                current = (violation.plate_source or "").lower()
                if current in {"real", "simulated", "unreadable"} and violation.plate_source:
                    # Jangan turunkan data yang sudah pernah ditandai.
                    if current != "unreadable":
                        continue
                if violation.license_plate and violation.license_plate != UNREADABLE:
                    violation.plate_source = "simulated"
                    violation.ocr_engine = violation.ocr_engine or "mock"
                    simulated += 1
                else:
                    violation.plate_source = "unreadable"
                    unreadable += 1

        session.commit()
        print(f"Job legacy ditandai : {marked_jobs}")
        print(f"Plat simulasi       : {simulated}")
        print(f"Plat tidak terbaca  : {unreadable}")
    finally:
        session.close()


if __name__ == "__main__":
    main()

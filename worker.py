"""Worker antrean in-process: satu video diproses pada satu waktu."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

from database import SessionLocal
from models import Job, utcnow

logger = logging.getLogger(__name__)

_jobs: "queue.Queue[str]" = queue.Queue()
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()
_stop = threading.Event()


def enqueue(job_id: str) -> None:
    _jobs.put(job_id)


def is_running() -> bool:
    return _thread is not None and _thread.is_alive()


def start_worker() -> None:
    global _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return
        _stop.clear()
        _thread = threading.Thread(target=_loop, name="traffic-worker", daemon=True)
        _thread.start()
        logger.info("Worker pipeline dimulai.")


def stop_worker(timeout: float = 5.0) -> None:
    _stop.set()
    thread = _thread
    if thread is not None:
        thread.join(timeout=timeout)
    logger.info("Worker pipeline dihentikan.")


def recover_stale_jobs() -> None:
    """Tandai job yang terputus dan masukkan ulang job PENDING ke antrean."""
    session = SessionLocal()
    pending: list[str] = []
    try:
        jobs = (
            session.query(Job)
            .filter(Job.status.in_(["PENDING", "PROCESSING"]))
            .all()
        )
        for job in jobs:
            if job.status == "PROCESSING":
                job.status = "FAILED"
                job.error_message = "Server berhenti saat job sedang diproses."
                job.finished_at = utcnow()
            else:
                pending.append(job.id)
        session.commit()
    finally:
        session.close()

    for job_id in pending:
        enqueue(job_id)
    if pending:
        logger.info("%s job PENDING dimasukkan kembali ke antrean.", len(pending))


def _loop() -> None:
    while not _stop.is_set():
        try:
            job_id = _jobs.get(timeout=1.0)
        except queue.Empty:
            continue
        try:
            # Import lazy agar startup API tetap ringan.
            from pipeline import run_job

            run_job(job_id)
        except Exception:  # pragma: no cover - run_job sudah menangani error
            logger.exception("Worker gagal memproses job %s", job_id)
        finally:
            _jobs.task_done()

"""Server FastAPI: upload job, status, pelanggaran, dan dashboard."""

from __future__ import annotations

import json
import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

import worker
from config import settings
from database import engine, get_db, init_db
from geometry import PolygonError, clamp_polygon, parse_polygon, polygon_area
from models import Job, Violation
from schemas import (
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    StatsResponse,
    ViolationResponse,
)
from video_io import probe_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v"}
VIOLATION_TYPE = "Melanggar Marka Garis Utuh"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.ensure_dirs()
    init_db()
    worker.recover_stale_jobs()
    worker.start_worker()
    logger.info(
        "Server siap | storage=%s | mock_pipeline=%s | mock_ocr=%s",
        settings.storage_dir,
        settings.mock_pipeline,
        settings.mock_ocr,
    )
    try:
        yield
    finally:
        worker.stop_worker()


app = FastAPI(
    title="Smart Traffic Violation & ALPR Surveillance",
    description=(
        "Pipeline pemantauan pelanggaran lalu lintas berbasis video: "
        "YOLOv11 + ByteTrack + PolygonZone + EasyOCR."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static/evidence",
    StaticFiles(directory=str(settings.evidence_dir)),
    name="evidence",
)
app.mount(
    "/static/output_videos",
    StaticFiles(directory=str(settings.output_dir)),
    name="output_videos",
)

# Aset dashboard (Tailwind + font) disajikan sendiri agar tidak bergantung CDN.
# Daftarkan MIME font secara eksplisit: registry Windows tidak mengenal .woff2,
# dan header yang salah bisa membuat sebagian browser menolak font.
mimetypes.add_type("font/woff2", ".woff2")
mimetypes.add_type("font/woff", ".woff")
assets_dir = settings.base_dir / "static" / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/static/assets",
    StaticFiles(directory=str(assets_dir)),
    name="assets",
)


def _parse_zone(raw: str) -> list[list[float]]:
    try:
        return parse_polygon(raw)
    except PolygonError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _parse_classes(raw: str) -> list[int]:
    raw = (raw or "").strip()
    if not raw:
        return [2, 3, 5, 7]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [part.strip() for part in raw.split(",") if part.strip()]

    if not isinstance(data, list) or not data:
        raise HTTPException(
            status_code=400,
            detail="target_classes harus berupa array kelas COCO, misal [2, 3, 5, 7]",
        )

    classes: list[int] = []
    for value in data:
        if isinstance(value, bool):
            raise HTTPException(status_code=400, detail="Kelas target tidak valid")
        try:
            class_id = int(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="Kelas target harus berupa angka"
            ) from exc
        if class_id < 0 or class_id > 79:
            raise HTTPException(
                status_code=400, detail="Kelas COCO harus di antara 0 dan 79"
            )
        if class_id not in classes:
            classes.append(class_id)

    return classes


def _json_list(raw: Optional[str], fallback: list) -> list:
    try:
        value = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return fallback
    return value if isinstance(value, list) else fallback


def _static_url(relative: Optional[str]) -> Optional[str]:
    if not relative:
        return None
    return "/static/" + relative.replace("\\", "/").lstrip("/")


def _job_response(job: Job) -> JobStatusResponse:
    output_url = None
    if job.status == "COMPLETED" and job.output_path:
        if settings.resolve(job.output_path).exists():
            output_url = _static_url(job.output_path)

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress_percentage=round(float(job.progress or 0.0), 1),
        total_frames=int(job.total_frames or 0),
        processed_frames=int(job.processed_frames or 0),
        violations_detected=int(job.total_violations or 0),
        output_video_url=output_url,
        error_message=job.error_message,
        original_filename=job.original_filename,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        zone_coordinates=_json_list(job.zone_coordinates, []),
        target_classes=_json_list(job.target_classes, []),
        mode=job.mode or "legacy",
        mock_ocr=bool(job.mock_ocr),
        mock_pipeline=bool(job.mock_pipeline),
        imgsz=int(job.imgsz or settings.imgsz),
        alpr_engine=job.alpr_engine or "",
    )


def _violation_response(violation: Violation) -> ViolationResponse:
    return ViolationResponse(
        id=violation.id,
        job_id=violation.job_id,
        tracker_id=violation.tracker_id,
        vehicle_type=violation.vehicle_type,
        violation_type=VIOLATION_TYPE,
        time_seconds=violation.time_seconds,
        frame_number=violation.frame_number,
        license_plate=violation.license_plate,
        ocr_confidence=violation.ocr_confidence,
        plate_source=violation.plate_source or "unreadable",
        plate_raw=violation.plate_raw or "",
        plate_format=violation.plate_format or "",
        plate_region=violation.plate_region or "",
        ocr_engine=violation.ocr_engine or "",
        plate_crop_url=_static_url(violation.plate_crop_path),
        vehicle_confidence=float(violation.vehicle_confidence or 0.0),
        needs_verification=bool(violation.needs_verification),
        evidence_image_url=_static_url(violation.proof_image_path),
        created_at=violation.created_at,
    )


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    database_ok = True
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        database_ok = False

    worker_ok = worker.is_running()
    return HealthResponse(
        status="ok" if database_ok and worker_ok else "degraded",
        database=database_ok,
        worker=worker_ok,
        mock_pipeline=settings.mock_pipeline,
        mock_ocr=settings.mock_ocr,
        model_weights=settings.model_weights,
        device=settings.device,
        imgsz=settings.imgsz,
        imgsz_auto=settings.imgsz_auto,
        ocr_engine=settings.ocr_engine,
    )


@app.post(
    "/api/v1/jobs",
    response_model=JobCreateResponse,
    status_code=202,
    tags=["jobs"],
)
async def create_job(
    file: UploadFile = File(...),
    zone_coordinates: str = Form(...),
    target_classes: str = Form("[2, 3, 5, 7]"),
    db: Session = Depends(get_db),
) -> JobCreateResponse:
    original_name = file.filename or "video"
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Format video tidak didukung. Gunakan salah satu: "
                + ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
            ),
        )

    points = _parse_zone(zone_coordinates)
    classes = _parse_classes(target_classes)

    job_id = f"job_{uuid4().hex[:12]}"
    destination = settings.uploads_dir / f"{job_id}{suffix}"
    bytes_written = 0

    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"Ukuran video melebihi batas "
                            f"{settings.max_upload_mb} MB"
                        ),
                    )
                handle.write(chunk)
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Gagal menyimpan file video: {exc}"
        ) from exc
    finally:
        await file.close()

    if bytes_written == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File video kosong")

    info = probe_video(destination)
    if info is None:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=(
                "File tidak dapat didekode sebagai video. Pastikan codec video "
                "didukung OpenCV."
            ),
        )

    clamped = clamp_polygon(points, info.width, info.height)
    if polygon_area(clamped) < 100:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Zona tidak berada di dalam area video",
        )

    job = Job(
        id=job_id,
        original_filename=original_name,
        stored_path=settings.relative(destination),
        status="PENDING",
        progress=0.0,
        total_violations=0,
        total_frames=info.total_frames,
        processed_frames=0,
        fps=info.fps,
        width=info.width,
        height=info.height,
        zone_coordinates=json.dumps(clamped),
        target_classes=json.dumps(classes),
        mode=(
            "simulated"
            if (settings.mock_pipeline or settings.mock_ocr)
            else "real"
        ),
        mock_ocr=settings.mock_ocr,
        mock_pipeline=settings.mock_pipeline,
        imgsz=settings.imgsz,
        model_weights=settings.model_weights,
        alpr_engine="mock" if settings.mock_ocr else settings.ocr_engine,
    )
    db.add(job)
    try:
        db.commit()
    except Exception:
        db.rollback()
        destination.unlink(missing_ok=True)
        raise

    worker.enqueue(job_id)
    logger.info(
        "Job %s dibuat (%s, %s bytes, kelas=%s)",
        job_id,
        original_name,
        bytes_written,
        classes,
    )

    return JobCreateResponse(
        job_id=job_id,
        status="PENDING",
        message="Video berhasil diunggah dan masuk antrean pemrosesan.",
    )


@app.get("/api/v1/jobs", response_model=list[JobStatusResponse], tags=["jobs"])
def list_jobs(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[JobStatusResponse]:
    statement = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if status:
        statement = statement.where(Job.status == status.upper())
    statement = statement.offset(offset).limit(limit)
    jobs = db.scalars(statement).all()
    return [_job_response(job) for job in jobs]


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatusResponse:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan")
    return _job_response(job)


@app.get(
    "/api/v1/violations",
    response_model=list[ViolationResponse],
    tags=["violations"],
)
def list_violations(
    job_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ViolationResponse]:
    statement = select(Violation).order_by(
        Violation.created_at.desc(), Violation.id.desc()
    )
    if job_id:
        statement = statement.where(Violation.job_id == job_id)
    statement = statement.offset(offset).limit(limit)
    violations = db.scalars(statement).all()
    return [_violation_response(violation) for violation in violations]


@app.get(
    "/api/v1/violations/{violation_id}",
    response_model=ViolationResponse,
    tags=["violations"],
)
def get_violation(
    violation_id: int, db: Session = Depends(get_db)
) -> ViolationResponse:
    violation = db.get(Violation, violation_id)
    if violation is None:
        raise HTTPException(status_code=404, detail="Pelanggaran tidak ditemukan")
    return _violation_response(violation)


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["system"])
def stats(db: Session = Depends(get_db)) -> StatsResponse:
    status_rows = db.execute(
        select(Job.status, func.count(Job.id)).group_by(Job.status)
    ).all()
    counts = {str(status): int(total) for status, total in status_rows}

    # `plate_source` adalah sumber kebenaran, bukan teks plat (yang bisa berupa
    # "TIDAK TERBACA" pada job nyata maupun job simulasi). Job legacy dikeluarkan
    # dari statistik utama agar data era mock tidak tampak seperti hasil model.
    source_rows = db.execute(
        select(Violation.plate_source, func.count(Violation.id))
        .join(Job, Violation.job_id == Job.id)
        .where(Job.mode != "legacy")
        .group_by(Violation.plate_source)
    ).all()
    source_counts = {
        str(source or "unreadable"): int(total)
        for source, total in source_rows
    }
    real_violations = source_counts.get("real", 0)
    unreadable_plates = source_counts.get("unreadable", 0)
    total_violations = real_violations + unreadable_plates
    # Simulasi dihitung lintas job (termasuk legacy) agar operator tahu berapa
    # banyak baris lama yang tidak dipakai pada statistik utama.
    simulated_violations = int(
        db.scalar(
            select(func.count(Violation.id)).where(
                Violation.plate_source == "simulated"
            )
        )
        or 0
    )

    legacy_jobs = int(
        db.scalar(
            select(func.count(Job.id)).where(Job.mode == "legacy")
        )
        or 0
    )
    legacy_violations = int(
        db.scalar(
            select(func.count(Violation.id))
            .join(Job, Violation.job_id == Job.id)
            .where(Job.mode == "legacy")
        )
        or 0
    )

    vehicle_rows = db.execute(
        select(Violation.vehicle_type, func.count(Violation.id))
        .join(Job, Violation.job_id == Job.id)
        .where(Job.mode != "legacy", Violation.plate_source != "simulated")
        .group_by(Violation.vehicle_type)
    ).all()
    vehicle_breakdown = {
        str(vehicle_type): int(total)
        for vehicle_type, total in vehicle_rows
        if vehicle_type
    }

    return StatsResponse(
        total_jobs=sum(counts.values()),
        total_violations=total_violations,
        pending_jobs=counts.get("PENDING", 0),
        processing_jobs=counts.get("PROCESSING", 0),
        completed_jobs=counts.get("COMPLETED", 0),
        failed_jobs=counts.get("FAILED", 0),
        readable_plates=real_violations,
        unreadable_plates=unreadable_plates,
        simulated_plates=simulated_violations,
        simulated_violations=simulated_violations,
        legacy_jobs=legacy_jobs,
        legacy_violations=legacy_violations,
        vehicle_breakdown=vehicle_breakdown,
    )


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    index_path = settings.base_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html tidak ditemukan")
    return FileResponse(index_path)

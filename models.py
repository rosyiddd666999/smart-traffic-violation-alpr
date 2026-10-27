"""Skema tabel database."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base


def utcnow() -> datetime:
    """Waktu UTC naive agar konsisten di SQLite dan JSON."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    stored_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="PENDING", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    total_violations: Mapped[int] = mapped_column(Integer, default=0)

    total_frames: Mapped[int] = mapped_column(Integer, default=0)
    processed_frames: Mapped[int] = mapped_column(Integer, default=0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)

    zone_coordinates: Mapped[str] = mapped_column(Text, default="[]")
    target_classes: Mapped[str] = mapped_column(Text, default="[2, 3, 5, 7]")

    output_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Snapshot konfigurasi saat job diproses. Wajib disimpan agar riwayat tidak
    # tampak "asli" hanya karena server hari ini berjalan di mode asli.
    mode: Mapped[str] = mapped_column(String(20), default="legacy", index=True)
    mock_ocr: Mapped[bool] = mapped_column(Boolean, default=False)
    mock_pipeline: Mapped[bool] = mapped_column(Boolean, default=False)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    model_weights: Mapped[str] = mapped_column(Text, default="")
    alpr_engine: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    violations: Mapped[List["Violation"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Violation(Base):
    __tablename__ = "violations"
    __table_args__ = (
        UniqueConstraint("job_id", "tracker_id", name="uq_violations_job_tracker"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(32), nullable=False, default="Mobil")
    frame_number: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    time_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    license_plate: Mapped[str] = mapped_column(String(20), nullable=False, default="TIDAK TERBACA")
    ocr_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Sumber kebenaran plat: real (OCR video), simulated (mock/legacy), unreadable.
    plate_source: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unreadable", index=True
    )
    plate_raw: Mapped[str] = mapped_column(Text, nullable=False, default="")
    plate_format: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    plate_region: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    ocr_engine: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    plate_crop_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vehicle_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    needs_verification: Mapped[bool] = mapped_column(Boolean, default=False)
    proof_image_path: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    job: Mapped[Job] = relationship(back_populates="violations")

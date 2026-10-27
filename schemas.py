"""Skema request/response API."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress_percentage: float
    total_frames: int
    processed_frames: int
    violations_detected: int
    output_video_url: Optional[str] = None
    error_message: Optional[str] = None
    original_filename: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    zone_coordinates: List[List[float]] = Field(default_factory=list)
    target_classes: List[int] = Field(default_factory=list)
    mode: str = "legacy"
    mock_ocr: bool = False
    mock_pipeline: bool = False
    imgsz: int = 640
    alpr_engine: str = ""


class ViolationResponse(BaseModel):
    id: int
    job_id: str
    tracker_id: int
    vehicle_type: str
    violation_type: str
    time_seconds: float
    frame_number: int
    license_plate: str
    ocr_confidence: float
    plate_source: str = "unreadable"
    plate_raw: str = ""
    plate_format: str = ""
    plate_region: str = ""
    ocr_engine: str = ""
    plate_crop_url: Optional[str] = None
    vehicle_confidence: float = 0.0
    needs_verification: bool = False
    evidence_image_url: Optional[str] = None
    created_at: Optional[datetime] = None


class StatsResponse(BaseModel):
    total_jobs: int
    total_violations: int
    pending_jobs: int
    processing_jobs: int
    completed_jobs: int
    failed_jobs: int
    readable_plates: int
    unreadable_plates: int
    simulated_plates: int = 0
    simulated_violations: int = 0
    legacy_jobs: int = 0
    legacy_violations: int = 0
    vehicle_breakdown: Dict[str, int] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    database: bool
    worker: bool
    mock_pipeline: bool
    mock_ocr: bool
    model_weights: str
    device: str
    imgsz: int
    imgsz_auto: bool = False
    ocr_engine: str = ""

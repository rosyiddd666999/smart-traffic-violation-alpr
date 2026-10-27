"""Konfigurasi aplikasi berbasis environment variable (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


def _env_origins() -> tuple[str, ...]:
    raw = os.getenv("CORS_ORIGINS", "*")
    origins = tuple(item.strip() for item in raw.split(",") if item.strip())
    return origins or ("*",)


def adaptive_imgsz(width: int, fallback: int, auto: bool) -> int:
    """Resolusi inferensi menyesuaikan video agar objek kecil tidak hilang.

    Video 4K diturunkan ke 1280 px (bukan 640) supaya kendaraan jauh tetap
    terklasifikasi benar. Nilai IMGSZ eksplisit selalu menang.
    """
    if not auto:
        return fallback
    if width >= 2560:
        return 1280
    if width >= 1600:
        return 960
    return 640


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    storage_dir: Path
    uploads_dir: Path
    output_dir: Path
    evidence_dir: Path
    database_url: str
    model_weights: str
    device: str
    conf_threshold: float
    iou_threshold: float
    imgsz: int
    frame_stride: int
    max_upload_mb: int
    ocr_gpu: bool
    ocr_conf_threshold: float
    ocr_engine: str
    alpr_detector_model: str
    alpr_ocr_model: str
    alpr_max_samples: int
    imgsz_auto: bool
    mock_pipeline: bool
    mock_ocr: bool
    cors_origins: tuple[str, ...]
    progress_interval_frames: int

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "violations.db"

    def ensure_dirs(self) -> None:
        for directory in (
            self.storage_dir,
            self.uploads_dir,
            self.output_dir,
            self.evidence_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str | Path) -> Path:
        """Ubah path relatif terhadap storage menjadi path absolut."""
        path = Path(relative_path)
        return path if path.is_absolute() else self.storage_dir / path

    def relative(self, path: Path) -> str:
        """Path absolut -> path relatif storage dengan separator '/'.
        Fallback ke nama file bila path berada di luar storage.
        """
        try:
            return path.resolve().relative_to(self.storage_dir.resolve()).as_posix()
        except ValueError:
            return path.name


def load_settings() -> Settings:
    storage_dir = Path(os.getenv("STORAGE_DIR", "storage").strip() or "storage")
    if not storage_dir.is_absolute():
        storage_dir = BASE_DIR / storage_dir

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        database_url = f"sqlite:///{(storage_dir / 'violations.db').as_posix()}"

    return Settings(
        base_dir=BASE_DIR,
        storage_dir=storage_dir,
        uploads_dir=storage_dir / "uploads",
        output_dir=storage_dir / "output_videos",
        evidence_dir=storage_dir / "evidence",
        database_url=database_url,
        model_weights=os.getenv("MODEL_WEIGHTS", "yolo11n.pt").strip() or "yolo11n.pt",
        device=os.getenv("DEVICE", "cpu").strip() or "cpu",
        conf_threshold=_env_float("CONF_THRESHOLD", 0.35),
        iou_threshold=_env_float("IOU_THRESHOLD", 0.5),
        imgsz=max(320, _env_int("IMGSZ", 640)),
        frame_stride=max(1, _env_int("FRAME_STRIDE", 1)),
        max_upload_mb=max(1, _env_int("MAX_UPLOAD_MB", 500)),
        ocr_gpu=_env_bool("OCR_GPU", False),
        ocr_conf_threshold=_env_float("OCR_CONF_THRESHOLD", 0.35),
        ocr_engine=(os.getenv("OCR_ENGINE", "fast-alpr").strip().lower() or "fast-alpr"),
        alpr_detector_model=(
            os.getenv("ALPR_DETECTOR_MODEL", "").strip()
            or "yolo-v9-t-384-license-plate-end2end"
        ),
        alpr_ocr_model=(
            os.getenv("ALPR_OCR_MODEL", "").strip() or "cct-s-v2-global-model"
        ),
        alpr_max_samples=max(1, _env_int("ALPR_MAX_SAMPLES", 6)),
        imgsz_auto=os.getenv("IMGSZ", "").strip() == "",
        mock_pipeline=_env_bool("MOCK_PIPELINE", False),
        mock_ocr=_env_bool("MOCK_OCR", False),
        cors_origins=_env_origins(),
        progress_interval_frames=max(1, _env_int("PROGRESS_INTERVAL_FRAMES", 10)),
    )


settings = load_settings()
settings.ensure_dirs()

"""Pipeline computer vision: YOLO + ByteTrack + PolygonZone + ALPR."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import cv2
import numpy as np
import supervision as sv

from alpr import (
    SOURCE_REAL,
    SOURCE_SIMULATED,
    UNREADABLE,
    build_evidence_image,
    format_plate,
    read_plate_multi,
    save_evidence_image,
)
from config import adaptive_imgsz, settings
from database import SessionLocal
from geometry import build_zone, clamp_polygon, parse_polygon, polygon_area
from models import Job, Violation, utcnow
from video_io import VideoWriter, probe_video

logger = logging.getLogger(__name__)

VEHICLE_TYPES = {
    "car": "Mobil",
    "motorcycle": "Motor",
    "bus": "Bus",
    "truck": "Truk",
}

DEFAULT_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck

# Plat yang "terbaca" pada mode mock agar konsisten dengan video sintetis.
DEMO_PLATES = {101: "B1234XYZ", 102: "D5678AB"}


@dataclass
class Candidate:
    """Kendaraan pelanggar dengan frame terbaik (area terbesar)."""

    tracker_id: int
    vehicle_type: str
    area: float
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    frame_number: int
    time_seconds: float
    vehicle_confidence: float = 0.0


@dataclass
class PlateSample:
    """Satu frame kandidat untuk pembacaan plat (multi-frame voting)."""

    crop: np.ndarray
    score: float
    frame_number: int
    time_seconds: float


def _sample_score(crop: np.ndarray) -> float:
    """Nilai sampel: area kendaraan (plat lebih besar) + ketajaman tepi."""
    if crop.size == 0:
        return 0.0
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return math.sqrt(max(1.0, float(width * height))) * math.log1p(
        min(sharpness, 5000.0)
    )


class Detector(Protocol):
    names: dict[int, str]

    def detect(self, frame: np.ndarray, index: int, total: int) -> sv.Detections:
        ...

    def close(self) -> None:
        ...


class MockDetector:
    """Deteksi deterministik untuk mode demo/tes tanpa mengunduh model AI."""

    names = {2: "car", 3: "motorcycle"}

    def detect(self, frame: np.ndarray, index: int, total: int) -> sv.Detections:
        height, width = frame.shape[:2]
        denominator = max(1, total - 1) if total > 1 else 239
        progress = (index % (denominator + 1)) / denominator if total <= 1 else index / denominator
        progress = min(1.0, max(0.0, progress))

        specs = [
            # tracker, class, cx ratio, bottom awal, bottom akhir, lebar awal, lebar akhir
            (101, 2, 0.40, 0.70, 0.99, 0.075, 0.210),
            (102, 3, 0.63, 0.72, 1.00, 0.045, 0.135),
        ]

        boxes: list[list[float]] = []
        confidences: list[float] = []
        class_ids: list[int] = []
        tracker_ids: list[int] = []

        for tracker_id, class_id, cx_ratio, start_y, end_y, start_w, end_w in specs:
            bottom = height * (start_y + (end_y - start_y) * progress)
            box_width = width * (start_w + (end_w - start_w) * progress)
            box_height = box_width * 0.62
            center_x = width * cx_ratio + width * 0.01 * math.sin(progress * math.tau)
            x1 = max(0.0, center_x - box_width / 2)
            x2 = min(float(width - 1), center_x + box_width / 2)
            y2 = min(float(height - 1), bottom)
            y1 = max(0.0, bottom - box_height)
            if x2 - x1 < 6 or y2 - y1 < 6:
                continue
            boxes.append([x1, y1, x2, y2])
            confidences.append(0.93 if class_id == 2 else 0.87)
            class_ids.append(class_id)
            tracker_ids.append(tracker_id)

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            tracker_id=np.array(tracker_ids, dtype=int),
        )

    def close(self) -> None:
        return None


class YoloDetector:
    def __init__(self, classes: list[int], imgsz: int) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - tergantung environment
            raise RuntimeError(
                "ultralytics belum terpasang. Jalankan: pip install ultralytics"
            ) from exc

        logger.info(
            "Memuat model %s di device=%s (imgsz=%s)",
            settings.model_weights,
            settings.device,
            imgsz,
        )
        self.model = YOLO(settings.model_weights)
        self.names = {int(key): str(value) for key, value in self.model.names.items()}
        self.classes = classes
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray, index: int, total: int) -> sv.Detections:
        results = self.model.track(
            source=frame,
            persist=True,
            classes=self.classes,
            conf=settings.conf_threshold,
            iou=settings.iou_threshold,
            imgsz=self.imgsz,
            tracker="bytetrack.yaml",
            device=settings.device,
            verbose=False,
        )
        if not results:
            return sv.Detections.empty()
        return sv.Detections.from_ultralytics(results[0])

    def close(self) -> None:
        self.model = None  # type: ignore[assignment]


def create_detector(classes: list[int], imgsz: int) -> Detector:
    if settings.mock_pipeline:
        logger.info("MOCK_PIPELINE aktif: memakai deteksi sintetis.")
        return MockDetector()
    return YoloDetector(classes, imgsz)


def _crop_vehicle(
    frame: np.ndarray, bbox: tuple[float, float, float, float]
) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = (x2 - x1) * 0.08
    pad_y = (y2 - y1) * 0.08
    left = int(max(0, x1 - pad_x))
    top = int(max(0, y1 - pad_y))
    right = int(min(width, x2 + pad_x))
    bottom = int(min(height, y2 + pad_y))
    if right <= left or bottom <= top:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[top:bottom, left:right].copy()


def _annotate_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    polygon: list[list[float]],
    violating_tracks: set[int],
    names: dict[int, str],
    job_id: str,
    frame_index: int,
    total_frames: int,
    unique_violations: int,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    contour = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

    overlay = annotated.copy()
    cv2.fillPoly(overlay, [contour], (0, 140, 255))
    cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
    cv2.polylines(annotated, [contour], True, (0, 190, 255), 2, cv2.LINE_AA)

    font_scale = max(0.4, min(0.9, width / 1600.0))
    thickness = 2 if width >= 900 else 1

    if detections is not None and len(detections) > 0:
        for index in range(len(detections)):
            x1, y1, x2, y2 = [int(value) for value in detections.xyxy[index]]
            class_id = (
                int(detections.class_id[index])
                if detections.class_id is not None
                else 2
            )
            confidence = (
                float(detections.confidence[index])
                if detections.confidence is not None
                else 0.0
            )
            tracker_id = (
                int(detections.tracker_id[index])
                if detections.tracker_id is not None
                and detections.tracker_id[index] is not None
                else -1
            )

            violating = tracker_id in violating_tracks
            color = (50, 50, 255) if violating else (60, 210, 90)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            raw_name = names.get(class_id, str(class_id))
            label = f"#{tracker_id} {VEHICLE_TYPES.get(raw_name, raw_name)} {confidence:.2f}"
            if violating:
                label = "PELANGGARAN | " + label

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            label_top = max(0, y1 - text_height - baseline - 8)
            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + text_width + 8, y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, y1 - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

    bar_height = max(46, int(height * 0.075))
    top_overlay = annotated.copy()
    cv2.rectangle(top_overlay, (0, 0), (width, bar_height), (15, 23, 42), -1)
    cv2.addWeighted(top_overlay, 0.82, annotated, 0.18, 0, annotated)

    mode = " | MOCK MODE" if settings.mock_pipeline else ""
    title = f"JOB {job_id} | FRAME {frame_index + 1}/{total_frames or '?'}{mode}"
    subtitle = (
        f"Kendaraan pelanggar: {unique_violations} | "
        f"Status frame: {len(violating_tracks)} di zona"
    )
    cv2.putText(
        annotated,
        title,
        (14, int(bar_height * 0.42)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        subtitle,
        (14, int(bar_height * 0.82)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.8,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    return annotated


def _update_job_progress(
    job: Job,
    session,
    *,
    processed_frames: int,
    total_frames: int,
    progress: float,
) -> None:
    job.processed_frames = processed_frames
    job.progress = round(max(0.0, min(100.0, progress)), 2)
    if total_frames > 0:
        job.total_frames = total_frames
    session.commit()


def run_job(job_id: str) -> None:
    """Proses satu job sampai selesai. Dipanggil oleh worker thread."""
    session = SessionLocal()
    capture: Optional[cv2.VideoCapture] = None
    writer: Optional[VideoWriter] = None
    detector: Optional[Detector] = None
    output_relative: Optional[str] = None

    try:
        job = session.get(Job, job_id)
        if job is None:
            logger.warning("Job %s tidak ditemukan", job_id)
            return
        if job.status in {"COMPLETED", "FAILED"}:
            logger.info("Job %s sudah berstatus %s, dilewati", job_id, job.status)
            return

        job.status = "PROCESSING"
        job.started_at = utcnow()
        job.progress = 0.0
        job.processed_frames = 0
        job.total_violations = 0
        job.error_message = None
        session.commit()

        video_path = settings.resolve(job.stored_path)
        info = probe_video(video_path)
        if info is None:
            raise RuntimeError("Video tidak dapat dibaca atau tidak memiliki frame")

        points = clamp_polygon(
            parse_polygon(job.zone_coordinates), info.width, info.height
        )
        if polygon_area(points) < 100:
            raise RuntimeError(
                "Zona tidak valid setelah disesuaikan dengan resolusi video"
            )

        raw_classes = json.loads(job.target_classes or "[]")
        classes = [int(value) for value in raw_classes] or list(DEFAULT_CLASSES)

        job.total_frames = info.total_frames
        job.fps = info.fps
        job.width = info.width
        job.height = info.height
        effective_imgsz = adaptive_imgsz(
            info.width, settings.imgsz, settings.imgsz_auto
        )
        job.imgsz = effective_imgsz
        job.mode = (
            "simulated"
            if (settings.mock_pipeline or settings.mock_ocr)
            else "real"
        )
        job.mock_ocr = settings.mock_ocr
        job.mock_pipeline = settings.mock_pipeline
        job.model_weights = settings.model_weights
        job.alpr_engine = "mock" if settings.mock_ocr else settings.ocr_engine
        output_relative = f"output_videos/{job_id}.mp4"
        job.output_path = output_relative
        session.commit()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("OpenCV gagal membuka video")

        detector = create_detector(classes, effective_imgsz)
        zone = build_zone(points)
        candidates: dict[int, Candidate] = {}
        plate_samples: dict[int, list[PlateSample]] = {}

        stride = settings.frame_stride
        output_fps = max(1.0, info.fps / stride)
        writer = VideoWriter(
            settings.storage_dir / output_relative,
            info.width,
            info.height,
            output_fps,
        )

        frame_index = 0
        processed_frames = 0
        last_progress_at = 0.0
        total_frames = info.total_frames

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            index = frame_index
            frame_index += 1

            if stride > 1 and index % stride != 0:
                continue

            detections = detector.detect(frame, index, total_frames)
            violating_tracks: set[int] = set()
            rows: list[tuple[int, tuple[float, float, float, float], float]] = []
            if detections is not None and len(detections) > 0:
                mask = zone.trigger(detections)
                for detection_index in range(len(detections)):
                    if detections.tracker_id is None:
                        continue
                    tracker_value = detections.tracker_id[detection_index]
                    if tracker_value is None:
                        continue
                    tracker_id = int(tracker_value)

                    x1, y1, x2, y2 = [
                        float(value) for value in detections.xyxy[detection_index]
                    ]
                    confidence = (
                        float(detections.confidence[detection_index])
                        if detections.confidence is not None
                        else 0.0
                    )
                    rows.append((tracker_id, (x1, y1, x2, y2), confidence))

                    if not bool(mask[detection_index]):
                        continue
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    class_id = (
                        int(detections.class_id[detection_index])
                        if detections.class_id is not None
                        else 2
                    )
                    raw_name = detector.names.get(class_id, str(class_id))
                    vehicle_type = VEHICLE_TYPES.get(raw_name, raw_name.capitalize())
                    violating_tracks.add(tracker_id)

                    current = candidates.get(tracker_id)
                    if current is None or area > current.area:
                        crop = _crop_vehicle(frame, (x1, y1, x2, y2))
                        if crop.size == 0:
                            continue
                        candidates[tracker_id] = Candidate(
                            tracker_id=tracker_id,
                            vehicle_type=vehicle_type,
                            area=area,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            crop=crop,
                            frame_number=index,
                            time_seconds=index / info.fps if info.fps > 0 else 0.0,
                            vehicle_confidence=confidence,
                        )

            # Kumpulkan beberapa frame terbaik per track pelanggar untuk ALPR.
            # Plat pada frame 50 bisa lebih kecil/blur dibanding frame berikutnya.
            for tracker_id, bbox, _confidence in rows:
                if tracker_id not in candidates:
                    continue
                crop = _crop_vehicle(frame, bbox)
                if crop.size == 0:
                    continue
                sample = PlateSample(
                    crop=crop,
                    score=_sample_score(crop),
                    frame_number=index,
                    time_seconds=index / info.fps if info.fps > 0 else 0.0,
                )
                bucket = plate_samples.setdefault(tracker_id, [])
                if len(bucket) < settings.alpr_max_samples:
                    bucket.append(sample)
                elif sample.score > bucket[-1].score:
                    bucket[-1] = sample
                else:
                    continue
                bucket.sort(key=lambda item: item.score, reverse=True)

            annotated = _annotate_frame(
                frame,
                detections,
                points,
                violating_tracks,
                detector.names,
                job_id,
                index,
                total_frames,
                len(candidates),
            )
            writer.write(annotated)
            processed_frames += 1

            if total_frames > 0:
                progress = min(95.0, 95.0 * frame_index / total_frames)
            else:
                progress = min(95.0, processed_frames * 0.5)

            now = time.monotonic()
            if (
                processed_frames % settings.progress_interval_frames == 0
                or now - last_progress_at > 0.75
            ):
                _update_job_progress(
                    job,
                    session,
                    processed_frames=min(total_frames, frame_index)
                    if total_frames > 0
                    else frame_index,
                    total_frames=total_frames,
                    progress=progress,
                )
                last_progress_at = now

        capture.release()
        capture = None
        writer.close()
        writer = None

        if processed_frames == 0:
            raise RuntimeError("Tidak ada frame yang berhasil diproses")

        # frame_index adalah jumlah frame yang benar-benar terbaca.
        job.total_frames = frame_index
        job.processed_frames = frame_index
        job.progress = 96.0
        session.commit()

        total_candidates = len(candidates)
        for order, candidate in enumerate(
            sorted(candidates.values(), key=lambda item: item.tracker_id), start=1
        ):
            # Plat demo mengikuti kendaraan pada video sintetis (mode mock saja).
            mock_plate = (
                DEMO_PLATES.get(candidate.tracker_id)
                or f"B{1000 + (candidate.tracker_id % 8999)}XYZ"
            )
            track_samples = plate_samples.get(candidate.tracker_id) or [
                PlateSample(
                    crop=candidate.crop,
                    score=_sample_score(candidate.crop),
                    frame_number=candidate.frame_number,
                    time_seconds=candidate.time_seconds,
                )
            ]
            plate = read_plate_multi(
                [sample.crop for sample in track_samples],
                mock_plate=mock_plate,
            )

            # Simpan crop plat terpisah agar bisa diperiksa manusia.
            plate_crop_relative = None
            if plate.plate_image is not None and plate.plate_image.size:
                plate_image = plate.plate_image
                if plate_image.shape[1] < 360:
                    factor = min(6.0, 360.0 / max(1, plate_image.shape[1]))
                    plate_image = cv2.resize(
                        plate_image,
                        None,
                        fx=factor,
                        fy=factor,
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                plate_crop_relative = (
                    f"evidence/{job_id}_id_{candidate.tracker_id}_plate.jpg"
                )
                save_evidence_image(
                    plate_image, settings.storage_dir / plate_crop_relative
                )
                plate.plate_image = plate_image

            needs_verification = (
                plate.source != SOURCE_REAL
                or plate.confidence < 0.70
                or candidate.vehicle_confidence < 0.75
            )
            if plate.source == SOURCE_SIMULATED:
                needs_verification = False

            evidence = build_evidence_image(
                candidate.crop,
                plate,
                job_id=job_id,
                tracker_id=candidate.tracker_id,
                vehicle_type=candidate.vehicle_type,
                time_seconds=candidate.time_seconds,
                frame_number=candidate.frame_number,
                needs_verification=needs_verification,
                vehicle_confidence=candidate.vehicle_confidence,
            )
            evidence_relative = (
                f"evidence/{job_id}_id_{candidate.tracker_id}.jpg"
            )
            save_evidence_image(
                evidence, settings.storage_dir / evidence_relative
            )

            session.add(
                Violation(
                    job_id=job_id,
                    tracker_id=candidate.tracker_id,
                    vehicle_type=candidate.vehicle_type,
                    frame_number=candidate.frame_number,
                    time_seconds=round(candidate.time_seconds, 3),
                    license_plate=format_plate(plate.license_plate) or UNREADABLE,
                    ocr_confidence=round(float(plate.confidence), 4),
                    plate_source=plate.source,
                    plate_raw=plate.raw_text,
                    plate_format=plate.plate_format,
                    plate_region=plate.region,
                    ocr_engine=plate.engine or job.alpr_engine,
                    plate_crop_path=plate_crop_relative,
                    vehicle_confidence=round(float(candidate.vehicle_confidence), 4),
                    needs_verification=needs_verification,
                    proof_image_path=evidence_relative,
                )
            )
            job.total_violations = order
            job.progress = min(
                99.0,
                96.0 + 3.0 * order / max(1, total_candidates),
            )
            session.commit()

        job.status = "COMPLETED"
        job.progress = 100.0
        job.processed_frames = frame_index
        job.finished_at = utcnow()
        session.commit()
        logger.info(
            "Job %s selesai: %s frame, %s pelanggaran",
            job_id,
            frame_index,
            job.total_violations,
        )

    except Exception as exc:  # noqa: BLE001 - job harus selalu ditutup
        logger.exception("Job %s gagal", job_id)

        if writer is not None:
            try:
                writer.close()
            except Exception:  # pragma: no cover
                logger.exception("Gagal menutup writer")
            writer = None
        if capture is not None:
            capture.release()
            capture = None
        if detector is not None:
            detector.close()
            detector = None
        if output_relative:
            try:
                (settings.storage_dir / output_relative).unlink(missing_ok=True)
            except OSError:
                pass

        session.rollback()
        try:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "FAILED"
                job.error_message = str(exc)[:2000]
                job.finished_at = utcnow()
                session.commit()
        except Exception:  # pragma: no cover
            logger.exception("Gagal menandai job %s sebagai FAILED", job_id)

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:  # pragma: no cover
                logger.exception("Gagal menutup writer")
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.close()
        session.close()

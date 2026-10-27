"""ALPR: deteksi plat, OCR khusus plat, konsensus multi-frame, dan bukti visual.

Engine utama adalah `fast-alpr` (detektor ONNX + fast-plate-ocr). EasyOCR
tetap tersedia sebagai fallback lewat `OCR_ENGINE=easyocr`.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import settings

logger = logging.getLogger(__name__)

ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
UNREADABLE = "TIDAK TERBACA"

SOURCE_REAL = "real"
SOURCE_SIMULATED = "simulated"
SOURCE_UNREADABLE = "unreadable"

# Pola Indonesia: 1-2 huruf, 1-4 angka, 0-3 huruf.
ID_PATTERN = re.compile(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{0,3})$")
# Plat Eropa tengah (mis. Ceko): 4AB 2345 -> 4AB2345.
CZ_PATTERN = re.compile(r"^(\d[A-Z]{2})(\d{3,4})$")
GENERIC_PATTERN = re.compile(r"^[A-Z0-9]{4,9}$")

# Karakter yang sering tertukar antara huruf dan angka pada plat.
AMBIGUOUS = {
    "0": "O",
    "O": "0",
    "1": "I",
    "I": "1",
    "5": "S",
    "S": "5",
    "8": "B",
    "B": "8",
    "2": "Z",
    "Z": "2",
    "6": "G",
    "G": "6",
    "7": "T",
    "T": "7",
}

_alpr = None
_alpr_lock = threading.Lock()
_reader = None
_reader_lock = threading.Lock()
_ocr_lock = threading.Lock()


@dataclass
class PlateResult:
    license_plate: str
    confidence: float
    raw_text: str = ""
    plate_image: Optional[np.ndarray] = None
    source: str = SOURCE_UNREADABLE
    plate_format: str = ""
    region: str = ""
    engine: str = ""
    reason: str = ""
    samples: int = 0
    agreement: float = 0.0


def _text_variants(value: str, limit: int = 256) -> list[str]:
    """Kombinasi koreksi karakter ambigu, dibatasi agar tetap murah."""
    found = {value}
    frontier = [value]
    for _ in range(4):
        next_frontier: list[str] = []
        for item in frontier:
            for index, char in enumerate(item):
                replacement = AMBIGUOUS.get(char)
                if replacement is None:
                    continue
                candidate = f"{item[:index]}{replacement}{item[index + 1:]}"
                if candidate in found:
                    continue
                found.add(candidate)
                next_frontier.append(candidate)
                if len(found) >= limit:
                    return list(found)
        frontier = next_frontier
    return list(found)


def _is_plate_like(compact: str) -> bool:
    """Plat generik: 5-9 karakter, minimal 2 huruf dan 2 angka.

    Aturan ini menerima `4AB2345` (Ceko) tanpa memaksa pola Indonesia, tetapi
    tetap menolak logo/sponsor seperti "MAN" atau "ZAKO".
    """
    if not 5 <= len(compact) <= 9:
        return False
    if not GENERIC_PATTERN.match(compact):
        return False
    letters = sum(char.isalpha() for char in compact)
    digits = sum(char.isdigit() for char in compact)
    return letters >= 2 and digits >= 2


def normalize_plate_text(raw: str) -> str:
    """Uppercase, buang pemisah, lalu validasi pola plat Indonesia/generik."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if not cleaned:
        return ""
    if ID_PATTERN.match(cleaned):
        return cleaned

    # Koreksi karakter ambigu lebih dulu agar pola Indonesia tetap dikenali,
    # mis. "81234XYZ" -> "B1234XYZ" atau "B1O34XYZ" -> "B1034XYZ".
    variants = [item for item in _text_variants(cleaned) if item != cleaned]
    id_candidates = [
        item for item in variants if len(item) >= 5 and ID_PATTERN.match(item)
    ]
    if id_candidates:
        return min(
            id_candidates,
            key=lambda item: (
                sum(1 for a, b in zip(item, cleaned) if a != b),
                abs(len(item) - 8),
            ),
        )

    if _is_plate_like(cleaned):
        return cleaned

    generic_candidates = [
        item for item in variants if len(item) >= 5 and _is_plate_like(item)
    ]
    if generic_candidates:
        return min(
            generic_candidates,
            key=lambda item: (
                sum(1 for a, b in zip(item, cleaned) if a != b),
                abs(len(item) - 7),
            ),
        )
    return ""


def plate_format_of(compact: str) -> str:
    if ID_PATTERN.match(compact or ""):
        return "id"
    if CZ_PATTERN.match(compact or ""):
        return "cz"
    if _is_plate_like(compact or ""):
        return "generic"
    return ""


def format_plate(plate: str) -> str:
    """Format tampilan: B 1234 XYZ / 4AB 2345 / teks apa adanya."""
    text = (plate or "").strip()
    if not text or text == UNREADABLE:
        return text
    compact = re.sub(r"[^A-Za-z0-9]", "", text).upper()

    match = ID_PATTERN.match(compact)
    if match:
        return " ".join(part for part in match.groups() if part)

    match = CZ_PATTERN.match(compact)
    if match:
        return f"{match.group(1)} {match.group(2)}"

    return compact or text


def _mean(values) -> float:  # noqa: ANN001
    items = [float(value) for value in values]
    return float(sum(items) / len(items)) if items else 0.0


def _alpr_providers() -> Optional[list[str]]:
    """Provider ONNX untuk ALPR, dikendalikan khusus oleh OCR_GPU.

    OCR_GPU=false selalu memaksa CPUExecutionProvider, bahkan ketika YOLO
    memakai GPU (DEVICE=0). Ini penting untuk GPU Pascal lama yang tidak punya
    kernel CUDA di onnxruntime-gpu modern.

    OCR_GPU=true baru memakai CUDA bila CUDAExecutionProvider tersedia. Bila
    tidak tersedia, ALPR tetap berjalan di CPU agar job tidak gagal.
    """
    if not settings.ocr_gpu:
        logger.info("ALPR dipaksa memakai CPUExecutionProvider (OCR_GPU=false).")
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover - onnxruntime wajib ada di image
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        logger.warning(
            "OCR_GPU=true tetapi CUDAExecutionProvider tidak ada "
            "(tersedia: %s). ALPR berjalan di CPU.",
            ", ".join(available),
        )
        return ["CPUExecutionProvider"]
    logger.info("ALPR memakai CUDAExecutionProvider (fallback CPU disiapkan).")
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _get_alpr():
    """Muat detektor plat + OCR ONNX sekali saja (lazy, thread-safe)."""
    global _alpr
    if _alpr is not None:
        return _alpr
    with _alpr_lock:
        if _alpr is None:
            try:
                from fast_alpr import ALPR
            except ImportError as exc:  # pragma: no cover - tergantung environment
                raise RuntimeError(
                    "fast-alpr belum terpasang. Jalankan: pip install 'fast-alpr[onnx]'"
                ) from exc
            providers = _alpr_providers()
            logger.info(
                "Memuat ALPR: detektor=%s ocr=%s providers=%s",
                settings.alpr_detector_model,
                settings.alpr_ocr_model,
                providers or "default",
            )
            _alpr = ALPR(
                detector_model=settings.alpr_detector_model,
                ocr_model=settings.alpr_ocr_model,
                detector_providers=providers,
                ocr_providers=providers,
            )
            logger.info("Model ALPR siap.")
    return _alpr


def _get_reader():
    """EasyOCR hanya dipakai bila OCR_ENGINE=easyocr."""
    global _reader
    if _reader is not None:
        return _reader
    with _reader_lock:
        if _reader is None:
            try:
                import easyocr
            except ImportError as exc:  # pragma: no cover - tergantung environment
                raise RuntimeError(
                    "easyocr belum terpasang. Jalankan: pip install easyocr"
                ) from exc
            logger.info("Memuat model EasyOCR (gpu=%s)...", settings.ocr_gpu)
            _reader = easyocr.Reader(
                ["en"],
                gpu=settings.ocr_gpu,
                verbose=False,
            )
            logger.info("Model EasyOCR siap.")
    return _reader


def _plate_crop_from_bbox(
    image: np.ndarray, bbox, pad_ratio: float = 0.04
) -> Optional[np.ndarray]:
    try:
        x1, y1, x2, y2 = [int(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    height, width = image.shape[:2]
    pad_x = max(1, int((x2 - x1) * pad_ratio))
    pad_y = max(1, int((y2 - y1) * pad_ratio))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right]
    return crop.copy() if crop.size else None


def _fast_alpr_read(vehicle_crop: np.ndarray) -> list[PlateResult]:
    """Deteksi plat lalu baca teksnya dari crop kendaraan."""
    if vehicle_crop.size == 0:
        return []

    alpr = _get_alpr()
    try:
        with _ocr_lock:
            results = alpr.predict(vehicle_crop)
    except Exception:  # pragma: no cover - ONNX bisa gagal pada crop aneh
        logger.exception("fast-alpr gagal membaca crop")
        return []

    readings: list[PlateResult] = []
    for item in results:
        detection = getattr(item, "detection", None)
        ocr = getattr(item, "ocr", None)
        if detection is None or ocr is None:
            continue
        text = normalize_plate_text(getattr(ocr, "text", "") or "")
        if not text:
            continue

        char_probs = list(getattr(ocr, "confidence", []) or [])
        # Buang probabilitas padding agar confidence tidak terlalu optimistis.
        if char_probs:
            char_probs = char_probs[: len(text)]
        confidence = _mean(char_probs) if char_probs else 0.0
        bbox = getattr(detection, "bounding_box", None)
        plate_image = (
            _plate_crop_from_bbox(vehicle_crop, bbox) if bbox is not None else None
        )
        readings.append(
            PlateResult(
                license_plate=text,
                confidence=confidence,
                raw_text=(getattr(ocr, "text", "") or "").strip(),
                plate_image=plate_image,
                source=SOURCE_REAL,
                plate_format=plate_format_of(text),
                region=str(getattr(ocr, "region", "") or ""),
                engine="fast-alpr",
                samples=1,
            )
        )
    return readings


def plate_region(vehicle_crop: np.ndarray) -> np.ndarray:
    """Fallback EasyOCR: ambil bagian bawah bodi kendaraan."""
    if vehicle_crop.size == 0:
        return vehicle_crop
    height, width = vehicle_crop.shape[:2]
    top = int(height * 0.52)
    region = vehicle_crop[top:height, :]
    if region.size == 0:
        region = vehicle_crop
    pad_x = max(1, int(width * 0.04))
    return cv2.copyMakeBorder(
        region,
        10,
        10,
        pad_x,
        pad_x,
        cv2.BORDER_CONSTANT,
        value=(20, 20, 20),
    )


def _preprocess_variants(region: np.ndarray) -> list[tuple[str, np.ndarray]]:
    height, width = region.shape[:2]
    if height == 0 or width == 0:
        return []
    scale = min(4.0, max(2.0, 220.0 / height))
    upscaled = cv2.resize(
        region,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.bilateralFilter(clahe, 9, 75, 75)
    sharpened = cv2.filter2D(
        smooth,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        ("color", upscaled),
        ("clahe", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)),
        ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
    ]


@dataclass
class PlateCandidate:
    text: str
    raw: str
    confidence: float
    y_center: float
    valid: bool


def _read_variant(reader, image: np.ndarray) -> list[PlateCandidate]:
    """Baca satu varian gambar EasyOCR dan hasilkan kandidat teks plat."""
    height = max(1, image.shape[0])
    try:
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
        )
    except Exception:  # pragma: no cover - EasyOCR kadang gagal pada crop kecil
        logger.exception("EasyOCR gagal membaca crop")
        return []

    entries: list[dict] = []
    for box, text, confidence in results:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
        if not cleaned or confidence < 0.1:
            continue
        xs = [float(point[0]) for point in box]
        ys = [float(point[1]) for point in box]
        entries.append(
            {
                "x1": min(xs),
                "x2": max(xs),
                "y1": min(ys),
                "y2": max(ys),
                "cy": sum(ys) / len(ys),
                "text": cleaned,
                "raw": text,
                "conf": float(confidence),
            }
        )

    if not entries:
        return []

    groups: list[dict] = []
    for entry in sorted(entries, key=lambda item: item["cy"]):
        placed = False
        for group in groups:
            overlap = min(group["y2"], entry["y2"]) - max(group["y1"], entry["y1"])
            min_height = min(
                group["y2"] - group["y1"],
                entry["y2"] - entry["y1"],
            )
            if min_height > 0 and overlap / min_height >= 0.5:
                group["entries"].append(entry)
                group["y1"] = min(group["y1"], entry["y1"])
                group["y2"] = max(group["y2"], entry["y2"])
                placed = True
                break
        if not placed:
            groups.append(
                {"y1": entry["y1"], "y2": entry["y2"], "entries": [entry]}
            )

    candidates: list[PlateCandidate] = []
    seen: set[tuple[str, int]] = set()

    def add_candidate(raw_text: str, confidence: float, center_y: float) -> None:
        normalized = normalize_plate_text(raw_text)
        if not normalized:
            return
        y_ratio = max(0.0, min(1.0, center_y / height))
        bucket = round(y_ratio * 20)
        key = (normalized, bucket)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            PlateCandidate(
                text=normalized,
                raw=raw_text,
                confidence=float(confidence),
                y_center=y_ratio,
                valid=_is_plate_like(normalized),
            )
        )

    for group in groups:
        group_entries = sorted(group["entries"], key=lambda item: item["x1"])
        raw_joined = " ".join(entry["raw"] for entry in group_entries)
        weight = sum(max(1, len(entry["text"])) for entry in group_entries)
        confidence = (
            sum(
                entry["conf"] * max(1, len(entry["text"]))
                for entry in group_entries
            )
            / weight
        )
        center_y = sum(entry["cy"] for entry in group_entries) / len(group_entries)
        add_candidate(raw_joined, confidence, center_y)

    for entry in entries:
        add_candidate(entry["raw"], entry["conf"], entry["cy"])

    return candidates


def _easyocr_read(vehicle_crop: np.ndarray) -> list[PlateResult]:
    region = plate_region(vehicle_crop)
    if region.size == 0:
        return []

    reader = _get_reader()
    best: Optional[PlateCandidate] = None
    best_score = -1.0

    with _ocr_lock:
        for _name, image in _preprocess_variants(region):
            for candidate in _read_variant(reader, image):
                if not candidate.valid:
                    continue
                score = candidate.confidence + 0.25 * candidate.y_center
                if score > best_score:
                    best, best_score = candidate, score

    if best is None:
        return []

    return [
        PlateResult(
            license_plate=best.text,
            confidence=best.confidence,
            raw_text=best.raw,
            plate_image=region,
            source=SOURCE_REAL,
            plate_format=plate_format_of(best.text),
            engine="easyocr",
            samples=1,
        )
    ]


def _read_single(vehicle_crop: np.ndarray) -> list[PlateResult]:
    engine = (settings.ocr_engine or "fast-alpr").lower()
    if engine == "easyocr":
        return _easyocr_read(vehicle_crop)
    return _fast_alpr_read(vehicle_crop)


def extract_plate(
    vehicle_crop: np.ndarray,
    mock_plate: Optional[str] = None,
) -> PlateResult:
    """Baca plat dari satu crop kendaraan (satu frame)."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return PlateResult(UNREADABLE, 0.0, reason="Crop kendaraan kosong")

    if settings.mock_ocr:
        plate = normalize_plate_text(mock_plate or "B1234XYZ") or "B1234XYZ"
        return PlateResult(
            plate,
            0.92,
            plate,
            plate_image=vehicle_crop,
            source=SOURCE_SIMULATED,
            plate_format=plate_format_of(plate),
            engine="mock",
            samples=1,
        )

    readings = _read_single(vehicle_crop)
    if not readings:
        return PlateResult(
            UNREADABLE,
            0.0,
            engine=settings.ocr_engine,
            reason="Plat tidak terdeteksi pada crop kendaraan",
        )

    best = max(readings, key=lambda item: item.confidence)
    if best.confidence < settings.ocr_conf_threshold:
        return PlateResult(
            UNREADABLE,
            best.confidence,
            best.raw_text,
            plate_image=best.plate_image,
            source=SOURCE_UNREADABLE,
            plate_format=best.plate_format,
            region=best.region,
            engine=best.engine,
            reason="Keyakinan OCR di bawah ambang",
            samples=1,
        )
    return best


def _consensus(readings: list[PlateResult]) -> Optional[PlateResult]:
    """Voting multi-frame: pilih panjang terkuat, lalu suara per karakter."""
    valid = [r for r in readings if r.source == SOURCE_REAL and r.license_plate]
    if not valid:
        return None

    length_support: dict[int, list[PlateResult]] = defaultdict(list)
    for reading in valid:
        length_support[len(reading.license_plate)].append(reading)

    target_length = max(
        length_support,
        key=lambda length: (
            len(length_support[length]),
            sum(item.confidence for item in length_support[length]),
        ),
    )
    group = length_support[target_length]

    positions: list[dict[str, float]] = [
        defaultdict(float) for _ in range(target_length)
    ]
    for reading in group:
        for index, char in enumerate(reading.license_plate):
            positions[index][char] += max(0.01, reading.confidence)

    voted = "".join(
        max(chars.items(), key=lambda item: item[1])[0] for chars in positions
    )
    if not _is_plate_like(voted):
        voted = max(
            group,
            key=lambda item: item.confidence,
        ).license_plate

    exact = [item for item in group if item.license_plate == voted]
    supporters = exact or group
    confidence = _mean(item.confidence for item in supporters)
    best = max(supporters, key=lambda item: item.confidence)
    agreement = len(exact) / len(group) if group else 0.0

    return PlateResult(
        license_plate=voted,
        confidence=confidence,
        raw_text=best.raw_text,
        plate_image=best.plate_image,
        source=SOURCE_REAL,
        plate_format=plate_format_of(voted),
        region=best.region,
        engine=best.engine,
        samples=len(group),
        agreement=agreement,
    )


def read_plate_multi(
    vehicle_crops: list[np.ndarray],
    mock_plate: Optional[str] = None,
) -> PlateResult:
    """Baca beberapa frame per track lalu ambil konsensus.

    Frame dengan plat lebih besar/tajam dipilih pipeline; di sini semua sampel
    dibaca dan divoting per karakter agar satu frame blur tidak merusak hasil.
    """
    crops = [crop for crop in vehicle_crops if crop is not None and crop.size]
    if not crops:
        return PlateResult(UNREADABLE, 0.0, reason="Tidak ada sampel frame")

    if settings.mock_ocr:
        return extract_plate(crops[0], mock_plate=mock_plate)

    readings: list[PlateResult] = []
    for crop in crops[: max(1, settings.alpr_max_samples)]:
        readings.extend(_read_single(crop))

    consensus = _consensus(readings)
    if consensus is None:
        best_confidence = max((item.confidence for item in readings), default=0.0)
        raw = max(readings, key=lambda item: item.confidence).raw_text if readings else ""
        return PlateResult(
            UNREADABLE,
            best_confidence,
            raw,
            source=SOURCE_UNREADABLE,
            engine=settings.ocr_engine,
            reason="Plat tidak terdeteksi pada semua sampel",
            samples=len(crops),
        )

    if consensus.confidence < settings.ocr_conf_threshold:
        consensus.source = SOURCE_UNREADABLE
        consensus.reason = "Keyakinan OCR di bawah ambang"
        consensus.license_plate = UNREADABLE
    elif consensus.agreement < 0.34:
        # Voting per karakter bisa membentuk teks yang tidak pernah muncul utuh
        # pada satu frame. Tanpa dukungan utuh, hasil dianggap tidak terbaca.
        consensus.source = SOURCE_UNREADABLE
        consensus.reason = "Tidak ada frame yang saling mendukung (agreement rendah)"
        consensus.license_plate = UNREADABLE

    return consensus


def build_evidence_image(
    vehicle_crop: np.ndarray,
    plate: PlateResult,
    *,
    job_id: str,
    tracker_id: int,
    vehicle_type: str,
    time_seconds: float,
    frame_number: int,
    needs_verification: bool = False,
    vehicle_confidence: float = 0.0,
) -> np.ndarray:
    """Susun gambar bukti tanpa mencetak plat karangan sebagai fakta."""
    if vehicle_crop.size == 0:
        vehicle_crop = np.full((240, 360, 3), 40, dtype=np.uint8)

    height, width = vehicle_crop.shape[:2]
    scale = 1.0
    if width > 760:
        scale = 760.0 / width
    elif width < 520:
        scale = min(4.0, 520.0 / width)
    if height * scale > 620:
        scale = 620.0 / height

    resized = cv2.resize(
        vehicle_crop,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    image_h, image_w = resized.shape[:2]

    canvas_w = max(image_w + 32, 840)
    header_h = 66
    footer_h = 150
    canvas_h = header_h + image_h + footer_h
    canvas = np.full((canvas_h, canvas_w, 3), (15, 23, 42), dtype=np.uint8)

    simulated = plate.source == SOURCE_SIMULATED
    unreadable = plate.source == SOURCE_UNREADABLE or not plate.license_plate
    accent = (60, 180, 255) if simulated else (60, 60, 255)
    cv2.line(canvas, (0, 0), (canvas_w, 0), accent, 4)
    cv2.putText(
        canvas,
        f"BUKTI PELANGGARAN - {vehicle_type}",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{job_id} | Track #{tracker_id} | {time_seconds:.2f}s (frame {frame_number})",
        (18, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    if simulated:
        badge = "PLAT SIMULASI"
        badge_color = (0, 165, 235)
    elif unreadable:
        badge = "PLAT TIDAK TERBACA"
        badge_color = (100, 116, 139)
    else:
        badge = "HASIL OCR - PERLU VERIFIKASI"
        badge_color = (16, 145, 120)

    (badge_w, badge_h), _ = cv2.getTextSize(
        badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        canvas,
        (canvas_w - badge_w - 34, 14),
        (canvas_w - 14, 14 + badge_h + 16),
        badge_color,
        -1,
    )
    cv2.putText(
        canvas,
        badge,
        (canvas_w - badge_w - 24, 14 + badge_h + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (12, 18, 30),
        1,
        cv2.LINE_AA,
    )

    x_offset = (canvas_w - image_w) // 2
    canvas[header_h:header_h + image_h, x_offset:x_offset + image_w] = resized

    footer_y = header_h + image_h
    cv2.line(canvas, (0, footer_y), (canvas_w, footer_y), (51, 65, 85), 1)

    plate_img = plate.plate_image
    text_x = 24
    if plate_img is not None and plate_img.size:
        plate_w = min(320, canvas_w // 2 - 24)
        ratio = plate_w / plate_img.shape[1]
        plate_resized = cv2.resize(
            plate_img,
            (plate_w, max(1, int(plate_img.shape[0] * ratio))),
            interpolation=cv2.INTER_CUBIC,
        )
        if plate_resized.shape[0] > 96:
            ratio = 96 / plate_resized.shape[0]
            plate_resized = cv2.resize(
                plate_resized,
                (max(1, int(plate_resized.shape[1] * ratio)), 96),
                interpolation=cv2.INTER_AREA,
            )
        cv2.rectangle(
            canvas,
            (18, footer_y + 10),
            (18 + plate_resized.shape[1] + 6, footer_y + 10 + plate_resized.shape[0] + 6),
            (241, 245, 249),
            -1,
        )
        canvas[
            footer_y + 13:footer_y + 13 + plate_resized.shape[0],
            21:21 + plate_resized.shape[1],
        ] = plate_resized
        text_x = 18 + plate_resized.shape[1] + 28

    if simulated:
        headline = f"PLAT SIMULASI (data uji): {format_plate(plate.license_plate)}"
        headline_color = (60, 200, 255)
    elif unreadable:
        headline = "PLAT TIDAK TERBACA"
        headline_color = (203, 213, 225)
    else:
        headline = f"HASIL OCR: {format_plate(plate.license_plate)}"
        headline_color = (255, 255, 255)

    cv2.putText(
        canvas,
        headline,
        (text_x, footer_y + 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72 if len(headline) > 30 else 0.9,
        headline_color,
        2,
        cv2.LINE_AA,
    )

    if simulated:
        source_line = "Bukan hasil pembacaan video - jangan dipakai sebagai bukti."
    else:
        engine = plate.engine or settings.ocr_engine
        confidence = f"{plate.confidence * 100:.1f}%"
        region = f" | region: {plate.region}" if plate.region else ""
        source_line = f"Sumber: {engine} | keyakinan {confidence}{region}"

    cv2.putText(
        canvas,
        source_line,
        (text_x, footer_y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (148, 163, 184),
        1,
        cv2.LINE_AA,
    )

    if not simulated and not unreadable:
        samples = f"{plate.samples} frame" if plate.samples else "1 frame"
        note = f"Bukti: {samples} | verifikasi manual tetap diperlukan"
        if needs_verification:
            note = "PERLU VERIFIKASI: confidence kendaraan/plat di bawah ambang"
        if vehicle_confidence:
            note += f" | kendaraan {vehicle_confidence * 100:.0f}%"
        cv2.putText(
            canvas,
            note,
            (text_x, footer_y + 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (100, 116, 139),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Smart Traffic Violation & ALPR Surveillance",
        (text_x, footer_y + 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )

    return canvas


def save_evidence_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Gagal menyimpan bukti ke {path}")

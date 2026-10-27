"""Unduh + inisialisasi model AI saat build image.

Tujuannya agar container produksi bisa berjalan tanpa internet: bobot YOLO
dan dua model ONNX fast-alpr sudah tersimpan di dalam image (cache pengguna
appuser). Script ini dijalankan oleh Dockerfile sebagai user non-root.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger("warmup")

# Ultralytics & matplotlib menulis konfigurasi/cache ke HOME; arahkan ke
# direktori yang bisa ditulis user container.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.home() / ".config" / "Ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "matplotlib"))


def warm_yolo() -> Path:
    """Muat bobot YOLO (mengunduh bila belum ada di image)."""
    from ultralytics import YOLO

    weights = os.getenv("MODEL_WEIGHTS", "yolo11n.pt").strip() or "yolo11n.pt"
    path = Path(weights)
    if not path.is_absolute():
        path = Path.cwd() / path

    logger.info("Menyiapkan YOLO: %s", path)
    YOLO(str(path))

    if not path.exists():
        raise RuntimeError(f"Bobot YOLO tidak ditemukan setelah dimuat: {path}")
    logger.info("YOLO siap (%.1f MB).", path.stat().st_size / (1024 * 1024))
    return path


def warm_alpr() -> None:
    """Buat sesi ONNX detektor + OCR agar model terunduh ke cache image."""
    import numpy as np
    from fast_alpr import ALPR

    detector = (
        os.getenv("ALPR_DETECTOR_MODEL", "").strip()
        or "yolo-v9-t-384-license-plate-end2end"
    )
    ocr = os.getenv("ALPR_OCR_MODEL", "").strip() or "cct-s-v2-global-model"

    logger.info("Menyiapkan ALPR: detector=%s ocr=%s", detector, ocr)
    alpr = ALPR(detector_model=detector, ocr_model=ocr)

    # Satu inferensi pada gambar kosong memaksa kedua sesi ONNX benar-benar
    # dibuat (bukan hanya objek Python-nya).
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    alpr.predict(blank)
    logger.info("ALPR siap (detektor + OCR).")


def main() -> int:
    if os.getenv("SKIP_MODEL_WARMUP", "").strip() == "1":
        logger.warning("SKIP_MODEL_WARMUP=1 -> model tidak disiapkan saat build.")
        return 0

    warm_yolo()
    warm_alpr()
    logger.info("Semua model siap dipakai offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

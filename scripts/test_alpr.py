"""Uji cepat modul ALPR memakai plat sintetis beresolusi tinggi.

Jalankan:
    python scripts/test_alpr.py

Skrip ini memaksa MOCK_OCR=0 sehingga engine ALPR asli (default fast-alpr)
dipakai. Model ONNX diunduh sekali saat pertama kali dijalankan.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["MOCK_OCR"] = "0"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from alpr import build_evidence_image, extract_plate, save_evidence_image  # noqa: E402


def make_vehicle_crop() -> np.ndarray:
    crop = np.full((420, 760, 3), (58, 60, 64), dtype=np.uint8)
    cv2.rectangle(crop, (80, 90), (680, 420), (46, 62, 190), -1)
    cv2.rectangle(crop, (170, 40), (590, 100), (46, 62, 190), -1)
    cv2.rectangle(crop, (200, 55), (560, 95), (150, 170, 185), -1)

    cv2.rectangle(crop, (240, 300), (520, 392), (245, 245, 245), -1)
    cv2.rectangle(crop, (237, 297), (523, 395), (20, 20, 20), 3)
    cv2.putText(
        crop,
        "B 1234 XYZ",
        (252, 367),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.55,
        (15, 15, 15),
        4,
        cv2.LINE_AA,
    )
    return crop


def main() -> None:
    crop = make_vehicle_crop()
    result = extract_plate(crop)

    print(f"Plat terbaca : {result.license_plate}")
    print(f"Raw OCR      : {result.raw_text}")
    print(f"Confidence   : {result.confidence:.3f}")

    evidence = build_evidence_image(
        crop,
        result,
        job_id="job_ocr_test",
        tracker_id=99,
        vehicle_type="Mobil",
        time_seconds=1.23,
        frame_number=31,
    )
    output = Path("storage/evidence/ocr_test.jpg")
    save_evidence_image(evidence, output)
    print(f"Bukti disimpan: {output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a small, anonymised demo dataset from a live installation.

The public repository must never ship real licence plates. This script takes the
newest completed jobs from a production database, then:

1. copies only the evidence images referenced by those jobs,
2. blurs the plate area in the photo, the plate thumbnail, and rewrites the
   "HASIL OCR" headline with a synthetic plate number,
3. rewrites the database so plate text matches the redacted images,
4. drops every other job and every upload/output reference.

Run it inside the application image (it needs fast-alpr for plate detection):

    docker run --rm -i \
        --entrypoint python \
        -v "$PWD:/work" \
        -v traffic-alpr_app_data:/data:ro \
        traffic-alpr:gpu /work/scripts/make_demo_seed.py \
            --db /data/violations.db \
            --storage /data \
            --out /work/seed \
            --jobs 3

The result is deterministic: the same input plate always maps to the same
synthetic plate, so re-running produces an identical seed.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import cv2
import numpy as np

# Colours used by alpr.build_evidence_image().
CANVAS_BG = (15, 23, 42)
THUMBNAIL_BG = (241, 245, 249)
HEADLINE_COLOR = (255, 255, 255)
HEADER_H = 66
FOOTER_H = 150

# Confidences below this are treated as false positives when blurring.
DETECTION_MIN_CONF = 0.35


def synthetic_plate(index: int, plate_format: str) -> str:
    """Return a deterministic, format-compatible fake plate."""
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    a = letters[index % len(letters)]
    b = letters[(index * 7 + 3) % len(letters)]
    c = letters[(index * 11 + 5) % len(letters)]
    # Four distinct-looking digits derived from the index.
    n = ((index * 37 + 1234) % 9000) + 1000
    digits = f"{n:04d}"
    if plate_format == "id":
        return f"{a} {digits} {b}{c}"
    return f"{digits[0]}{a}{b} {digits[1:]}"


def _blur_region(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """Pixelate then blur a rectangle so the original text is unrecoverable."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    region = image[y1:y2, x1:x2]
    small = cv2.resize(
        region, (max(1, region.shape[1] // 12), max(1, region.shape[0] // 8)),
        interpolation=cv2.INTER_AREA,
    )
    restored = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(restored, (0, 0), sigmaX=6)


def _detect_plates(image: np.ndarray, detector) -> list[tuple[int, int, int, int]]:
    """Return plate bounding boxes found in the vehicle photo."""
    height = image.shape[0]
    photo = image[HEADER_H:height - FOOTER_H]
    if photo.size == 0:
        return []
    boxes: list[tuple[int, int, int, int]] = []
    try:
        for det in detector.predict(photo):
            box = det.bounding_box
            if float(det.confidence) < DETECTION_MIN_CONF:
                continue
            boxes.append(
                (int(box.x1) - 6, int(box.y1) + HEADER_H - 6, int(box.x2) + 6, int(box.y2) + HEADER_H + 6)
            )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ! plate detector failed: {exc}")
    return boxes


def _find_thumbnail(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Locate the white-framed plate thumbnail in the footer."""
    footer_top = image.shape[0] - FOOTER_H
    footer = image[footer_top:, : image.shape[1] // 2]
    # JPEG compression shifts the exact colour, so accept a bright range.
    mask = cv2.inRange(footer, (215, 220, 225), (255, 255, 255))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 2000 or w < 80 or h < 20:
            continue
        if best is None or area > best[4]:
            best = (x, y, w, h, area)
    if best is None:
        return None
    x, y, w, h, _ = best
    return (x + 1, footer_top + y + 1, x + w - 1, footer_top + y + h - 1)


def redact_evidence(path: Path, detector, fake_plate: str) -> bool:
    """Blur plates and rewrite the OCR headline. Returns False when unsafe."""
    image = cv2.imread(str(path))
    if image is None:
        return False

    footer_top = image.shape[0] - FOOTER_H
    boxes = _detect_plates(image, detector)
    if boxes:
        for box in boxes:
            _blur_region(image, *box)
    else:
        # Conservative fallback: blur the lower-centre of the photo, where a
        # plate is expected. Better to over-redact than to leak a plate.
        photo_h = image.shape[0] - FOOTER_H - HEADER_H
        cx = image.shape[1] // 2
        _blur_region(image, cx - image.shape[1] // 4, HEADER_H + int(photo_h * 0.55),
                     cx + image.shape[1] // 4, HEADER_H + photo_h)
        print("  ! no plate detected, used fallback blur")

    # A sharp foreground plate can hide a background plate (or push it below the
    # detector's confidence threshold). Blurring the foreground changes the
    # context, so run the detector again on the modified photo and redact any
    # new boxes. Two extra passes are cheap and catch the common case of a
    # second vehicle visible above the main crop.
    for _ in range(2):
        extra = _detect_plates(image, detector)
        if not extra:
            break
        for box in extra:
            _blur_region(image, *box)

    thumb = _find_thumbnail(image)
    if thumb:
        _blur_region(image, *thumb)
        text_x = thumb[2] + 8
    else:
        text_x = 24

    # Erase the original headline and draw a synthetic one in the same style.
    cv2.rectangle(image, (text_x - 4, footer_top + 18), (image.shape[1] - 4, footer_top + 58),
                  CANVAS_BG, -1)
    headline = f"HASIL OCR: {fake_plate}"
    cv2.putText(image, headline, (text_x, footer_top + 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 if len(headline) <= 30 else 0.72, HEADLINE_COLOR, 2, cv2.LINE_AA)

    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return True


def build_seed(db_path: Path, storage: Path, out_dir: Path, keep_jobs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    evidence_out = out_dir / "evidence"
    if evidence_out.exists():
        shutil.rmtree(evidence_out)
    evidence_out.mkdir(parents=True)

    demo_db = out_dir / "demo.db"
    shutil.copy2(db_path, demo_db)
    conn = sqlite3.connect(demo_db)
    # SQLite does not enforce ON DELETE CASCADE unless foreign keys are enabled.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id FROM jobs WHERE status = 'COMPLETED' ORDER BY created_at DESC LIMIT ?",
        (keep_jobs,),
    ).fetchall()
    keep = [row["id"] for row in rows]
    if not keep:
        raise SystemExit("no completed jobs found")
    print(f"keeping {len(keep)} jobs: {', '.join(keep)}")

    placeholders = ",".join("?" for _ in keep)
    conn.execute(f"DELETE FROM jobs WHERE id NOT IN ({placeholders})", keep)

    from fast_alpr import ALPR

    detector = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-s-v2-global-model",
        detector_providers=["CPUExecutionProvider"],
        ocr_providers=["CPUExecutionProvider"],
    ).detector

    plate_map: dict[str, str] = {}

    def fake_for(original: str, plate_format: str) -> str:
        if original not in plate_map:
            plate_map[original] = synthetic_plate(len(plate_map), plate_format)
        return plate_map[original]

    violations = conn.execute(
        f"SELECT id, job_id, license_plate, plate_format, plate_raw, proof_image_path, "
        f"plate_crop_path FROM violations WHERE job_id IN ({placeholders})",
        keep,
    ).fetchall()

    for violation in violations:
        original = (violation["license_plate"] or "").strip()
        if original and original != "TIDAK TERBACA":
            fake = fake_for(original, violation["plate_format"] or "")
            conn.execute(
                "UPDATE violations SET license_plate = ?, plate_raw = ? WHERE id = ?",
                (fake, fake.replace(" ", ""), violation["id"]),
            )
            for column in ("proof_image_path", "plate_crop_path"):
                relative = violation[column]
                if not relative:
                    continue
                source = storage / relative
                target = evidence_out / Path(relative).name
                if not source.exists():
                    print(f"  ! missing {source}")
                    continue
                if column == "proof_image_path":
                    shutil.copy2(source, target)
                    redact_evidence(target, detector, fake)
                else:
                    # The thumbnail is only the plate: destroy it entirely.
                    img = cv2.imread(str(source))
                    if img is not None:
                        _blur_region(img, 0, 0, img.shape[1], img.shape[0])
                        _blur_region(img, 0, 0, img.shape[1], img.shape[0])
                        cv2.imwrite(str(target), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                print(f"  redacted {target.name}")

    conn.execute("UPDATE jobs SET stored_path = '', output_path = NULL, error_message = NULL")
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    jobs = sqlite3.connect(demo_db).execute(
        "SELECT (SELECT COUNT(*) FROM jobs), (SELECT COUNT(*) FROM violations)"
    ).fetchone()
    print(f"seed ready: {jobs[0]} jobs, {jobs[1]} violations, "
          f"{len(list(evidence_out.glob('*.jpg')))} evidence files")
    print(f"plate mapping: {plate_map}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("/data/violations.db"))
    parser.add_argument("--storage", type=Path, default=Path("/data"))
    parser.add_argument("--out", type=Path, default=Path("seed"))
    parser.add_argument("--jobs", type=int, default=3, help="number of newest completed jobs")
    args = parser.parse_args()
    build_seed(args.db, args.storage, args.out, args.jobs)


if __name__ == "__main__":
    main()

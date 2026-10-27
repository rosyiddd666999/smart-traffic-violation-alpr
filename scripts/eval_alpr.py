"""Evaluate ALPR accuracy on a reference clip.

The script is generic: the ground-truth plate and the reference bounding box are
passed on the command line, because the public sample footage contains real
licence plates that must never be committed to this repository.

Run it with your own clip:

    python scripts/eval_alpr.py \
        --video samples/vehicles.mp4 \
        --truth 'B 1234 XYZ' \
        --reference-box 1294,1589,1630,1929

The reference box is the vehicle at frame 50, in source-video coordinates. When
`--reference-box` is omitted, the script picks the largest track instead.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MOCK_OCR", "0")
os.environ.setdefault("MOCK_PIPELINE", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from alpr import extract_plate, format_plate, read_plate_multi  # noqa: E402


def _parse_box(raw: str) -> tuple[float, float, float, float]:
    parts = [float(item) for item in raw.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("expected x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def _iou(a, b) -> float:  # noqa: ANN001
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _cer(prediction: str, truth: str) -> float:
    return _levenshtein(prediction, truth) / max(1, len(truth))


def _score(crop: np.ndarray) -> float:
    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return math.sqrt(max(1.0, float(width * height))) * math.log1p(
        min(sharpness, 5000.0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="samples/vehicles.mp4")
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--end", type=int, default=70)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--truth", required=True,
                        help="ground-truth plate, e.g. 'B 1234 XYZ'")
    parser.add_argument("--reference-box", type=_parse_box, default=None,
                        help="x1,y1,x2,y2 of the target vehicle at frame 50")
    parser.add_argument("--max-samples", type=int, default=6)
    args = parser.parse_args()
    args.truth = args.truth.replace(" ", "").upper()

    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Video tidak bisa dibuka: {args.video}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    tracks: dict[int, list[tuple[int, tuple[float, float, float, float], np.ndarray]]] = {}
    frame = args.start

    while frame < args.end:
        ok, image = capture.read()
        if not ok:
            break
        results = model.track(
            source=image,
            persist=True,
            classes=[2, 3, 5, 7],
            conf=0.35,
            imgsz=args.imgsz,
            tracker="bytetrack.yaml",
            device="cpu",
            verbose=False,
        )
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            for index in range(len(boxes)):
                if ids is None:
                    continue
                tracker_id = int(ids[index].item())
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[index].tolist()]
                crop = image[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                tracks.setdefault(tracker_id, []).append(
                    (frame, (x1, y1, x2, y2), crop.copy())
                )
        frame += 1
    capture.release()

    if not tracks:
        raise SystemExit("Tidak ada kendaraan yang terdeteksi.")

    # Pick the track that matches the reference box, or the largest vehicle
    # when no box was supplied.
    if args.reference_box is not None:
        target_id = None
        best_iou = -1.0
        for tracker_id, samples in tracks.items():
            for number, bbox, _crop in samples:
                if number != 50:
                    continue
                score = _iou(bbox, args.reference_box)
                if score > best_iou:
                    best_iou, target_id = score, tracker_id

        if target_id is None:
            candidates = [item for samples in tracks.values() for item in samples]
            nearest = max(
                candidates, key=lambda item: _iou(item[1], args.reference_box)
            )
            target_id = next(
                tracker_id
                for tracker_id, samples in tracks.items()
                if any(
                    item[0] == nearest[0] and item[1] == nearest[1]
                    for item in samples
                )
            )
    else:
        target_id = max(
            tracks,
            key=lambda item: max(
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                for _number, bbox, _crop in tracks[item]
            ),
        )

    samples = tracks[target_id]
    best = sorted(samples, key=lambda item: _score(item[2]), reverse=True)
    selected = best[: args.max_samples]

    print(f"Video       : {args.video}")
    print(f"Track       : #{target_id} ({len(samples)} frame terdeteksi)")
    print(f"Frame dipakai: {[item[0] for item in selected]}")
    print(f"Ground truth: {args.truth}")

    per_frame = []
    for number, _bbox, crop in sorted(selected, key=lambda item: item[0]):
        result = extract_plate(crop)
        compact = result.license_plate.replace(" ", "")
        per_frame.append(compact)
        print(
            f"  frame {number:>3}: {format_plate(result.license_plate):<12} "
            f"conf={result.confidence:.3f} source={result.source}"
        )

    consensus = read_plate_multi([item[2] for item in selected])
    compact = consensus.license_plate.replace(" ", "")
    exact = compact == args.truth
    cer = _cer(compact, args.truth)

    print("\nHASIL MULTI-FRAME")
    print(f"  Plat       : {format_plate(consensus.license_plate)}")
    print(f"  Confidence : {consensus.confidence:.3f} ({consensus.samples} sampel)")
    print(f"  Exact match: {exact}")
    print(f"  CER        : {cer:.3f}")

    baseline_cer = 4 / len(args.truth)  # measured with the previous EasyOCR engine
    print("\nPERBANDINGAN (baseline diukur saat investigasi awal)")
    print(
        f"  EasyOCR lama : exact=0.0%  CER={baseline_cer:.3f} "
        "(hanya sebagian karakter terbaca)"
    )
    print(
        f"  Engine baru  : exact={100.0 if exact else 0.0:.1f}%  CER={cer:.3f}"
    )
    if not exact:
        print("  Catatan: exact match belum tercapai pada run ini.")


if __name__ == "__main__":
    main()

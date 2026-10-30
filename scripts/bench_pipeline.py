"""Benchmark terpisah per tahap pipeline: decode, deteksi, anotasi, encode.

Dipakai untuk menjawab "tahap mana yang sebenarnya lambat" dan "apakah
resolusi deteksi 960 sudah cukup dibanding 1280/1536".

Contoh:
    python scripts/bench_pipeline.py --video storage/uploads/job_35a6a93f5250.mp4
    python scripts/bench_pipeline.py --video clip.mp4 --frames 120 --encode-frames 60
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# Timpa STORAGE_DIR milik container supaya config.ensure_dirs() tidak menulis
# ke volume aplikasi saat benchmark dijalankan dengan repo read-only.
os.environ["STORAGE_DIR"] = os.getenv("BENCH_STORAGE_DIR") or tempfile.mkdtemp(
    prefix="bench-storage-"
)
os.environ.pop("IMGSZ", None)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import _annotate_frame  # noqa: E402
from video_io import VideoWriter, probe_video  # noqa: E402


@dataclass
class Timing:
    label: str
    frames: int
    seconds: float

    @property
    def per_frame_ms(self) -> float:
        return 1000.0 * self.seconds / max(1, self.frames)

    @property
    def fps(self) -> float:
        return self.frames / max(1e-9, self.seconds)


def _iter_frames(path: Path, limit: int):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Tidak bisa membuka {path}")
    try:
        count = 0
        while count < limit:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            yield count, frame
    finally:
        capture.release()


def bench_decode(path: Path, limit: int) -> Timing:
    start = time.perf_counter()
    frames = 0
    size = None
    for _, frame in _iter_frames(path, limit):
        frames += 1
        size = frame.shape
    elapsed = time.perf_counter() - start
    print(f"  decode  {frames} frame {size[1]}x{size[0]}")
    return Timing("decode", frames, elapsed)


def _detections_for(frame: np.ndarray, imgsz: int, conf: float, model) -> "object":
    from supervision import Detections

    results = model.track(
        source=frame,
        persist=True,
        classes=[2, 3, 5, 7],
        conf=conf,
        iou=0.5,
        imgsz=imgsz,
        tracker="bytetrack.yaml",
        device=0,
        verbose=False,
    )
    if not results:
        return Detections.empty()
    return Detections.from_ultralytics(results[0])


def bench_detect(
    path: Path, limit: int, sizes: list[int], conf: float, keep: bool
) -> tuple[list[Timing], dict[int, list]]:
    from ultralytics import YOLO

    timings: list[Timing] = []
    collected: dict[int, list] = {}
    for imgsz in sizes:
        model = YOLO(os.getenv("MODEL_WEIGHTS", "yolo11n.pt"))
        # warmup
        for _, frame in _iter_frames(path, 2):
            _detections_for(frame, imgsz, conf, model)
        start = time.perf_counter()
        frames = 0
        per_frame = []
        for _, frame in _iter_frames(path, limit):
            detections = _detections_for(frame, imgsz, conf, model)
            frames += 1
            if keep:
                per_frame.append(detections)
        elapsed = time.perf_counter() - start
        if keep:
            collected[imgsz] = per_frame
        timings.append(Timing(f"detect@{imgsz}", frames, elapsed))
        print(f"  deteksi imgsz={imgsz:<5} {elapsed:7.2f}s  "
              f"{1000 * elapsed / max(1, frames):6.1f} ms/frame")
        del model
    return timings, collected


def bench_annotate(
    path: Path, limit: int, imgsz: int, conf: float
) -> Timing:
    from ultralytics import YOLO

    model = YOLO(os.getenv("MODEL_WEIGHTS", "yolo11n.pt"))
    info = probe_video(path)
    polygon = [
        [0.10 * info.width, 0.62 * info.height],
        [0.90 * info.width, 0.62 * info.height],
        [0.97 * info.width, 0.98 * info.height],
        [0.03 * info.width, 0.98 * info.height],
    ]
    frames = 0
    start = time.perf_counter()
    for index, frame in _iter_frames(path, limit):
        detections = _detections_for(frame, imgsz, conf, model)
        _annotate_frame(
            frame, detections, polygon, set(), model.names, "bench",
            index, info.total_frames, 0,
        )
        frames += 1
    elapsed = time.perf_counter() - start
    print(f"  anotasi {frames} frame {elapsed:7.2f}s")
    return Timing("annotate", frames, elapsed)


def bench_encode(
    path: Path, frames: int, out_height: int, crf: int = 23
) -> Timing:
    info = probe_video(path)
    scale = out_height / info.height
    width = int(round(info.width * scale / 2) * 2)
    out = Path(tempfile.mkdtemp(prefix="bench-enc-")) / f"out_{out_height}.mp4"
    writer = VideoWriter(out, width, out_height, info.fps)
    start = time.perf_counter()
    count = 0
    for _, frame in _iter_frames(path, frames):
        if out_height != info.height:
            frame = cv2.resize(frame, (width, out_height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
    writer.close()
    elapsed = time.perf_counter() - start
    size_mb = out.stat().st_size / 2**20
    print(f"  encode {count} frame -> {width}x{out_height} "
          f"{elapsed:7.2f}s  {size_mb:6.1f} MB")
    return Timing(f"encode@{out_height}", count, elapsed)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def bench_agreement(
    collected: dict[int, list], reference: int, candidates: list[int]
) -> dict:
    ref_frames = collected[reference]
    report = {}
    for imgsz in candidates:
        frames = collected[imgsz]
        total = matched = 0
        by_class: dict[int, list[int]] = {}
        by_size: dict[str, list[int]] = {
            "<80": [0, 0], "80-160": [0, 0], "160-320": [0, 0], ">320": [0, 0],
        }
        for ref, cand in zip(ref_frames, frames):
            if len(ref) == 0:
                continue
            cand_boxes = cand.xyxy if len(cand) else np.empty((0, 4))
            cand_cls = cand.class_id if len(cand) else np.empty((0,))
            for index in range(len(ref)):
                box = ref.xyxy[index]
                cls = int(ref.class_id[index]) if ref.class_id is not None else 2
                total += 1
                hit = any(
                    int(cand_cls[j]) == cls and _iou(box, cand_boxes[j]) >= 0.5
                    for j in range(len(cand_boxes))
                )
                matched += int(hit)
                slot = by_class.setdefault(cls, [0, 0])
                slot[0] += int(hit)
                slot[1] += 1
                width = float(box[2] - box[0])
                bucket = ("<80" if width < 80 else "80-160" if width < 160
                          else "160-320" if width < 320 else ">320")
                by_size[bucket][0] += int(hit)
                by_size[bucket][1] += 1
        report[imgsz] = {
            "recall": matched / max(1, total),
            "matched": matched,
            "total": total,
            "by_class": {k: v[0] / max(1, v[1]) for k, v in by_class.items()},
            "by_size": {
                k: (v[0] / v[1] if v[1] else None) for k, v in by_size.items()
            },
            "size_counts": {k: v[1] for k, v in by_size.items()},
        }
        print(f"  recall {imgsz} vs {reference}: {report[imgsz]['recall']:.3f} "
              f"({matched}/{total})")
        for bucket, (hit, tot) in sorted(by_size.items()):
            if tot:
                print(f"      lebar {bucket:>8} px: {hit}/{tot} = {hit / tot:.2f}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--encode-frames", type=int, default=60)
    parser.add_argument("--sizes", default="640,960,1280,1536")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--reference", type=int, default=1536)
    parser.add_argument("--skip", default="")
    args = parser.parse_args()

    path = Path(args.video).resolve()
    info = probe_video(path)
    sizes = [int(v) for v in args.sizes.split(",") if v.strip()]
    skip = {v.strip() for v in args.skip.split(",") if v.strip()}
    print(f"video {path.name}: {info.width}x{info.height} "
          f"{info.total_frames} frame @ {info.fps:.2f} fps")

    results: dict[str, object] = {"video": str(path), "conf": args.conf}

    if "decode" not in skip:
        results["decode"] = vars(bench_decode(path, args.frames))

    collected = {}
    if "detect" not in skip:
        timings, collected = bench_detect(
            path, args.frames, sizes, args.conf, keep=True
        )
        results["detect"] = {t.label: round(t.per_frame_ms, 2) for t in timings}
        if args.reference in collected:
            print("  --- kecocokan deteksi relatif ---")
            candidates = [s for s in sizes if s != args.reference]
            results["agreement"] = bench_agreement(
                collected, args.reference, candidates
            )
        del collected
    if "annotate" not in skip:
        results["annotate"] = round(
            bench_annotate(path, args.frames, sizes[0], args.conf).per_frame_ms, 2
        )
    if "encode" not in skip:
        heights = [info.height]
        if info.height > 1080:
            heights.append(1080)
        results["encode"] = {
            f"{h}p": round(bench_encode(path, args.encode_frames, h).per_frame_ms, 2)
            for h in heights
        }

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

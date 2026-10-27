"""Fine-tune YOLO11 on a traffic dataset and validate the best checkpoint.

The shipped application uses COCO class IDs ``2, 3, 5, 7`` (car, motorcycle,
bus, truck). A custom four-class dataset normally uses ``0, 1, 2, 3``. If you
fine-tune, update the dashboard/API class mapping as well, otherwise the new
model will receive the wrong class IDs and detect nothing.

Examples
--------
::

    python scripts/train_yolo.py \
        --data configs/traffic-data.yaml \
        --weights yolo11n.pt \
        --imgsz 960 --batch 4 --epochs 100

The run directory contains the training curves, ``results.csv``, the confusion
matrix, and ``weights/best.pt``. Keep raw footage and labelled plates out of git;
``runs/`` and ``datasets/`` are ignored by default.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset YAML")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="", help="'', 'cpu', '0', ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="traffic-yolo11n")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cache",
        action="store_true",
        help="cache images in RAM; only for small datasets on a 4 GB host",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable AMP; leave off on Pascal if training is unstable",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    if not data_path.exists():
        raise SystemExit(f"dataset YAML not found: {data_path}")
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    from ultralytics import YOLO

    print(f"dataset : {data_path.resolve()}")
    print(f"weights : {weights_path.resolve()}")
    print(f"sha256  : {_sha256(weights_path)}")
    print(f"imgsz   : {args.imgsz}   batch: {args.batch}   epochs: {args.epochs}")
    print("class IDs: a custom 4-class model uses 0-3; update the app mapping.")

    model = YOLO(str(weights_path))
    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=args.device or None,
        seed=args.seed,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=args.cache,
        amp=args.amp,
        plots=True,
        val=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"training finished but best.pt was not found: {best}")

    print(f"\nValidating best checkpoint: {best}")
    best_model = YOLO(str(best))
    metrics = best_model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        # Validation defaults: no confidence cut-off, standard NMS IoU, so AP is
        # not truncated at a serving threshold. Use scripts/eval_yolo.py for
        # per-class tables and an operating-point confusion matrix.
        conf=0.001,
        iou=0.7,
        device=args.device or None,
        plots=True,
        project=args.project,
        name=f"{args.name}-val",
    )

    results = metrics.results_dict
    print("\nBest-checkpoint metrics")
    print(f"  precision : {results.get('metrics/precision(B)', 0.0):.4f}")
    print(f"  recall    : {results.get('metrics/recall(B)', 0.0):.4f}")
    print(f"  mAP50     : {results.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"  mAP50-95  : {results.get('metrics/mAP50-95(B)', 0.0):.4f}")
    print(f"  artifacts : {Path(metrics.save_dir).resolve()}")
    print(
        "\nNext: record the dataset hash/split, update MODEL_WEIGHTS, and change "
        "the dashboard/API class IDs to match the custom model."
    )


if __name__ == "__main__":
    main()

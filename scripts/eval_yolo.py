"""Evaluate a YOLO detector and print reproducible detection metrics.

This is a thin, explicit wrapper around Ultralytics ``model.val()``. It does not
train anything and it never invents numbers: everything printed here comes from
the validation run.

Examples
--------
COCO val2017, only the four classes used by the pipeline::

    python scripts/eval_yolo.py \
        --data configs/coco-val2017.yaml \
        --weights yolo11n.pt \
        --classes 2,3,5,7 \
        --imgsz 640 \
        --name coco-val2017

A fine-tuned four-class traffic model (class IDs 0-3)::

    python scripts/eval_yolo.py \
        --data configs/traffic-data.yaml \
        --weights runs/train/traffic/weights/best.pt \
        --classes 0,1,2,3 \
        --imgsz 960 \
        --name traffic-val

``--classes`` restricts both ground truth and predictions to the listed class
IDs, so the numbers describe the four vehicle classes only. The run directory
then also contains a compact ``confusion_matrix-selected.png``/``.csv``,
``curves-selected.png``, and ``results-selected.csv`` for those classes.
Ultralytics also writes ``BoxPR_curve.png``, ``BoxF1_curve.png``,
``BoxP_curve.png`` and ``BoxR_curve.png``; because the ground truth is filtered,
those curves contain only the selected classes.

The defaults (``--conf 0.001 --iou 0.7``) are Ultralytics' validation defaults.
Keep them for AP/mAP and PR curves: raising ``conf`` drops low-confidence
predictions and truncates recall, so the reported mAP would depend on that
cut-off. Per-class P/R/F1 are best-F1 values; the operating threshold for each
class is printed after the run.

Do **not** commit ``val_batch*.jpg``: those are dataset images and may contain
people or plates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_classes(raw: str) -> list[int] | None:
    raw = raw.strip()
    if not raw:
        return None
    return [int(item) for item in raw.replace(" ", "").split(",") if item]


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return names.get(class_id, str(class_id))
    try:
        return names[class_id]
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def _make_filtered_validator(classes: list[int]):
    """Return a DetectionValidator that keeps only ``classes``.

    Ultralytics 8.4 does not apply the ``classes`` argument to the validation
    postprocess, so we filter ground truth and predictions ourselves. This
    mirrors what the application does at inference time.
    """

    import torch
    from ultralytics.models.yolo.detect.val import DetectionValidator

    keep = torch.tensor(classes, dtype=torch.long)

    class FilteredValidator(DetectionValidator):
        def _class_mask(self, tensor):
            return torch.isin(tensor.long(), keep.to(tensor.device))

        def _prepare_batch(self, si, batch):
            prepared = super()._prepare_batch(si, batch)
            if prepared["cls"].numel():
                mask = self._class_mask(prepared["cls"])
                prepared["cls"] = prepared["cls"][mask]
                prepared["bboxes"] = prepared["bboxes"][mask]
            return prepared

        def postprocess(self, preds):
            outputs = super().postprocess(preds)
            for output in outputs:
                if output["cls"].numel():
                    mask = self._class_mask(output["cls"])
                    for key in list(output):
                        output[key] = output[key][mask]
            return outputs

    return FilteredValidator


def _selected_indices(box, classes: list[int] | None) -> list[int]:
    class_ids = [int(item) for item in box.ap_class_index]
    if classes is None:
        return list(range(len(class_ids)))
    selected = [index for index, class_id in enumerate(class_ids) if class_id in classes]
    if not selected:
        raise SystemExit("none of the requested classes have metrics; check --classes")
    missing = [class_id for class_id in classes if class_id not in class_ids]
    if missing:
        print(f"warning : no ground truth or predictions for classes {missing}")
    return selected


def _best_f1_confidence(box, index: int) -> float:
    curve = box.f1_curve[index]
    px = box.px
    best = max(range(len(curve)), key=lambda i: curve[i])
    return float(px[best])


def _write_selected_results(save_dir: Path, box, names, selected: list[int]) -> None:
    class_ids = [int(item) for item in box.ap_class_index]
    rows = []
    for index in selected:
        p = float(box.p[index])
        r = float(box.r[index])
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        rows.append(
            {
                "class_id": class_ids[index],
                "class_name": _class_name(names, class_ids[index]),
                "precision": p,
                "recall": r,
                "f1": f1,
                "f1_conf": _best_f1_confidence(box, index),
                "map50": float(box.ap50[index]),
                "map50_95": float(box.ap[index]),
            }
        )

    macro_p = sum(row["precision"] for row in rows) / len(rows)
    macro_r = sum(row["recall"] for row in rows) / len(rows)
    macro_f1 = sum(row["f1"] for row in rows) / len(rows)
    macro_map50 = sum(row["map50"] for row in rows) / len(rows)
    macro_map = sum(row["map50_95"] for row in rows) / len(rows)

    with (save_dir / "results-selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1",
                "f1_conf",
                "map50",
                "map50_95",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: _fmt(value) if isinstance(value, float) else value for key, value in row.items()}
            )
        writer.writerow(
            {
                "class_id": "macro",
                "class_name": f"{len(rows)} classes",
                "precision": _fmt(macro_p),
                "recall": _fmt(macro_r),
                "f1": _fmt(macro_f1),
                "f1_conf": "",
                "map50": _fmt(macro_map50),
                "map50_95": _fmt(macro_map),
            }
        )


def _write_confusion_matrix(save_dir: Path, matrix, names, classes: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.asarray(matrix)
    background = matrix.shape[0] - 1
    keep = [class_id for class_id in classes if 0 <= class_id < background]
    if background not in keep:
        keep.append(background)
    subset = matrix[np.ix_(keep, keep)]
    labels = [
        _class_name(names, class_id) if class_id != background else "background"
        for class_id in keep
    ]

    with (save_dir / "confusion_matrix-selected.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["predicted\\true", *labels])
        for label, row in zip(labels, subset):
            writer.writerow([label, *[int(value) for value in row]])

    totals = subset.sum(axis=0, keepdims=True)
    normalized = subset / np.where(totals == 0, 1, totals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    panels = (
        (axes[0], subset, "Counts", "d"),
        (axes[1], normalized, "Normalized by true class", ".2f"),
    )
    for axis, array, title, kind in panels:
        peak = float(array.max()) if array.size else 1.0
        image = axis.imshow(array, cmap="Blues", vmin=0, vmax=peak or 1.0, interpolation="none")
        for row in range(array.shape[0]):
            for column in range(array.shape[1]):
                value = float(array[row, column])
                text = f"{int(value)}" if kind == "d" else f"{value:.2f}"
                axis.text(
                    column,
                    row,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if peak and value > peak * 0.5 else "black",
                )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrix - selected classes")
    fig.tight_layout()
    fig.savefig(save_dir / "confusion_matrix-selected.png", dpi=200)
    plt.close(fig)


def _write_curves(save_dir: Path, box, names, selected: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    class_ids = [int(item) for item in box.ap_class_index]
    precision = np.asarray(box.p_curve)[selected]
    recall = np.asarray(box.r_curve)[selected]
    f1 = np.asarray(box.f1_curve)[selected]
    px = np.asarray(box.px)
    if precision.ndim == 1:
        precision = precision[None, :]
        recall = recall[None, :]
        f1 = f1[None, :]

    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(len(selected), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for row, index in enumerate(selected):
        label = f"{class_ids[index]} {_class_name(names, class_ids[index])}"
        axes[0].plot(recall[row], precision[row], color=colors[row], label=label)
        axes[1].plot(px, f1[row], color=colors[row], label=label)
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("Precision-recall")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.01)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Confidence")
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 vs confidence")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.01)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_dir / "curves-selected.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="dataset YAML with a val split")
    parser.add_argument("--weights", default="yolo11n.pt", help="model checkpoint")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    # Ultralytics validation defaults: do not truncate low-confidence
    # predictions, otherwise AP and the PR curve are cut at the operating point.
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="", help="'', 'cpu', '0', ...")
    parser.add_argument(
        "--classes",
        default="",
        help="comma-separated class IDs to evaluate; empty = all model classes",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--matrix-conf",
        type=float,
        default=0.25,
        help="confidence floor for the compact confusion matrix (an operating point, "
        "not an AP threshold)",
    )
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="val")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data)
    weights_path = Path(args.weights)
    if not data_path.exists():
        raise SystemExit(f"dataset YAML not found: {data_path}")
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    from ultralytics import YOLO

    classes = _parse_classes(args.classes)
    print(f"dataset : {data_path.resolve()}")
    print(f"weights : {weights_path.resolve()}")
    print(f"sha256  : {_sha256(weights_path)}")
    print(f"imgsz   : {args.imgsz}   batch: {args.batch}   conf: {args.conf}   iou: {args.iou}")
    print(f"classes : {classes if classes is not None else 'all'}")

    model = YOLO(str(weights_path))
    validator = _make_filtered_validator(classes) if classes else None
    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        classes=classes,
        workers=args.workers,
        plots=not args.no_plots,
        project=args.project,
        name=args.name,
        verbose=True,
        validator=validator,
    )

    box = metrics.box
    names = metrics.names
    selected = _selected_indices(box, classes)
    class_ids = [int(item) for item in box.ap_class_index]

    print("\nSelected classes")
    print("  id  class            P        R        F1       conf@F1   mAP50    mAP50-95")
    for index in selected:
        p = float(box.p[index])
        r = float(box.r[index])
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        print(
            f"  {class_ids[index]:>2}  {_class_name(names, class_ids[index]):<15} "
            f"{_fmt(p)}  {_fmt(r)}  {_fmt(f1)}  "
            f"{_fmt(_best_f1_confidence(box, index))}  "
            f"{_fmt(float(box.ap50[index]))}  {_fmt(float(box.ap[index]))}"
        )

    macro_p = sum(float(box.p[index]) for index in selected) / len(selected)
    macro_r = sum(float(box.r[index]) for index in selected) / len(selected)
    # Mean of per-class F1, matching the macro row in results-selected.csv.
    macro_f1 = sum(
        2 * float(box.p[index]) * float(box.r[index])
        / (float(box.p[index]) + float(box.r[index]))
        if float(box.p[index]) + float(box.r[index]) > 0
        else 0.0
        for index in selected
    ) / len(selected)
    macro_map50 = sum(float(box.ap50[index]) for index in selected) / len(selected)
    macro_map = sum(float(box.ap[index]) for index in selected) / len(selected)

    print("\nSelected-class macro average")
    print(f"  precision : {_fmt(macro_p)}")
    print(f"  recall    : {_fmt(macro_r)}")
    print(f"  F1        : {_fmt(macro_f1)}")
    print(f"  mAP50     : {_fmt(macro_map50)}")
    print(f"  mAP50-95  : {_fmt(macro_map)}")
    print("  note      : P/R are Ultralytics' best-F1 operating point;")
    print("              mAP is threshold-independent.")

    if classes is None:
        print(f"\nFull {len(box.ap_class_index)}-class result")
        print(f"  precision : {_fmt(box.mp)}")
        print(f"  recall    : {_fmt(box.mr)}")
        print(f"  mAP50     : {_fmt(box.map50)}")
        print(f"  mAP50-95  : {_fmt(box.map)}")

    speed = metrics.speed or {}
    if speed:
        print("\nSpeed (ms/image)")
        for key, value in speed.items():
            print(f"  {key:<8}: {float(value):.2f}")

    save_dir = Path(metrics.save_dir)
    if classes:
        matrix = metrics.confusion_matrix.matrix
        if abs(args.matrix_conf - args.conf) > 1e-9:
            # AP is computed without a confidence cut-off, but a matrix at
            # conf=0.001 is swamped by background predictions. Re-run only to
            # count matches at the requested operating point.
            second = model.val(
                data=str(data_path),
                imgsz=args.imgsz,
                batch=args.batch,
                conf=args.matrix_conf,
                iou=args.iou,
                device=args.device or None,
                classes=classes,
                workers=args.workers,
                plots=True,
                project=args.project,
                name=f"{args.name}-matrix",
                verbose=False,
                validator=_make_filtered_validator(classes),
            )
            matrix = second.confusion_matrix.matrix
            shutil.rmtree(Path(second.save_dir), ignore_errors=True)
        _write_selected_results(save_dir, box, names, selected)
        _write_confusion_matrix(save_dir, matrix, names, classes)
        _write_curves(save_dir, box, names, selected)
        print(f"\nConfusion matrix counts use predictions with conf >= {args.matrix_conf}.")

    print(f"\nArtifacts: {save_dir.resolve()}")
    filenames = [
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ]
    if classes:
        filenames += [
            "results-selected.csv",
            "confusion_matrix-selected.csv",
            "confusion_matrix-selected.png",
            "curves-selected.png",
        ]
    for filename in filenames:
        path = save_dir / filename
        print(f"  {'ok ' if path.exists() else '-- '} {path.name}")
    print("\nDo not commit val_batch*.jpg: those images come from the dataset.")


if __name__ == "__main__":
    main()

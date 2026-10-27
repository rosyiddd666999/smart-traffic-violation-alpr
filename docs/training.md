# Training and fine-tuning YOLO11

This guide describes how to fine-tune the detector on your own traffic footage.
It is a **process guide**, not a claim that the shipped model was trained on this
deployment.

## 1. Model provenance

The application ships with the official Ultralytics **`yolo11n.pt` COCO
checkpoint**. It is not fine-tuned on the reference deployment, and the
repository contains no custom training run or training curves. The only
published confusion matrix is the COCO val2017 one in
[`model-evaluation.md`](model-evaluation.md); there is no domain matrix.

The app filters the COCO model to four classes:

| COCO ID | Class |
|---:|---|
| 2 | car |
| 3 | motorcycle |
| 5 | bus |
| 7 | truck |

The published COCO val2017 check in
[`model-evaluation.md`](model-evaluation.md) measures those four classes on the
official COCO holdout. It is a sanity check, not field accuracy.

## 2. When fine-tuning is worth it

Fine-tuning helps when the pretrained model misses vehicles in your scenes, for
example:

- fixed CCTV angles with strong perspective;
- night, rain, glare, or motion blur;
- small or partially occluded vehicles;
- local vehicle types that COCO represents poorly;
- a scene where false positives (road furniture, shadows) are common.

Collect a validation set first. If the pretrained model already meets your
recall target, fine-tuning may not be worth the labelling effort.

## 3. Data collection and privacy

- Use footage you are allowed to process. Do not publish raw clips that contain
  real plates or identifiable people.
- Keep datasets outside the repository. `datasets/`, `samples/`, and `runs/` are
  git-ignored; do not force-add them.
- Store labels and crops in a private location. A label file contains bounding
  boxes, not plate text, but the images still identify vehicles.
- If you publish examples, redact plate pixels and replace plate strings with
  synthetic values, as the demo seed does.

## 4. Labelling

Use four classes with contiguous IDs:

```text
0 car
1 motorcycle
2 bus
3 truck
```

Guidelines:

- Draw a tight box around the visible vehicle body, not the shadow.
- Include partially occluded vehicles when a human can still identify them.
- Exclude vehicles that are too small to label consistently.
- Include empty/negative frames so the model learns the road background.
- Keep one label file per image in YOLO format:
  `class_id x_center y_center width height`, normalised to `0..1`.
- Review a random sample of labels before training; label noise is the most
  common cause of a disappointing run.

Tools such as CVAT, Label Studio, or Roboflow can export YOLO format. Whatever
you use, record the dataset version and export date.

## 5. Split by scene, not by frame

Random frame splits leak near-identical frames into validation and inflate the
metrics. Split by **video, camera, or scene**:

```text
70-80% train
10-15% validation
10-15% test (optional, held out until the end)
```

Keep the split file under version control (the list of clips and their split),
but not the images. Use a fixed seed if you split programmatically.

## 6. Dataset YAML

Copy [`configs/traffic-data.example.yaml`](../configs/traffic-data.example.yaml)
to `configs/traffic-data.yaml` and edit `path`:

```yaml
path: /datasets/traffic
train: images/train
val: images/val

names:
  0: car
  1: motorcycle
  2: bus
  3: truck
```

Expected layout:

```text
/datasets/traffic/
├── images/train/*.jpg
├── images/val/*.jpg
├── labels/train/*.txt
└── labels/val/*.txt
```

## 7. Training

On the reference GTX 1050 Ti (4 GB), start conservatively:

```bash
python scripts/train_yolo.py \
  --data configs/traffic-data.yaml \
  --weights yolo11n.pt \
  --imgsz 960 \
  --batch 4 \
  --epochs 100 \
  --patience 20 \
  --workers 2 \
  --device 0 \
  --name traffic-yolo11n
```

Notes:

- `imgsz=960` matches the warm inference size used by the app. If VRAM is tight,
  train at 640 and evaluate at 960; do not mix resolutions without recording it.
- `batch=4` is a safe starting point for 4 GB. Increase only if VRAM allows.
- Leave `--amp` off on Pascal if you see NaNs or unstable losses. The wrapper
  defaults to AMP off.
- `--cache` loads images into RAM; only use it for a small dataset.
- Early stopping via `--patience` is enabled by default in the wrapper.
- Training writes to `runs/train/...`, which is git-ignored.

The wrapper runs `model.val()` on `weights/best.pt` after training and prints
precision, recall, mAP50, and mAP50-95.

## 8. Validation artifacts

The run directory contains:

```text
results.csv
args.yaml
confusion_matrix.png
confusion_matrix_normalized.png
BoxPR_curve.png
BoxF1_curve.png
BoxP_curve.png
BoxR_curve.png
weights/best.pt
weights/last.pt
```

Review:

- **Per-class recall.** A high overall mAP can hide a weak class (often bus or
  truck).
- **Confusion matrix.** Look for car↔truck and motorcycle↔bicycle confusion.
- **PR curve.** Check the operating point you actually need; the dashboard uses
  `CONF_THRESHOLD=0.35` by default.
- **Failure images.** Inspect false negatives at night, in rain, and at the
  frame edges.
- **Validation loss.** If it rises while training loss falls, you are
  overfitting; add data or augmentation, or stop earlier.

Do not publish `val_batch*.jpg` or training batches: they contain dataset images.

## 9. Class-ID mapping (important)

The app currently sends COCO class IDs `[2, 3, 5, 7]` from the dashboard and
`pipeline.py`. A custom four-class model outputs `0, 1, 2, 3`. If you swap the
weights without updating the mapping, the model receives the wrong class filter
and detects nothing.

Two options:

1. **Change the app mapping** to `0, 1, 2, 3` in `DEFAULT_CLASSES`, the dashboard
   checkbox values, and the API examples. Update the docs and tests.
2. **Add a model-to-UI mapping** in `pipeline.py`: translate the UI's logical
   classes to the model's class IDs before calling `model.track()`, and translate
   the model's names back for display. This keeps the API stable but is a code
   change.

Do not train an 80-class model with only four populated classes just to keep the
old IDs; it wastes output channels and confuses maintenance.

## 10. Deploying a fine-tuned model

1. Copy `best.pt` to the host, outside git, for example
   `/srv/traffic-alpr/models/traffic-yolo11n.pt`.
2. Set `MODEL_WEIGHTS=/models/traffic-yolo11n.pt` (or the path inside the
   container) in `.env`.
3. Mount the model into the app container if it is not baked into the image.
4. Update the class mapping as described above.
5. Restart the app and check `/health`.
6. Run a known clip end to end and compare evidence with the baseline.
7. Keep the previous weights so you can roll back.

The app loads the model at startup; a restart is required.

## 11. Reproducibility checklist

Record the following with every reported result:

- dataset name, version, hash, and split file;
- number of images and instances per class;
- Ultralytics and PyTorch versions;
- GPU model and driver/CUDA version;
- exact training command and `args.yaml`;
- random seed;
- best epoch and early-stopping state;
- precision, recall, F1, mAP50, mAP50-95 per class;
- confusion matrix and PR/F1 curves;
- known failure cases.

If you cannot reproduce a number from the saved artifacts, do not publish it.

## 12. What not to claim

- Do not call COCO val2017 results “traffic accuracy” or “field accuracy”.
- Do not call F1 “accuracy”; F1 is threshold-dependent.
- Do not publish a confusion matrix from a different model or dataset than the
  weights you ship.
- Do not publish training curves unless the run actually happened.
- Do not commit raw footage, labelled plates, or dataset images.

For the evaluation side, see [`model-evaluation.md`](model-evaluation.md).

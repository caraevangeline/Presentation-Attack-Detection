# Presentation-Attack-Detection

![Python](https://img.shields.io/badge/python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.11-EE4C2C?logo=pytorch&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18-005CED?logo=onnx&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)

Screen-attack presentation attack detection (PAD): given a face image,
decide whether it was captured from a live, bona fide subject or recaptured
from a screen (phone, tablet, monitor) held up to the camera.

## Primary method: MiniFASNet

**[`MiniFASNet/`](MiniFASNet/)** is the recommended method in this repo: a
face-detect + crop + classify pipeline built around
[minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)'s
MiniFASNetV2 architecture, fine-tuned on this project's own real/spoof
dataset and exported to ONNX for deployment.

### Run inference (Docker, no local setup)

```bash
docker build -f MiniFASNet/Dockerfile -t minifasnet-pad .
docker run --rm \
    -v /path/to/your/images:/data/input:ro \
    -v /path/to/output:/data/output \
    minifasnet-pad
```

Mounts a folder of images in, writes annotated images (green box = bona
fide, red box = spoof, with confidence) plus a `results.csv` (one row per
detected face) out. See [`MiniFASNet/Dockerfile`](MiniFASNet/Dockerfile) for
build/run details and how to override any `infer.py` flag (decision
threshold, crop scale, etc.) at `docker run` time.

### Run inference (local Python)

```bash
cd MiniFASNet
pip install -r requirements-infer.txt

# single image
python infer.py --input_image path/to/photo.jpg --output_image annotated.jpg \
    --pad_model artifacts/m2.v7/model.onnx --pad_config artifacts/m2.v7/config.json

# a folder of images -> annotated folder + CSV
python infer.py --input_dir path/to/images --output_dir annotated/ \
    --pad_model artifacts/m2.v7/model.onnx --pad_config artifacts/m2.v7/config.json
```

For scoring a folder against ground truth and getting the full PAD metrics
table, confusion matrices, and ROC/PR curves instead of annotated images,
use `val_onnx.py` (same interface as `infer.py`'s batch mode, but built for
evaluation rather than visual output):

```bash
python val_onnx.py --input_dir . --output_csv scores.csv \
    --labeled_dir /path/to/labeled_data --model_dir artifacts/m2.v7
```

See [`MiniFASNet/README.md`](MiniFASNet/README.md) for the full writeup:
why this architecture, the fine-tuning recipe, known limitations (including
an honest look at a cross-dataset generalization gap found during
evaluation), and how fine-tuning itself works (requires separately cloning
the upstream `Silent-Face-Anti-Spoofing` repo; not bundled here since the
shipped inference path only needs the exported ONNX model).

## Exploratory methods

Two other approaches were built alongside MiniFASNet as points of
comparison, not as competing production candidates:

- **[`FFT_SVM/`](FFT_SVM/)**: a hand-crafted-feature baseline. Scores each
  image with an RBF-SVM over FFT-derived features (radial power spectrum,
  spectral peakiness, per-channel color-moire signal) that target the
  physical mechanism behind screen recapture: a display's fixed sub-pixel
  grid aliasing against the camera sensor's grid. Far more sample-efficient
  than a CNN when attack examples are scarce; see its README for why this
  was the first thing built here.
- **[`DeepFace/`](DeepFace/)**: wraps
  [DeepFace](https://github.com/serengil/deepface)'s off-the-shelf
  anti-spoofing model (itself a MiniFASNet ensemble, run unmodified, no
  fine-tuning on this project's data) as a reference point for what the
  original, properly-calibrated model gets without any of this repo's
  fine-tuning choices in play.

## Repo structure

```
Presentation-Attack-Detection/
├── MiniFASNet/       primary method: fine-tuned MiniFASNetV2 + ONNX export + Docker
├── FFT_SVM/           exploratory: hand-crafted FFT features + SVM
├── DeepFace/          exploratory: off-the-shelf DeepFace anti-spoofing wrapper
├── common/            pad_eval.py: shared PAD metrics/plotting, used by all three methods' val.py
└── yolov8_face/        YOLOv8-nano face detector (ONNX) + crop utilities, shared by MiniFASNet
```

## Shared evaluation methodology

All three methods' `val.py`/`val_onnx.py` scripts share the same evaluation
code (`common/pad_eval.py`), so their results are directly comparable: given
predicted attack scores and ground truth, it computes ROC-AUC, PR-AUC, EER,
and APCER/BPCER/ACER/precision/recall/F1 at both a configured decision
threshold and the EER threshold, and renders a metrics table + confusion
matrices (`metrics_summary.png`) plus ROC/PR curves as images alongside the
raw CSV/JSON.

## Known limitations

- Evaluating the fine-tuned MiniFASNet model against a genuinely
  independent, held-out test set (not used in training or validation)
  surfaced a real generalization gap relative to its own in-distribution
  validation numbers. See [`MiniFASNet/README.md`](MiniFASNet/README.md)
  for the specifics and the crop-scale mismatch that likely contributes to
  it.
- None of the three methods here have been validated against video replay
  attacks or paper/print attacks; all evaluation has been against still
  screen recaptures.
- Face detection (YOLOv8-nano) and the PAD classifiers are separate models
  run in sequence; a missed or low-confidence face detection means no PAD
  score at all for that image, not a fallback "unsure" verdict.

## Acknowledgments

`MiniFASNet/` fine-tunes and vendors architecture code from
[minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(Apache-2.0). `DeepFace/` depends on
[serengil/deepface](https://github.com/serengil/deepface).

## License

No license has been specified for this repository yet. The vendored
Silent-Face-Anti-Spoofing architecture/weights this project fine-tunes from
are Apache-2.0 licensed by minivision-ai.

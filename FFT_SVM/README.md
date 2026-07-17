# Method 1: FFT + SVM Screen-Attack Detector

## What this is

An add-on baseline experiment for the Screen Attack Detection PAD exercise.
It scores each image with `P(screen attack)` using hand-crafted FFT
features and an RBF-SVM.

## Why FFT + SVM, not a CNN

The attack class in this dataset is much smaller than the bona fide
class. That kind of imbalance is not enough to fine-tune a deep net (e.g.
EfficientNet) without either severe overfitting or leaning on synthetic
augmentation whose realism is unverified. A small (~40-dim) hand-crafted
feature vector plus an SVM is far more sample-efficient, trains in
seconds, and runs inference in milliseconds per image on CPU: a
reasonable first baseline given how little attack data there is to learn
from.

The features target a physical mechanism specific to screen recapture:
an LCD/OLED's fixed sub-pixel grid aliasing against the camera sensor's
pixel grid (moire), which shows up as sharp peaks in the 2D FFT spectrum
instead of the smooth ~1/f falloff of natural images. See `features.py`
for the full rationale and feature list (radial power spectrum bins,
spectral peakiness, high-frequency energy ratio, spectral slope,
per-channel color-moire peakiness, a blur/sharpness proxy).

Deliberately excluded: raw image resolution and aspect ratio. In the
originally supplied data, bona fide images were all exactly 1024x1024
while attack images had arbitrary native resolutions, almost certainly a
dataset-assembly artifact rather than a real signal. Everything is
resized to a fixed 384x384 before feature extraction so the model cannot
shortcut on this.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.8/3.10, opencv-python-headless, scikit-learn 1.3.

## Train

```bash
python train.py --data_dir /path/to/data --output_dir ./artifacts
```

Expects `<data_dir>/real/*` and `<data_dir>/spoof/*`.

This runs a **nested cross-validation**: an outer 5-fold split gives an
honest, out-of-fold performance estimate, while an inner 3-fold grid
search (over SVM `C`/`gamma`) picks hyperparameters inside each outer
training fold, so the reported metrics are not contaminated by tuning on
the same data they are evaluated on. A separate grid search is then
refit on *all* data to produce the deployed model in `artifacts/`.

Outputs:
- `artifacts/model.joblib`: fitted `StandardScaler` + `SVC` pipeline
- `artifacts/config.json`: feature names, decision threshold, chosen hyperparameters
- `artifacts/cv_metrics.json`: nested-CV performance numbers

## Inference

```bash
python val.py --input_dir /path/to/images --output_csv scores.csv \
    --model_dir ./artifacts
```

Writes a CSV with `filename, attack_score, predicted_label, error` for
every image in the input folder (non-recursive, `.png/.jpg/.jpeg/.bmp`).
`attack_score` is `P(attack)` in `[0, 1]`; `predicted_label` applies the
threshold stored in `config.json`.

## Evaluation mode

If you have ground truth, `val.py` can additionally compute PAD metrics
against it:

```bash
python val.py --input_dir . --output_csv scores.csv \
    --labeled_dir /path/to/data --model_dir ./artifacts
```

`--labeled_dir` points at a folder laid out like the training data
(`real/` and `spoof/` subfolders). For a flat folder instead, use
`--labels_csv labels.csv` with columns `filename,label`.

This writes to `--metrics_output_dir` (default: `<output_csv's folder>/eval`):
- `pad_metrics.csv` / `pad_metrics.json`: ROC-AUC, PR-AUC, EER, and
  APCER/BPCER/ACER/precision/recall/F1 at both the model's configured
  threshold and the EER threshold
- `metrics_summary.png`: the same table and confusion matrices rendered as an image
- `roc_curve.png`, `pr_curve.png`

The dataset-loading and metrics/plotting code behind this lives in
`../common/pad_eval.py`, shared across model folders so other methods can
reuse the same evaluation pipeline instead of duplicating it.

## Known limitations / what would break this

- The attack class is still much smaller than the bona fide class, so
  any single evaluation split carries real sampling uncertainty; treat
  point-estimate metrics with caution and prefer cross-validated numbers
  where possible.
- No face localization: features are computed on the whole image, so a
  screen attack with a very small or cropped visible screen, or a bona
  fide image with strong background texture or moire-like patterns (e.g.
  striped clothing, fine mesh, a monitor visible in the background),
  could confuse the frequency-domain cues this method relies on.
- Heavy re-compression or resizing of the input image after capture can
  suppress or alias the very moire signal this method looks for.
- This has not been validated against video replay attacks or
  paper-based attacks, only still screen recaptures.

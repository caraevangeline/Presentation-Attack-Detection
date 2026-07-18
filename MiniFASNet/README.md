# Method 2: Fine-tuned MiniFASNetV2 (Silent-Face-Anti-Spoofing)

## What this is

A second, model-based approach to the same Screen Attack Detection PAD
exercise as `../FFT_SVM`. It fine-tunes the `2.7_80x80_MiniFASNetV2`
checkpoint from
[minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
(vendored under `../Silent-Face-Anti-Spoofing`) on our own real/spoof data,
rather than training a network from scratch.

## Why fine-tune this checkpoint

`datasets/train/PAD-v6` face crops are produced by
`../yolov8_face/yolov8_face_detect_align.py` at a 1.5x bounding-box scale
margin (see `datasets/face_detection/celeba_real_crops_1.5` /
`celeba_spoof_crops_1.5`), not the 2.7x margin the `2.7_80x80_MiniFASNetV2`
checkpoint was originally trained on. The original repo only ships
checkpoints for 2.7x and 4x margins (`resources/anti_spoof_models/`), so
there is no exact match available for 1.5x; `2.7_80x80` is the closer of
the two shipped options in spirit (it still keeps a margin around the face,
enough to capture some screen bezel/background, unlike the tight 1x crop),
so that is what is fine-tuned here, with the scale mismatch left for the
fine-tuning itself to adapt to. See "Known limitations" below.

Training a MiniFASNet from scratch needs a dataset far larger than what we
have; our real/spoof counts are a small fraction of what the original
checkpoint was trained on. Starting from the pretrained weights keeps the
backbone's learned low-level texture and frequency features and only asks
the network to adapt them to this dataset, which is a much better fit for
the amount of data available here.

The repo's own training code (`MultiFTNet`) always builds the SE variant of
the backbone (`MiniFASNetV2SE`) together with an auxiliary Fourier-spectrum
regression head (`FTGenerator`), even though the shipped `2.7_80x80`
checkpoint is a plain `MiniFASNetV2` (no SE blocks, no FT head). Both
variants share the same backbone channel layout, differing only in the
SE-attention modules appended to the last block of each residual stage, so
`finetune.py` loads the pretrained weights into every layer name/shape that
matches (`load_pretrained_backbone`) and leaves the SE modules, the FT head,
and the final classifier (3 classes in the original checkpoint, 2 here:
real vs spoof, no print/replay subtypes) at random initialization to be
learned from our data.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.8, PyTorch 1.11 (CUDA 11.3), on the `core_ml` conda
environment. Also needs the vendored `../Silent-Face-Anti-Spoofing/src`
package and `../common/pad_eval.py` (both imported via `sys.path`, no
install step needed for either).

## Data prep

```bash
python data_prep.py --source_dir D:/assignment/datasets/train/PAD-v6 \
    --output_dir ./data --val_frac 0.15
```

Does a stratified real/spoof split into `data/train/{0,1}` and
`data/val/{0,1}` (class 0 = real, class 1 = spoof, matching
`torchvision.datasets.ImageFolder`'s alphabetical folder-name ordering),
skipping any image OpenCV cannot decode. The held-out `data/val` split is
used by `finetune.py` for per-epoch validation and checkpoint selection, not
just used at the end.

## Fine-tune

```bash
python finetune.py --data_dir ./data --epochs 20 --batch_size 128 --lr 1e-3
```

Each epoch trains with the original recipe's loss (0.5 * classification
cross-entropy + 0.5 * Fourier-spectrum MSE), then evaluates on the held-out
validation split. The classification loss is class-weighted (inversely
proportional to train-split class frequency) since spoof examples are the
minority class. The checkpoint with the lowest validation ACER (at that
epoch's EER threshold) is kept as the deployed model.

Outputs to `artifacts/`:
- `model.pth`: the best epoch's `MultiFTNet` state dict
- `config.json`: architecture params (`conv6_kernel`, `embedding_size`,
  `num_classes`), the decision threshold (validation EER threshold at the
  best epoch), and the full validation metrics at that epoch
- `training_history.json`: per-epoch train/val metrics, for inspecting the
  training curve

## Inference

```bash
python val.py --input_dir /path/to/images --output_csv scores.csv \
    --model_dir ./artifacts/m2.v7
```

Writes a CSV with `filename, attack_score, predicted_label, error` for
every image in the input folder (non-recursive, `.png/.jpg/.jpeg/.bmp`).
`attack_score` is `P(attack)` in `[0, 1]`; `predicted_label` applies the
threshold stored in `config.json`.

`MultiFTNet` is training-only scaffolding: its `FTGenerator` branch shapes
the backbone's learned features during training but does not itself produce
a prediction, and the FT loss/branch have no role once the model is trained.
So `val.py` does not build a `MultiFTNet` at all; `load_artifacts` builds
just the classification backbone (`MiniFASNetV2SE`) and loads the `model.*`
subset of the saved checkpoint into it (stripped of that prefix), discarding
the `FTGenerator.*` weights. This matches how the original repo's own
shipped checkpoints are packaged: `2.7_80x80_MiniFASNetV2.pth` and
`4_0_0_80x80_MiniFASNetV1SE.pth` are both backbone-only, no FT head.

## Evaluation mode

If you have ground truth, `val.py` can additionally compute PAD metrics
against it:

```bash
python val.py --input_dir . --output_csv scores.csv \
    --labeled_dir /path/to/data --model_dir ./artifacts/m2.v7
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

This is the same shared evaluation code `../FFT_SVM/val.py` uses
(`../common/pad_eval.py`), so both methods' results are directly comparable.

## Known limitations / what would break this

- **Crop-scale mismatch**: PAD-v6 crops are taken at a 1.5x bounding-box
  margin, but the checkpoint being fine-tuned was trained on 2.7x-margin
  crops. Full-network fine-tuning can adapt to this (early results show
  strong validation performance), but it means the pretrained backbone's
  learned spatial layout doesn't line up exactly with what it now sees at
  inference; a model fine-tuned on crops actually matching the checkpoint's
  original convention would likely transfer better. Not addressed here.
- Only the `2.7_80x80` scale is fine-tuned here. The original repo's
  deployed pipeline ensembles this with a second model at a wider `4x`
  crop scale; that would need face crops regenerated at that scale from
  original (uncropped) source images, which is not done here.
- The FT auxiliary loss and SE modules are trained from random
  initialization on a comparatively small dataset, so they are the parts
  of the network most at risk of overfitting; the validation-based
  checkpoint selection in `finetune.py` is there specifically to catch
  that rather than always taking the last epoch.
- No landmark-based face alignment is applied to the crops (deliberately,
  since rotation/interpolation can smear high-frequency moire signal, the
  same reasoning as the FFT method). A strongly off-axis or rotated face
  in the input may hurt accuracy.
- Like the FFT+SVM method, this has not been validated against video
  replay attacks or paper-based attacks, only still screen recaptures.

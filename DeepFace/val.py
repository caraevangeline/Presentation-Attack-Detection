"""Run DeepFace's off-the-shelf anti-spoofing model over a folder of images
and compute PAD metrics against ground truth. Mirrors ../FFT_SVM/val.py and
../MiniFASNet/val.py's CLI/eval-mode so all three methods can be compared
the same way.

Worth knowing: DeepFace's anti-spoofing backend (FasNet) is itself an
ensemble of two MiniFASNet models at 2.7x and 4x crop scales (see
site-packages/deepface/models/spoofing/FasNet.py), i.e. the same model
family and (likely) the same/similar upstream weights ../MiniFASNet fine-
tunes from -- this method is a useful "off-the-shelf, not fine-tuned on our
data" reference point against that one, not an unrelated third approach.

DeepFace's extract_faces returns "is_real" (bool) and "antispoof_score"
(confidence of *whichever* label won -- see FasNet.analyze: it's the
winning class's softmax probability, not a consistently-oriented P(real) or
P(attack)). To get a single continuous attack-likelihood score suitable for
ROC/PR curves and threshold sweeps, this file derives:
    attack_score = antispoof_score       if is_real is False
    attack_score = 1 - antispoof_score   if is_real is True
which is 0 for confidently-real, 1 for confidently-fake, ~0.5 for
uncertain -- and --threshold defaults to 0.5, which is where that derived
score flips sign relative to DeepFace's own is_real decision (its actual
decision boundary; there's no separate tunable threshold exposed by the
public API, unlike our own fine-tuned models).

Plain inference:
    python val.py --input_dir /path/to/images --output_csv scores.csv

Evaluation mode (optional, requires ground truth):
    python val.py --input_dir . --output_csv scores.csv \
        --labeled_dir /path/to/datasets/train/PAD-test-v1
    # or, for a flat folder with a separate labels file:
    python val.py --input_dir /path/to/images --output_csv scores.csv \
        --labels_csv labels.csv

If either --labeled_dir (a folder laid out like the training data, with
real/ and spoof/ subfolders) or --labels_csv (columns: filename, label) is
given, scores are additionally compared against ground truth and the
following are written to --metrics_output_dir (default: <output_csv's
folder>/eval): pad_metrics.csv / pad_metrics.json, metrics_summary.png,
roc_curve.png, pr_curve.png.

The dataset-loading and metrics/plotting code lives in ../common/pad_eval.py
and is shared with the other methods, not specific to this one.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from deepface import DeepFace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from common.pad_eval import (  # noqa: E402
    compute_pad_metrics,
    gather_labeled_dataset,
    list_images,
    load_labels_csv,
    rel_key,
    save_curves,
    save_metrics_image,
    save_metrics_table,
)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def score_images(image_paths, root):
    """rel_key -> (score or None, error or None)"""
    results = {}
    for path in image_paths:
        key = rel_key(path, root)
        try:
            faces = DeepFace.extract_faces(
                img_path=str(path), anti_spoofing=True, enforce_detection=False)
        except Exception as e:
            results[key] = (None, str(e))
            continue

        if len(faces) == 0:
            results[key] = (None, "NO_FACE_DETECTED")
            continue

        # first/largest detected face, matching detect.py's convention
        face = faces[0]
        is_real = face["is_real"]
        conf = face["antispoof_score"]
        attack_score = conf if not is_real else (1 - conf)
        results[key] = (float(attack_score), None)

    return results


def write_scores_csv(image_paths, results, threshold, output_csv, root):
    rows = []
    for path in image_paths:
        key = rel_key(path, root)
        score, err = results[key]
        rows.append({
            "filename": key,
            "attack_score": round(score, 6) if score is not None else "",
            "predicted_label": ("attack" if score >= threshold else "bonafide") if score is not None else "",
            "error": err or "",
        })
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "attack_score", "predicted_label", "error"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Folder of images to score")
    ap.add_argument("--output_csv", required=True, help="Where to write the scores CSV")
    ap.add_argument("--labeled_dir", default=None,
                    help="Optional eval mode: folder with real/ and spoof/ subfolders "
                         "(overrides --input_dir for which images get scored)")
    ap.add_argument("--labels_csv", default=None,
                    help="Optional eval mode: CSV with columns filename,label for images in --input_dir")
    ap.add_argument("--metrics_output_dir", default=None,
                    help="Where to save the PAD metrics table + ROC/PR curves "
                         "(default: <output_csv's folder>/eval)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Configured decision threshold on the derived attack_score "
                         "(default 0.5 matches DeepFace's own is_real decision boundary)")
    args = ap.parse_args()

    threshold = args.threshold

    labels = None
    if args.labeled_dir:
        root = args.labeled_dir
        image_paths, labels = gather_labeled_dataset(args.labeled_dir)
    else:
        root = args.input_dir
        image_paths = list_images(args.input_dir)
        if args.labels_csv:
            labels = load_labels_csv(args.labels_csv)

    print(f"Found {len(image_paths)} images")
    results = score_images(image_paths, root)
    rows = write_scores_csv(image_paths, results, threshold, args.output_csv, root)
    n_errors = sum(1 for r in rows if r["error"])
    print(f"Wrote {len(rows)} rows to {args.output_csv} ({n_errors} errors)")

    if labels is None:
        return

    y_true, y_score, missing = [], [], []
    for path in image_paths:
        key = rel_key(path, root)
        score, _ = results[key]
        if key not in labels:
            missing.append(key)
            continue
        if score is None:
            continue
        y_true.append(labels[key])
        y_score.append(score)

    if missing:
        print(f"WARNING: {len(missing)} images had no ground-truth label and were excluded from metrics")

    if len(set(y_true)) < 2:
        print("WARNING: need both classes present with valid scores to compute PAD metrics; skipping.")
        return

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    metrics = compute_pad_metrics(y_true, y_score, threshold)

    metrics_dir = args.metrics_output_dir or str(Path(args.output_csv).resolve().parent / "eval")
    csv_path, json_path = save_metrics_table(metrics, metrics_dir)
    img_path = save_metrics_image(metrics, metrics_dir)
    roc_path, pr_path = save_curves(y_true, y_score, metrics, metrics_dir)

    print(f"\n--- PAD metrics (n={len(y_true)}) ---")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics image (table + confusion matrices) to {img_path}")
    print(f"Saved metrics table to {csv_path} and {json_path}")
    print(f"Saved ROC curve to {roc_path}")
    print(f"Saved PR curve to {pr_path}")


if __name__ == "__main__":
    main()

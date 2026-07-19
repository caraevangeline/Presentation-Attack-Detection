"""Runs DeepFace's off-the-shelf anti-spoofing model over a labeled eval folder and computes PAD metrics
against ground truth. Mirrors ../FFT_SVM/val.py and ../MiniFASNet/val.py for consistent comparison.
DeepFace's backend is itself a MiniFASNet ensemble (2.7x + 4x crops), so this serves as an "off-the-shelf,
not fine-tuned" reference point.

Usage:
    python val.py --input_dir /path/to/PAD-test-v1 --output_csv scores.csv
    python val.py --input_dir /path/to/images --output_csv scores.csv --labels_csv labels.csv

--input_dir: needs real/spoof subfolders (or use --labels_csv for a flat folder)
--output_csv: Writes per-image scores
--metrics_output_dir (default: <output_csv folder>/eval): Saves metrics/plots
"""
from __future__ import annotations

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
    """Run DeepFace anti-spoofing over a list of images, returning a single continuous attack_score per image
    (0=confidently real, 1=confidently fake), keyed by path relative to `root`.
    Args:
        image_paths: list of image file paths to score.
        root: base directory used to compute each image's rel_key.
    """
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

        # first/largest detected face, matching infer.py's convention
        face = faces[0]
        is_real = face["is_real"]
        conf = face["antispoof_score"]
        attack_score = conf if not is_real else (1 - conf)
        results[key] = (float(attack_score), None)

    return results


def write_scores_csv(image_paths, results, threshold, output_csv, root):
    """Write per-image attack scores and threshold-based predictions to CSV.
    Args:
        image_paths: list of image file paths, in the order to write rows.
        results: dict from score_images(), keyed by rel_key -> (score, error).
        threshold: cutoff applied to attack_score to derive predicted_label.
        output_csv: path to write the CSV to (parent folders created if needed).
        root: base directory used to recompute each path's rel_key.
    """
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


def main(input_dir, output_csv, labels_csv, metrics_output_dir, threshold):
    """Score all images in input_dir, write results CSV, and (if ground truth is available via labels_csv or
    real/spoof subfolders) compute and save PAD metrics, plots, and a summary image.
    Args:
        input_dir: folder of images.
        output_csv: path to write per-image scores to.
        labels_csv: optional filename->label CSV for a flat input_dir.
        metrics_output_dir: folder for metrics/plots.
        threshold: cutoff applied to attack_score for predicted_label.
    """
    if labels_csv:
        root = input_dir
        image_paths = list_images(input_dir)
        labels = load_labels_csv(labels_csv)
    else:
        root = input_dir
        image_paths, labels = gather_labeled_dataset(input_dir)

    print(f"Found {len(image_paths)} images")
    results = score_images(image_paths, root)
    rows = write_scores_csv(image_paths, results, threshold, output_csv, root)
    n_errors = sum(1 for r in rows if r["error"])
    print(f"Wrote {len(rows)} rows to {output_csv} ({n_errors} errors)")

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

    metrics_dir = metrics_output_dir or str(Path(output_csv).resolve().parent / "eval")
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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True,
                     help="Eval folder with real/ and spoof/ subfolders "
                          "(or a flat folder of images if --labels_csv is given)")
    ap.add_argument("--output_csv", required=True, help="Where to write the scores CSV")
    ap.add_argument("--labels_csv", default=None,
                     help="Optional: CSV with columns filename,label for a flat --input_dir, "
                          "instead of the default real/spoof subfolder layout")
    ap.add_argument("--metrics_output_dir", default=None,
                     help="Where to save the PAD metrics table + ROC/PR curves "
                          "(default: <output_csv's folder>/eval)")
    ap.add_argument("--threshold", type=float, default=0.5,
                     help="Configured decision threshold on the derived attack_score "
                          "(default 0.5 matches DeepFace's own is_real decision boundary)")
    args = ap.parse_args()
    main(args.input_dir,
         args.output_csv,
         args.labels_csv,
         args.metrics_output_dir,
         args.threshold)

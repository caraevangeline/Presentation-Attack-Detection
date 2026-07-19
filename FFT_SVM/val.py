"""Runs the FFT + SVM screen-attack detector over a folder of images, optionally scored against ground truth.
Writes one row per image: filename, attack_score (0-1, higher = more likely a screen attack), predicted_label, error.

Usage:
    python val.py --input_dir /path/to/images --output_csv scores.csv

    # optional eval against ground truth:
    python val.py --input_dir . --output_csv scores.csv --labeled_dir /path/to/data
    python val.py --input_dir /path/to/images --output_csv scores.csv --labels_csv labels.csv

--labeled_dir expects real/spoof subfolders;
--labels_csv expects filename,label columns.
    Either writes PAD metrics (ROC-AUC, PR-AUC, EER, APCER/BPCER/ACER, etc.) and plots to --metrics_output_dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from joblib import load

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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

from features import extract_features

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_artifacts(model_dir):
    """Load the trained SVM model and its config from disk.
    Args:
        model_dir: folder containing 'model.joblib' and 'config.json'.
    """
    model_dir = Path(model_dir)
    model = load(model_dir / "model.joblib")
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    return model, config


def score_images(image_paths, model, root):
    """Extract FFT features and score each image with the trained model.
    Args:
        image_paths: list of image file paths to score.
        model: fitted sklearn pipeline (from load_artifacts()).
        root: base directory used to compute each image's rel_key.
    """
    results = {}
    for path in image_paths:
        key = rel_key(path, root)
        try:
            feats = extract_features(str(path)).reshape(1, -1)
            score = float(model.predict_proba(feats)[0, 1])
            results[key] = (score, None)
        except Exception as e:
            results[key] = (None, str(e))
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


def main(input_dir, output_csv, model_dir, labeled_dir, labels_csv, metrics_output_dir):
    """Score images with the trained FFT+SVM model, write results CSV, and (if ground truth is available) compute and
    save PAD metrics, plots, and a summary image.
    Args:
        input_dir: folder of images to score (used when labeled_dir is not given).
        output_csv: path to write per-image scores to.
        model_dir: folder with the trained model.joblib and config.json.
        labeled_dir: optional folder laid out with real/spoof subfolders.
        labels_csv: optional filename->label CSV, used with input_dir, for a flat folder (ignored if labeled_dir is given).
        metrics_output_dir: folder for metrics/plots.
    """
    model, config = load_artifacts(model_dir)
    threshold = config["threshold"]

    labels = None
    if labeled_dir:
        root = labeled_dir
        image_paths, labels = gather_labeled_dataset(labeled_dir)
    else:
        root = input_dir
        image_paths = list_images(input_dir)
        if labels_csv:
            labels = load_labels_csv(labels_csv)

    print(f"Found {len(image_paths)} images")
    results = score_images(image_paths, model, root)
    rows = write_scores_csv(image_paths, results, threshold, output_csv, root)
    n_errors = sum(1 for r in rows if r["error"])
    print(f"Wrote {len(rows)} rows to {output_csv} ({n_errors} errors)")

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
    ap.add_argument("--input_dir", required=True, help="Folder of images to score")
    ap.add_argument("--output_csv", required=True, help="Where to write the scores CSV")
    ap.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--labeled_dir", default=None,
                    help="Optional eval mode: folder with Bonafide/ and Screens/ subfolders "
                         "(overrides --input_dir for which images get scored)")
    ap.add_argument("--labels_csv", default=None,
                    help="Optional eval mode: CSV with columns filename,label for images in --input_dir")
    ap.add_argument("--metrics_output_dir", default=None,
                    help="Where to save the PAD metrics table + ROC/PR curves "
                         "(default: <output_csv's folder>/eval)")
    args = ap.parse_args()
    main(args.input_dir,
         args.output_csv,
         args.model_dir,
         args.labeled_dir,
         args.labels_csv,
         args.metrics_output_dir)

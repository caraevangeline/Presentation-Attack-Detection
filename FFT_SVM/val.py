"""Run the FFT + SVM screen-attack detector over a folder of images.

Plain inference:
    python val.py --input_dir /path/to/images --output_csv scores.csv \
        [--model_dir ./artifacts]

Writes one row per image: filename, attack_score (0-1, higher = more
likely a screen attack), predicted_label, error (empty if none).

Evaluation mode (optional, requires ground truth):
    python val.py --input_dir . --output_csv scores.csv \
        --labeled_dir /path/to/datasets/MLE_PlatformAI
    # or, for a flat folder with a separate labels file:
    python val.py --input_dir /path/to/images --output_csv scores.csv \
        --labels_csv labels.csv

If either --labeled_dir (a folder laid out like the training data, with
Bonafide/ and Screens/ subfolders) or --labels_csv (columns: filename,
label) is given, scores are additionally compared against ground truth
and the following are written to --metrics_output_dir
(default: <output_csv's folder>/eval):
    pad_metrics.csv / pad_metrics.json  -- ROC-AUC, PR-AUC, EER, and
        APCER/BPCER/ACER/precision/recall/F1 at both the model's
        configured threshold and the EER threshold
    metrics_summary.png  -- the same table + confusion matrices as an image
    roc_curve.png, pr_curve.png

The dataset-loading and metrics/plotting code is shared (not specific to
this model) and lives in ../common/pad_eval.py so other model folders can
reuse it without duplicating it.
"""

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
    model_dir = Path(model_dir)
    model = load(model_dir / "model.joblib")
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    return model, config


def score_images(image_paths, model, root):
    """rel_key -> (score or None, error or None)"""
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

    model, config = load_artifacts(args.model_dir)
    threshold = config["threshold"]

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
    results = score_images(image_paths, model, root)
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

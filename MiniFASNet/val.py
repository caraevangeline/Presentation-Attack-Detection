"""Scores a labeled eval folder with the exported ONNX MiniFASNet model
(see export_onnx.py) via onnxruntime and computes PAD metrics against
ground truth. No torch dependency needed to run this.

Usage:
    python val.py --input_dir /path/to/PAD-test-v1 --output_csv scores.csv \
        --model_dir ./artifacts/m2.v7
    python val.py --input_dir /path/to/images --output_csv scores.csv \
        --labels_csv labels.csv --model_dir ./artifacts/m2.v7

--input_dir: needs real/spoof subfolders (or use --labels_csv for a flat folder)
--model_dir: must contain model.onnx (from export_onnx.py) and config.json
--metrics_output_dir (default: <output_csv folder>/eval): saves metrics/plots
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

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


def _make_session(onnx_path):
    """Build an onnxruntime session, preferring CUDA if available. Falls back to CPU cleanly if the CUDA provider is
    listed as available but fails to actually load (needs a system-wide CUDA/cuDNN install matching onnxruntime's
    expected version, unlike torch which bundles its own CUDA runtime).
    Args:
        onnx_path: path to the exported model.onnx.
    """
    gpu_providers = [p for p in ["CUDAExecutionProvider"] if p in ort.get_available_providers()]
    if gpu_providers:
        try:
            return ort.InferenceSession(str(onnx_path), providers=gpu_providers + ["CPUExecutionProvider"])
        except Exception as e:
            print(f"WARNING: GPU execution provider failed to load ({e}); falling back to CPU.")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def load_artifacts(model_dir):
    """Load the ONNX session, its I/O tensor names, and config.json.
    Args:
        model_dir: folder containing model.onnx and config.json.
    """
    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    onnx_path = model_dir / "model.onnx"
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found -- run export_onnx.py --model_dir {model_dir} first")

    sess = _make_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name, config


def _softmax(x, axis=-1):
    """Numerically-stable softmax along `axis`."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def preprocess(img_bgr, input_size):
    """Convert a raw cv2-loaded image into the model's expected input tensor.
    Args:
        img_bgr: raw array from cv2.imread.
        input_size: (W, H) to resize to, from config.json.
    Returns:
        (C, H, W) float32 array in [0, 255], channel order preserved as-is
        (no BGR->RGB swap) -- see module docstring.
    """
    pil_img = Image.fromarray(img_bgr, mode="RGB")
    resized = pil_img.resize(tuple(input_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # H, W, C
    return arr.transpose(2, 0, 1)  # C, H, W


def score_images(image_paths, sess, input_name, output_name, input_size, root, batch_size=64):
    """Run the ONNX model over every image, batched.
    Args:
        image_paths: list of image file paths to score.
        sess: onnxruntime InferenceSession from load_artifacts().
        input_name: sess's input tensor name.
        output_name: sess's output tensor name.
        input_size: (W, H) to resize each image to.
        root: base directory used to compute each image's rel_key.
        batch_size: number of images per onnxruntime call.
    Returns:
        dict of rel_key -> (attack_score or None, error or None).
    """
    results = {}
    keys, tensors = [], []

    for path in image_paths:
        key = rel_key(path, root)
        img = cv2.imread(str(path))
        if img is None:
            results[key] = (None, "cv2.imread failed to decode image")
            continue
        try:
            tensor = preprocess(img, input_size)
        except Exception as e:
            results[key] = (None, str(e))
            continue
        keys.append(key)
        tensors.append(tensor)

    for start in range(0, len(tensors), batch_size):
        batch_keys = keys[start:start + batch_size]
        batch = np.stack(tensors[start:start + batch_size]).astype(np.float32)
        logits = sess.run([output_name], {input_name: batch})[0]
        probs = _softmax(logits, axis=1)[:, 1]
        for key, score in zip(batch_keys, probs):
            results[key] = (float(score), None)

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


def main(input_dir, output_csv, model_dir, labels_csv, metrics_output_dir, batch_size):
    """Score all images in input_dir against ground truth, write results CSV,
    and compute/save PAD metrics, plots, and a summary image.
    Args:
        input_dir: eval folder (real/spoof subfolders, or flat with labels_csv).
        output_csv: path to write per-image scores to.
        model_dir: folder with model.onnx (see export_onnx.py) + config.json.
        labels_csv: optional filename->label CSV for a flat input_dir.
        metrics_output_dir: folder for metrics/plots.
        batch_size: number of images per onnxruntime call.
    """
    sess, input_name, output_name, config = load_artifacts(model_dir)
    input_size = config["input_size"]
    threshold = config["threshold"]

    if labels_csv:
        root = input_dir
        image_paths = list_images(input_dir)
        labels = load_labels_csv(labels_csv)
    else:
        root = input_dir
        image_paths, labels = gather_labeled_dataset(input_dir)

    print(f"Found {len(image_paths)} images")
    results = score_images(image_paths, sess, input_name, output_name, input_size, root, batch_size)
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
    ap.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "artifacts"),
                     help="Folder with model.onnx (see export_onnx.py) + config.json")
    ap.add_argument("--labels_csv", default=None,
                     help="Optional: CSV with columns filename,label for a flat --input_dir, "
                          "instead of the default real/spoof subfolder layout")
    ap.add_argument("--metrics_output_dir", default=None,
                     help="Where to save the PAD metrics table + ROC/PR curves "
                          "(default: <output_csv's folder>/eval)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    main(args.input_dir,
         args.output_csv,
         args.model_dir,
         args.labels_csv,
         args.metrics_output_dir,
         args.batch_size)

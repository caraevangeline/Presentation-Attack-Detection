"""Same as val.py, but scores images with an exported ONNX model
(see export_onnx.py) via onnxruntime instead of loading the PyTorch
backbone directly. No torch dependency needed to run this.

Plain inference:
    python val_onnx.py --input_dir /path/to/images --output_csv scores.csv \
        --model_dir ./artifacts/m2.v7

Writes one row per image: filename, attack_score (0-1, higher = more
likely a screen attack), predicted_label, error (empty if none).

Evaluation mode (optional, requires ground truth):
    python val_onnx.py --input_dir . --output_csv scores.csv \
        --labeled_dir /path/to/datasets/train/PAD-v6 --model_dir ./artifacts/m2.v7
    # or, for a flat folder with a separate labels file:
    python val_onnx.py --input_dir /path/to/images --output_csv scores.csv \
        --labels_csv labels.csv --model_dir ./artifacts/m2.v7

If either --labeled_dir (a folder laid out like the training data, with
real/ and spoof/ subfolders) or --labels_csv (columns: filename, label) is
given, scores are additionally compared against ground truth and the
following are written to --metrics_output_dir (default: <output_csv's
folder>/eval): pad_metrics.csv / pad_metrics.json, metrics_summary.png,
roc_curve.png, pr_curve.png.

--model_dir must contain model.onnx (from export_onnx.py) and config.json.
Preprocessing matches training/val.py exactly: raw cv2 BGR array passed
through PIL WITHOUT a BGR->RGB swap (channel order preserved, see
val.py's docstring / functional.py's to_pil_image), resized with PIL
bilinear, fed as raw [0, 255] float values with NO /255 normalization.
"""

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
    """onnxruntime's CUDA provider needs a system-wide CUDA/cuDNN install
    matching its expected version (unlike torch, which bundles its own CUDA
    runtime) -- it can be "available" per get_available_providers() but
    still fail to actually load. Fall back to CPU cleanly instead of
    crashing if that happens."""
    gpu_providers = [p for p in ["CUDAExecutionProvider"] if p in ort.get_available_providers()]
    if gpu_providers:
        try:
            return ort.InferenceSession(str(onnx_path), providers=gpu_providers + ["CPUExecutionProvider"])
        except Exception as e:
            print(f"WARNING: GPU execution provider failed to load ({e}); falling back to CPU.")
    return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def load_artifacts(model_dir):
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
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def preprocess(img_bgr, input_size):
    """img_bgr: raw array from cv2.imread. Returns a (C, H, W) float32 array
    in [0, 255], channel order preserved as-is (no BGR->RGB swap) -- see
    module docstring."""
    pil_img = Image.fromarray(img_bgr, mode="RGB")
    resized = pil_img.resize(tuple(input_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # H, W, C
    return arr.transpose(2, 0, 1)  # C, H, W


def score_images(image_paths, sess, input_name, output_name, input_size, root, batch_size=64):
    """rel_key -> (score or None, error or None)"""
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
    ap.add_argument("--model_dir", default=str(Path(__file__).resolve().parent / "artifacts"),
                    help="Folder with model.onnx (see export_onnx.py) + config.json")
    ap.add_argument("--labeled_dir", default=None,
                    help="Optional eval mode: folder with real/ and spoof/ subfolders "
                         "(overrides --input_dir for which images get scored)")
    ap.add_argument("--labels_csv", default=None,
                    help="Optional eval mode: CSV with columns filename,label for images in --input_dir")
    ap.add_argument("--metrics_output_dir", default=None,
                    help="Where to save the PAD metrics table + ROC/PR curves "
                         "(default: <output_csv's folder>/eval)")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    sess, input_name, output_name, config = load_artifacts(args.model_dir)
    input_size = config["input_size"]
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
    results = score_images(image_paths, sess, input_name, output_name, input_size, root, args.batch_size)
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

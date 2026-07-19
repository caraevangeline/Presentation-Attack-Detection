"""Full PAD pipeline: YOLOv8n-face detection -> bbox expansion -> MiniFASNetV2 (ONNX) classification ->
annotated output image(s).

Usage:
    # single image
    python infer.py --input_image path/to/photo.jpg --output_image annotated.jpg --threshold 0.43
    # a folder of images -> annotated folder + CSV
    python infer.py --input_dir path/to/images --output_dir annotated/ [--output_csv results.csv]

Common options (defaults):
    [--face_model ../yolov8_face/yolov8n-face.onnx]
    [--pad_model ./artifacts/m2.v7/model.onnx]
    [--pad_config ./artifacts/m2.v7/config.json]
    [--crop_scale 1.5] [--min_face_conf 0.25] [--threshold <override>]
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
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
YOLOV8_FACE_DIR = REPO_ROOT / "MiniFASNet/yolov8_face"
sys.path.insert(0, str(YOLOV8_FACE_DIR))

from yolov8_face_detect_align import preprocess, postprocess, scale_bbox  # noqa: E402

FACE_MODEL_DEFAULT = YOLOV8_FACE_DIR / "yolov8n-face.onnx"
PAD_MODEL_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "model.onnx"
PAD_CONFIG_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "config.json"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CSV_FIELDS = ["filename", "face_index", "det_conf", "attack_score", "predicted_label",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "error"]


def _make_session(model_path):
    """Build an onnxruntime session, preferring CUDA if available. Falls back to CPU cleanly if the CUDA provider is
    listed as available but fails to actually load (needs a system-wide CUDA/cuDNN install matching onnxruntime's
    expected version, unlike torch which bundles its own CUDA runtime).
    Args:
        model_path: path to an .onnx model file.
    """
    gpu_providers = [p for p in ["CUDAExecutionProvider"] if p in ort.get_available_providers()]
    if gpu_providers:
        try:
            return ort.InferenceSession(str(model_path), providers=gpu_providers + ["CPUExecutionProvider"])
        except Exception as e:
            print(f"WARNING: GPU execution provider failed to load ({e}); falling back to CPU.")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def load_face_detector(model_path):
    """Load the YOLOv8n-face ONNX session and its I/O tensor names.
    Args:
        model_path: path to yolov8n-face.onnx.
    """
    sess = _make_session(model_path)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    return sess, input_names, output_names


def detect_faces(img, sess, input_names, output_names, img_size=(384, 640), min_conf=0.25):
    """Run YOLOv8n-face over a raw image.
    Args:
        img: raw BGR array from cv2.imread.
        sess: face detector session from load_face_detector().
        input_names, output_names: sess's tensor names.
        img_size: (h, w) the detector expects input resized to.
        min_conf: detections below this confidence are dropped.
    Returns:
        list of (box[x1,y1,x2,y2], conf) in original image coordinates.
    """
    im0s = [img]
    im = preprocess(im0s, img_size)
    outs = sess.run(output_names, {input_names[0]: np.array(im)})
    boxes, _kpts = postprocess(torch.Tensor(np.array(outs)), im, im0s)

    faces = []
    for output in boxes:
        conf = float(output[4])
        if conf < min_conf:
            continue
        box = [float(x) for x in output[:4]]
        faces.append((box, conf))
    return faces


def load_pad_model(onnx_path):
    """Load the exported PAD classifier ONNX session and its I/O tensor names.
    Args:
        onnx_path: path to model.onnx (see export_onnx.py).
    """
    sess = _make_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def _softmax(x):
    """Numerically-stable softmax over a 1-D array."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def score_crop(crop_bgr, sess, input_name, output_name, input_size):
    """Classify one face crop as real vs spoof.
    Args:
        crop_bgr: raw BGR numpy array (any size), as returned by cv2.
        sess: PAD model session from load_pad_model().
        input_name, output_name: sess's tensor names.
        input_size: (W, H) to resize the crop to.
    Returns:
        P(attack) as a float in [0, 1].
    """
    pil_img = Image.fromarray(crop_bgr, mode="RGB")
    resized = pil_img.resize(tuple(input_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # HWC, [0, 255]
    tensor = arr.transpose(2, 0, 1)[None, ...]  # 1, C, H, W

    logits = sess.run([output_name], {input_name: tensor})[0][0]
    probs = _softmax(logits)
    return float(probs[1])  # P(attack)


def annotate(img, results, threshold):
    """Draw a box + label per detected face onto a copy of img.
    Args:
        img: original BGR image.
        results: list of (box[x1,y1,x2,y2], det_conf, attack_score).
        threshold: attack_score cutoff for the SPOOF/REAL label and color.
    Returns:
        annotated copy of img (green box = real, red box = spoof).
    """
    out = img.copy()
    for box, conf, score in results:
        x1, y1, x2, y2 = [int(v) for v in box]
        is_attack = score >= threshold
        color = (0, 0, 255) if is_attack else (0, 200, 0)  # BGR: red / green
        label = f"{'SPOOF' if is_attack else 'REAL'} {score:.2f} (det {conf:.2f})"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, th + baseline + 4)
        cv2.rectangle(out, (x1, label_y - th - baseline - 4), (x1 + tw + 4, label_y), color, -1)
        cv2.putText(out, label, (x1 + 2, label_y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def process_image(img_path, face_sess, face_in, face_out, pad_sess, pad_in, pad_out,
                  input_size, threshold, crop_scale, min_face_conf, face_img_size):
    """Run the full detect -> crop -> classify -> annotate pipeline on one image.
    Args:
        img_path: path to the input photo.
        face_sess, face_in, face_out: face detector session + tensor names.
        pad_sess, pad_in, pad_out: PAD model session + tensor names.
        input_size: (W, H) the PAD model expects each crop resized to.
        threshold: attack_score cutoff for predicted_label.
        crop_scale: bbox expansion margin applied before classification.
        min_face_conf: minimum face-detection confidence to keep a detection.
        face_img_size: (h, w) the face detector expects input resized to.
    Returns:
        (annotated_image_or_None, csv_rows). annotated is None if the image
        couldn't be decoded, in which case csv_rows is a single error row.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, [{"filename": img_path.name, "face_index": "", "det_conf": "",
                       "attack_score": "", "predicted_label": "", "bbox_x1": "", "bbox_y1": "",
                       "bbox_x2": "", "bbox_y2": "", "error": "cv2.imread failed to decode image"}]

    img_h, img_w = img.shape[:2]
    faces = detect_faces(img, face_sess, face_in, face_out,
                         img_size=face_img_size, min_conf=min_face_conf)

    results, rows = [], []
    for i, (box, conf) in enumerate(faces):
        sx1, sy1, sx2, sy2 = scale_bbox(box, crop_scale, img_w, img_h)
        if sx2 <= sx1 or sy2 <= sy1:
            continue
        crop = img[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue

        score = score_crop(crop, pad_sess, pad_in, pad_out, input_size)
        label = "spoof" if score >= threshold else "bonafide"
        results.append(([sx1, sy1, sx2, sy2], conf, score))
        rows.append({
            "filename": img_path.name, "face_index": i, "det_conf": round(conf, 4),
            "attack_score": round(score, 6), "predicted_label": label,
            "bbox_x1": sx1, "bbox_y1": sy1, "bbox_x2": sx2, "bbox_y2": sy2, "error": "",
        })

    if not rows:
        rows = [{"filename": img_path.name, "face_index": "", "det_conf": "",
                 "attack_score": "", "predicted_label": "", "bbox_x1": "", "bbox_y1": "",
                 "bbox_x2": "", "bbox_y2": "", "error": "NO_FACE_DETECTED"}]

    annotated = annotate(img, results, threshold)
    return annotated, rows


def write_csv(rows, output_csv):
    """Write per-face result rows to CSV, creating parent folders as needed.
    Args:
        rows: list of dicts matching CSV_FIELDS.
        output_csv: path to write the CSV to.
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows):
    """Print a one-line summary per result row to stdout."""
    for r in rows:
        if r["error"]:
            print(f"  {r['filename']}: {r['error']}")
        else:
            print(f"  {r['filename']} face {r['face_index']} det_conf={r['det_conf']} "
                  f"bbox=({r['bbox_x1']},{r['bbox_y1']},{r['bbox_x2']},{r['bbox_y2']}) "
                  f"attack_score={r['attack_score']} -> {r['predicted_label']}")


def main(input_image, output_image, input_dir, output_dir, output_csv,
         face_model, pad_model, pad_config, crop_scale, min_face_conf,
         threshold, face_img_size):
    """Dispatch to single-image or batch mode and run the full PAD pipeline.
    Args:
        input_image, output_image: single-image mode paths (mutually
            exclusive with input_dir/output_dir).
        input_dir, output_dir: batch mode folders.
        output_csv: where to save the per-face results CSV (default:
            <output_dir>/results.csv in batch mode; optional in single mode).
        face_model: path to the YOLOv8n-face ONNX model.
        pad_model, pad_config: path to the exported PAD model.onnx + config.json.
        crop_scale: bbox expansion margin applied before classification.
        min_face_conf: minimum face-detection confidence to keep a detection.
        threshold: attack_score cutoff override (None uses pad_config's).
        face_img_size: (h, w) the face detector expects input resized to.
    """
    with open(pad_config) as f:
        pad_cfg = json.load(f)
    threshold = threshold if threshold is not None else pad_cfg["threshold"]
    input_size = pad_cfg["input_size"]

    face_sess, face_in, face_out = load_face_detector(face_model)
    pad_sess, pad_in, pad_out = load_pad_model(pad_model)

    if input_image:
        img_path = Path(input_image)
        annotated, rows = process_image(
            img_path, face_sess, face_in, face_out, pad_sess, pad_in, pad_out,
            input_size, threshold, crop_scale, min_face_conf, face_img_size)
        if annotated is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        print(f"Detected {sum(1 for r in rows if not r['error'])} face(s)")
        _print_rows(rows)

        out_path = Path(output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image to {out_path}")

        if output_csv:
            write_csv(rows, output_csv)
            print(f"Results written to {output_csv}")
        return

    # Batch mode
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Found {len(image_paths)} images")

    all_rows = []
    n_read_errors = 0
    for img_path in image_paths:
        annotated, rows = process_image(
            img_path, face_sess, face_in, face_out, pad_sess, pad_in, pad_out,
            input_size, threshold, crop_scale, min_face_conf, face_img_size)
        if annotated is None:
            n_read_errors += 1
        else:
            cv2.imwrite(str(output_dir / img_path.name), annotated)
        all_rows.extend(rows)

    output_csv = output_csv or str(output_dir / "results.csv")
    write_csv(all_rows, output_csv)

    n_no_face = sum(1 for r in all_rows if r["error"] == "NO_FACE_DETECTED")
    n_faces = sum(1 for r in all_rows if not r["error"])
    print(f"Processed {len(image_paths)} images: {n_faces} face(s) scored, "
          f"{n_no_face} image(s) with no face detected, {n_read_errors} read error(s)")
    print(f"Annotated images written to {output_dir}")
    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_image", default=None, help="Single-image mode: path to one photo")
    ap.add_argument("--output_image", default=None, help="Single-image mode: where to save the annotated photo")
    ap.add_argument("--input_dir", default=None, help="Batch mode: folder of photos")
    ap.add_argument("--output_dir", default=None, help="Batch mode: folder to save annotated photos into")
    ap.add_argument("--output_csv", default=None,
                    help="Batch mode: where to save the per-face results CSV "
                         "(default: <output_dir>/results.csv); also usable in single-image mode")
    ap.add_argument("--face_model", default=str(FACE_MODEL_DEFAULT))
    ap.add_argument("--pad_model", default=str(PAD_MODEL_DEFAULT))
    ap.add_argument("--pad_config", default=str(PAD_CONFIG_DEFAULT))
    ap.add_argument("--crop_scale", type=float, default=1.5)
    ap.add_argument("--min_face_conf", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the threshold stored in --pad_config")
    ap.add_argument("--face_img_size", type=int, nargs=2, default=[384, 640],
                    help="YOLOv8-face model input size (h w)")
    args = ap.parse_args()

    if bool(args.input_image) == bool(args.input_dir):
        ap.error("specify exactly one of --input_image (with --output_image) "
                 "or --input_dir (with --output_dir)")
    if args.input_image and not args.output_image:
        ap.error("--output_image is required with --input_image")
    if args.input_dir and not args.output_dir:
        ap.error("--output_dir is required with --input_dir")

    main(args.input_image, args.output_image, args.input_dir, args.output_dir, args.output_csv,
         args.face_model, args.pad_model, args.pad_config, args.crop_scale, args.min_face_conf,
         args.threshold, tuple(args.face_img_size))

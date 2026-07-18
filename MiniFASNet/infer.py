"""Full PAD pipeline: YOLOv8n-face detection -> bbox expansion ->
MiniFASNetV2SE (ONNX) classification -> annotated output image(s).

Unlike val.py (which scores pre-cropped face patches), this takes raw,
uncropped photos, detects the face(s) with ../yolov8_face/yolov8n-face.onnx,
expands each detected bbox by --crop_scale, and classifies each resulting
crop with the exported ONNX PAD model (see export_onnx.py).

--crop_scale defaults to 1.5, NOT the 2.7x the original (pre-fine-tuning)
MiniFASNetV2 checkpoint was trained on. PAD-v6/v7 (what finetune.py actually
fine-tuned on) are cropped at a 1.5x margin (see ../README.md's "Known
limitations"), so 1.5x is what the deployed model's weights now expect.
Using 2.7x here is a measured, not theoretical, mismatch: on a plain test
image, --crop_scale 2.7 scored a bona fide face at 0.988 (spoof), while
--crop_scale 1.5 scored the same face+detection at 0.048 (correctly
bona fide). Match whatever crop convention the model you're loading was
actually fine-tuned on, not the original upstream checkpoint's.

No landmark-based alignment is applied to the crop (deliberately -- see
../README.md's "Known limitations": rotation/interpolation can smear the
high-frequency moire signal this method partly relies on), matching how
the training crops themselves were produced.

Single-image mode:
    python infer.py --input_image path/to/photo.jpg --output_image annotated.jpg

Batch mode (a folder of images -> annotated folder + CSV):
    python infer.py --input_dir path/to/images --output_dir annotated/ \
        [--output_csv results.csv]

Common options (either mode):
        [--face_model ../yolov8_face/yolov8n-face.onnx] \
        [--pad_model ./artifacts/m2.v7/model.onnx] \
        [--pad_config ./artifacts/m2.v7/config.json] \
        [--crop_scale 1.5] [--min_face_conf 0.25] [--threshold <override>]

In batch mode, results.csv has one row per detected face (filename,
face_index, det_conf, attack_score, predicted_label, bbox_x1/y1/x2/y2), or
one row with error set if no face was found / the image couldn't be read.
"""

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
YOLOV8_FACE_DIR = REPO_ROOT / "yolov8_face"
sys.path.insert(0, str(YOLOV8_FACE_DIR))

from yolov8_face_detect_align import preprocess, postprocess, scale_bbox  # noqa: E402

FACE_MODEL_DEFAULT = YOLOV8_FACE_DIR / "yolov8n-face.onnx"
PAD_MODEL_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "model.onnx"
PAD_CONFIG_DEFAULT = Path(__file__).resolve().parent / "artifacts" / "config.json"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CSV_FIELDS = ["filename", "face_index", "det_conf", "attack_score", "predicted_label",
              "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "error"]


def _make_session(model_path):
    """onnxruntime's CUDA provider needs a system-wide CUDA/cuDNN install
    matching its expected version (unlike torch, which bundles its own CUDA
    runtime) -- it can be "available" per get_available_providers() but
    still fail to actually load. Fall back to CPU cleanly instead of
    crashing if that happens."""
    gpu_providers = [p for p in ["CUDAExecutionProvider"] if p in ort.get_available_providers()]
    if gpu_providers:
        try:
            return ort.InferenceSession(str(model_path), providers=gpu_providers + ["CPUExecutionProvider"])
        except Exception as e:
            print(f"WARNING: GPU execution provider failed to load ({e}); falling back to CPU.")
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def load_face_detector(model_path):
    sess = _make_session(model_path)
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    return sess, input_names, output_names


def detect_faces(img, sess, input_names, output_names, img_size=(384, 640), min_conf=0.25):
    """Returns a list of (box[x1,y1,x2,y2], conf) in original image coords."""
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
    sess = _make_session(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    return sess, input_name, output_name


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def score_crop(crop_bgr, sess, input_name, output_name, input_size):
    """crop_bgr: raw BGR numpy array (any size), as returned by cv2.

    Matches finetune.py/val.py's preprocessing exactly: the array is passed
    through PIL WITHOUT a BGR->RGB swap (torchvision's ToPILImage on a
    3-channel uint8 array just labels it 'RGB' without reordering channels,
    so channel order is preserved as given -- see functional.py's
    to_pil_image), resized with PIL bilinear, and fed as raw [0, 255] float
    values with NO /255 normalization (see functional.py's to_tensor, which
    explicitly does not divide by 255 in this repo's fork). Getting either
    of those wrong silently mismatches training and would degrade scores.
    """
    pil_img = Image.fromarray(crop_bgr, mode="RGB")
    resized = pil_img.resize(tuple(input_size), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32)  # HWC, [0, 255]
    tensor = arr.transpose(2, 0, 1)[None, ...]  # 1, C, H, W

    logits = sess.run([output_name], {input_name: tensor})[0][0]
    probs = _softmax(logits)
    return float(probs[1])  # P(attack)


def annotate(img, results, threshold):
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
    """Returns (annotated_image_or_None, csv_rows). annotated is None if the
    image couldn't be decoded, in which case csv_rows is a single error row."""
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
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows):
    for r in rows:
        if r["error"]:
            print(f"  {r['filename']}: {r['error']}")
        else:
            print(f"  {r['filename']} face {r['face_index']} det_conf={r['det_conf']} "
                  f"bbox=({r['bbox_x1']},{r['bbox_y1']},{r['bbox_x2']},{r['bbox_y2']}) "
                  f"attack_score={r['attack_score']} -> {r['predicted_label']}")


def main():
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

    with open(args.pad_config) as f:
        pad_config = json.load(f)
    threshold = args.threshold if args.threshold is not None else pad_config["threshold"]
    input_size = pad_config["input_size"]

    face_sess, face_in, face_out = load_face_detector(args.face_model)
    pad_sess, pad_in, pad_out = load_pad_model(args.pad_model)

    common_kwargs = dict(
        face_sess=face_sess, face_in=face_in, face_out=face_out,
        pad_sess=pad_sess, pad_in=pad_in, pad_out=pad_out,
        input_size=input_size, threshold=threshold, crop_scale=args.crop_scale,
        min_face_conf=args.min_face_conf, face_img_size=tuple(args.face_img_size),
    )

    if args.input_image:
        if not args.output_image:
            ap.error("--output_image is required with --input_image")
        img_path = Path(args.input_image)
        annotated, rows = process_image(
            img_path, face_sess, face_in, face_out, pad_sess, pad_in, pad_out,
            input_size, threshold, args.crop_scale, args.min_face_conf, tuple(args.face_img_size))
        if annotated is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        print(f"Detected {sum(1 for r in rows if not r['error'])} face(s)")
        _print_rows(rows)

        out_path = Path(args.output_image)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved annotated image to {out_path}")

        if args.output_csv:
            write_csv(rows, args.output_csv)
            print(f"Results written to {args.output_csv}")
        return

    # Batch mode
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"Found {len(image_paths)} images")

    all_rows = []
    n_read_errors = 0
    for img_path in image_paths:
        annotated, rows = process_image(
            img_path, face_sess, face_in, face_out, pad_sess, pad_in, pad_out,
            input_size, threshold, args.crop_scale, args.min_face_conf, tuple(args.face_img_size))
        if annotated is None:
            n_read_errors += 1
        else:
            cv2.imwrite(str(output_dir / img_path.name), annotated)
        all_rows.extend(rows)

    output_csv = args.output_csv or str(output_dir / "results.csv")
    write_csv(all_rows, output_csv)

    n_no_face = sum(1 for r in all_rows if r["error"] == "NO_FACE_DETECTED")
    n_faces = sum(1 for r in all_rows if not r["error"])
    print(f"Processed {len(image_paths)} images: {n_faces} face(s) scored, "
          f"{n_no_face} image(s) with no face detected, {n_read_errors} read error(s)")
    print(f"Annotated images written to {output_dir}")
    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    main()

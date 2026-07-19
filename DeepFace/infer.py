"""Run DeepFace's anti-spoofing model over a folder of images, saving annotated outputs
(green=REAL, red=SPOOF, with confidence) and a results CSV. Same annotation style as MiniFASNet/infer.py,
but uses DeepFace's off-the-shelf detector + model for comparison on the same images.

Usage:
    python infer.py --input_dir path/to/images  \
                    --output_dir annotated/ \
                    [--output_csv results.csv] \
                    [--known_label real|spoof]

--known_label is optional: if given, every image in --input_dir is treated as having that ground-truth label,
and the CSV flags disagreements with DeepFace's prediction
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from deepface import DeepFace

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def annotate(img, faces):
    """Draw bounding boxes and REAL/SPOOF labels on an image.
    Args:
        img: input image, BGR (as read by cv2).
        faces: list of DeepFace face dicts.
    """
    out = img.copy()
    for face in faces:
        area = face["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        is_real = face["is_real"]
        score = face["antispoof_score"]
        color = (0, 200, 0) if is_real else (0, 0, 255)  # BGR: green / red
        label = f"{'REAL' if is_real else 'SPOOF'} {score:.2f} (det {face['confidence']:.2f})"

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y, th + baseline + 4)
        cv2.rectangle(out, (x, label_y - th - baseline - 4), (x + tw + 4, label_y), color, -1)
        cv2.putText(out, label, (x + 2, label_y - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def process_folder(input_dir, output_dir, known_label=None):
    """Run DeepFace anti-spoofing over every image in input_dir, saving annotated images and
    returning per-image results.
    Args:
       input_dir: folder of images to process.
       output_dir: folder to save annotated output images into.
       known_label: optional ground-truth label ("real"/"spoof") shared by every image in input_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    results = []

    for path in image_paths:
        row = {"filename": path.name}
        if known_label is not None:
            row["known_label"] = known_label

        try:
            faces = DeepFace.extract_faces(
                img_path=str(path), anti_spoofing=True, enforce_detection=False)
        except Exception as e:
            row.update({"model_predicted_label": None, "model_confidence": None,
                        "flag": f"ERROR: {e}"})
            results.append(row)
            continue

        if len(faces) == 0:
            row.update({"model_predicted_label": None, "model_confidence": None,
                        "flag": "NO_FACE_DETECTED"})
            results.append(row)
            continue

        img = cv2.imread(str(path))
        annotated = annotate(img, faces)
        cv2.imwrite(str(output_dir / path.name), annotated)

        # Summarize using the first/largest detected face for the CSV row.
        face_obj = faces[0]
        model_label = "real" if face_obj["is_real"] else "spoof"
        row["model_predicted_label"] = model_label
        row["model_confidence"] = round(face_obj["antispoof_score"], 4)
        row["flag"] = ("DISAGREEMENT" if known_label is not None and model_label != known_label
                       else "OK")
        results.append(row)

    return results


def write_csv(results, output_csv, known_label=None):
    """Write per-image results to a CSV file.
    Args:
        results: list of result dicts, as returned by process_folder().
        output_csv: path to write the CSV to.
        known_label: pass the same value used in process_folder(), so the "known_label" column is included only
            when it was actually set.
    """
    fieldnames = ["filename"]
    if known_label is not None:
        fieldnames.append("known_label")
    fieldnames += ["model_predicted_label", "model_confidence", "flag"]

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main(input_dir, output_dir, known_label, output_csv):
    """Run detection over a folder, write results CSV, and print a summary.
    Args:
        input_dir: folder of images to process.
        output_dir: folder to save annotated output images into.
        known_label: optional ground-truth label ("real"/"spoof") for all images in input_dir.
        output_csv: path for the results CSV; defaults to "<output_dir>/results.csv" if not given.
    """
    results = process_folder(input_dir, output_dir, known_label)

    output_csv = output_csv or str(Path(output_dir) / "results.csv")
    write_csv(results, output_csv, known_label)

    n_total = len(results)
    n_no_face = sum(1 for r in results if r["flag"] == "NO_FACE_DETECTED")
    n_error = sum(1 for r in results if "ERROR" in str(r["flag"]))
    n_disagree = sum(1 for r in results if r["flag"] == "DISAGREEMENT")

    print(f"\n--- Summary for {input_dir} ---")
    print(f"Total images: {n_total}")
    if known_label is not None and n_total:
        print(f"Disagreements (possible mislabels): {n_disagree} ({100 * n_disagree / n_total:.1f}%)")
    print(f"No face detected: {n_no_face}")
    print(f"Errors: {n_error}")
    print(f"Annotated images written to {output_dir}")
    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", required=True, help="Folder of images to run detection on")
    ap.add_argument("--output_dir", required=True, help="Where to save annotated images")
    ap.add_argument("--output_csv", default=None,
                    help="Where to save the per-image results CSV (default: <output_dir>/results.csv)")
    ap.add_argument("--known_label", default=None, choices=["real", "spoof"],
                    help="Optional: if every image in --input_dir shares this ground-truth "
                         "label, flag disagreements against DeepFace's prediction")
    args = ap.parse_args()
    main(args.input_dir,
         args.output_dir,
         args.known_label,
         args.output_csv)

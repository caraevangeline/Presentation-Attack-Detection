"""
ONNX inference code for YOLOv8 Face model, with face-crop export at a
2.7x bounding-box scale margin (matching the MiniFASNet / deepface
anti-spoofing convention).

Usage:
    python yolov8_face_detect_crop.py --onnx_model_path <path/to/model>
                                       --input_path <path/to/image_folder>
                                       --output_path <path/to/crops_folder>
                                       --img_size <input_height> <input_width>
                                       --crop_scale 2.7

Example:
    python yolov8_face_detect_crop.py --onnx_model_path yolov8n-face.onnx
                                       --input_path images
                                       --output_path crops
                                       --img_size 384 640
                                       --crop_scale 2.7
"""

import argparse
import glob
import os
from pathlib import Path

import onnx
import onnxruntime as ort
import cv2
import numpy as np
import torch

from ops import non_max_suppression, scale_boxes, scale_coords, convert_torch2numpy_batch


def plot(pred, kpts, orig_img, shape=(384, 640)):
    """Draw bounding boxes and keypoints to the original input image.

    Args:
        pred: Bounding boxes
        kpts: Keypoints
        orig_img: Input image
        shape: Input size
    """
    # Draw bounding boxes with class name and confidence
    for i, output in enumerate(pred):
        conf = float(output[4])
        box = [int(x) for x in output[:4]]
        cv2.rectangle(orig_img, (box[0], box[1]), (box[2], box[3]), (123, 123, 212), 2)
        label_text = f'Face: {conf:.2f}'
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        cv2.rectangle(
            orig_img,
            (box[0], box[1] - text_height - 6),
            (box[0] + text_width, box[1]),
            (123, 123, 212),
            -1
        )
        cv2.putText(orig_img, label_text, (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Draw keypoints
    for j in range(kpts.shape[0]):
        for i, k in enumerate(kpts[j]):
            color_k = (255, 0, 0)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                cv2.circle(orig_img, (int(x_coord), int(y_coord)), 2, color_k, -1, lineType=cv2.LINE_AA)
    return orig_img


def scale_bbox(box, scale, img_w, img_h):
    """Scale a bounding box by `scale`, keeping it centered on the same
    center point, then clamp to image bounds.

    This follows the MiniFASNet / deepface anti-spoofing convention of
    cropping with a generous margin (e.g. 2.7x) around the detected face,
    since spoof-relevant context (screen bezels, moire, background) often
    sits outside a tight face-only crop.

    Args:
        box: [x1, y1, x2, y2] in original image pixel coordinates
        scale: scale factor to expand the box by (e.g. 2.7)
        img_w: original image width
        img_h: original image height

    Returns:
        [x1, y1, x2, y2] scaled and clamped to image bounds (ints)
    """
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    new_w = bw * scale
    new_h = bh * scale

    new_x1 = cx - new_w / 2.0
    new_y1 = cy - new_h / 2.0
    new_x2 = cx + new_w / 2.0
    new_y2 = cy + new_h / 2.0

    # Clamp to image bounds
    new_x1 = max(0, int(round(new_x1)))
    new_y1 = max(0, int(round(new_y1)))
    new_x2 = min(img_w, int(round(new_x2)))
    new_y2 = min(img_h, int(round(new_y2)))

    return [new_x1, new_y1, new_x2, new_y2]


def align_face(orig_img, kpts, min_kpt_conf=0.5):
    """Rotate the full image so the eyes are horizontal, using the
    YOLOv8-face 5-point landmarks (order: left_eye, right_eye, nose,
    left_mouth, right_mouth).

    Rotating the FULL image (not just the crop) and returning the
    rotation matrix lets us transform the bounding box consistently,
    so the crop margin stays correct after rotation.

    Args:
        orig_img: original full-resolution image (BGR, numpy array)
        kpts: (5, 3) array of [x, y, conf] landmarks for one face
        min_kpt_conf: skip alignment (return None) if eye keypoints are
                      low-confidence/not visible

    Returns:
        (rotated_img, rot_mat) or (None, None) if eyes aren't usable
    """
    left_eye = kpts[0]
    right_eye = kpts[1]

    if len(left_eye) == 3 and (left_eye[2] < min_kpt_conf or right_eye[2] < min_kpt_conf):
        return None, None

    lx, ly = float(left_eye[0]), float(left_eye[1])
    rx, ry = float(right_eye[0]), float(right_eye[1])

    dx, dy = rx - lx, ry - ly
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    h, w = orig_img.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv2.warpAffine(
        orig_img, rot_mat, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114)
    )
    return rotated, rot_mat


def transform_box(box, rot_mat):
    """Transform a [x1,y1,x2,y2] box through a 2x3 rotation matrix by
    rotating all 4 corners and taking the new axis-aligned bounding rect.
    """
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    corners_hom = np.hstack([corners, ones])          # (4, 3)
    transformed = (rot_mat @ corners_hom.T).T          # (4, 2)

    new_x1, new_y1 = transformed.min(axis=0)
    new_x2, new_y2 = transformed.max(axis=0)
    return [new_x1, new_y1, new_x2, new_y2]


def crop_and_save_faces(pred, kpts, orig_img, output_dir, stem, crop_scale=2.7,
                         min_conf=0.25, align=False, min_kpt_conf=0.5):
    """Crop each detected face bounding box (scaled by crop_scale, centered
    on the original box), optionally aligning by eye landmarks first, and
    save each crop to output_dir.

    Args:
        pred: detection results, each row [x1, y1, x2, y2, conf, cls]
        kpts: (num_faces, 5, 3) landmark array aligned by index with pred
        orig_img: original full-resolution image (BGR, numpy array)
        output_dir: folder to save crops into
        stem: base filename (without extension) used to name outputs
        crop_scale: scale factor applied to each bbox before cropping
        min_conf: skip detections below this confidence
        align: if True, rotate face to level eyes before cropping.
               Recommended for CNN-based classifiers (EfficientNet/MiniFASNet).
               NOT recommended for FFT/moire-based methods, since rotation +
               interpolation can smear the high-frequency moire signal.
        min_kpt_conf: minimum eye-keypoint confidence required to align;
                      falls back to unaligned crop if landmarks are unreliable

    Returns:
        number of crops saved
    """
    os.makedirs(output_dir, exist_ok=True)
    img_h, img_w = orig_img.shape[:2]
    saved = 0

    for i, output in enumerate(pred):
        conf = float(output[4])
        if conf < min_conf:
            continue

        box = [float(x) for x in output[:4]]
        working_img = orig_img
        working_box = box

        if align:
            face_kpts = kpts[i].cpu().numpy() if hasattr(kpts[i], "cpu") else np.array(kpts[i])
            rotated, rot_mat = align_face(orig_img, face_kpts, min_kpt_conf=min_kpt_conf)
            if rotated is not None:
                working_img = rotated
                working_box = transform_box(box, rot_mat)
            # else: eye keypoints unreliable, fall back to unaligned crop

        sx1, sy1, sx2, sy2 = scale_bbox(working_box, crop_scale, img_w, img_h)

        if sx2 <= sx1 or sy2 <= sy1:
            # degenerate box after clamping, skip
            continue

        crop = working_img[sy1:sy2, sx1:sx2]
        if crop.size == 0:
            continue

        suffix = "_aligned" if align else ""
        out_name = f"{stem}_face{i}_conf{conf:.2f}{suffix}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, crop)
        saved += 1

    return saved


def postprocess(preds, img, orig_imgs):
    """Return detection results for a given input image or list of images.
    Args:
        preds: Model output
        img: Pre-processed image
        orig_imgs: Input image
    """
    preds = non_max_suppression(
        preds[0],
        0.25,
        0.45,
        agnostic=False,
        max_det=300,
        classes=None,
        nc=1
    )

    if not isinstance(orig_imgs, list):
        orig_imgs = convert_torch2numpy_batch(orig_imgs)

    orig_img = orig_imgs[0]
    pred = preds[0]
    pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape).round()
    pred_kpts = pred[:, 6:].view(len(pred), 5, 3) if len(pred) else pred[:, 6:]
    pred_kpts = scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
    return pred[:, :6], pred_kpts


def letterbox(image=None, img_size=640, center=True):
    """Return image with added border.

    Args:
        image: Input image
        img_size: Input image size
        center: Place image in the center
    """
    img = image
    shape = img.shape[:2]
    new_shape = img_size
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if center:
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)) if center else 0, int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)) if center else 0, int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img


def preprocess(im, img_size):
    """Prepare input image before inference.

    Args:
        im (torch.Tensor | List[np.ndarray]): BCHW tensor or list of HWC images
        img_size: Input size of the model
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack([letterbox(image=x, img_size=img_size) for x in im])
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)

    im = im.to('cpu')
    im = im.float()
    if not_tensor:
        im /= 255
    print(im.shape)
    return im


def main(onnx_file_path, input_path, output_path, img_size, crop_scale,
         save_annotated=False, align=False, min_kpt_conf=0.5):
    """Detect faces using YOLOv8-face, then crop each detection at
    `crop_scale`x its bounding box and save to output_path.

    Args:
        onnx_file_path: Face detection model path
        input_path: Folder of input images
        output_path: Folder to save cropped faces into
        img_size: Model input size [h, w]
        crop_scale: Scale factor for bbox crop margin (e.g. 2.7)
        save_annotated: If True, also save the original image with boxes drawn
        align: If True, rotate faces to level eyes before cropping (use for
               CNN-based classifiers; skip for FFT/moire-based methods)
        min_kpt_conf: minimum eye-keypoint confidence required to align
    """
    model = onnx.load(onnx_file_path)
    onnx.checker.check_model(model)

    devices = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider']
               if p in ort.get_available_providers()]
    model_session = ort.InferenceSession(onnx_file_path, providers=devices)
    outname = [i.name for i in model_session.get_outputs()]
    inname = [i.name for i in model_session.get_inputs()]

    os.makedirs(output_path, exist_ok=True)

    total_images = 0
    total_faces = 0
    no_face_files = []

    for files in glob.glob(f'{input_path}/*'):
        img = cv2.imread(str(files))
        if img is None:
            continue
        total_images += 1

        im0s = [img]
        im = preprocess(im0s, img_size)
        outs = model_session.run(outname, {inname[0]: np.array(im)})

        box, kp = postprocess(torch.Tensor(np.array(outs)), im, im0s)
        stem = Path(files).stem

        if box.size()[0]:
            n_saved = crop_and_save_faces(
                box, kp, im0s[0], output_path, stem, crop_scale=crop_scale,
                align=align, min_kpt_conf=min_kpt_conf
            )
            total_faces += n_saved

            if save_annotated:
                plotted_img = plot(box, kp, im0s[0].copy(), shape=img_size)
                annotated_dir = os.path.join(output_path, "annotated")
                os.makedirs(annotated_dir, exist_ok=True)
                cv2.imwrite(os.path.join(annotated_dir, f"{stem}_annotated.jpg"), plotted_img)
        else:
            no_face_files.append(str(files))

    print(f"\nProcessed {total_images} images, saved {total_faces} face crops "
          f"(scale={crop_scale}x) to '{output_path}'")
    if no_face_files:
        print(f"{len(no_face_files)} images had no detected face:")
        for f in no_face_files[:10]:
            print(" ", f)
        if len(no_face_files) > 10:
            print(f"  ... and {len(no_face_files) - 10} more")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_path', type=str,
                        default='yolov8n-face.onnx',
                        help='Path to ONNX file')
    parser.add_argument('--input_path', type=str,
                        default='images',
                        help='Path to input image folder')
    parser.add_argument('--output_path', type=str,
                        default='crops',
                        help='Path to output folder for cropped faces')
    parser.add_argument('--img_size', type=int, nargs=2,
                        default=[384, 640],
                        help='Model input size (h w)')
    parser.add_argument('--crop_scale', type=float,
                        default=2.7,
                        help='Bounding box scale margin for crops (deepface/MiniFASNet convention)')
    parser.add_argument('--save_annotated', action='store_true',
                        help='Also save original images with boxes/keypoints drawn')
    parser.add_argument('--align', action='store_true',
                        help='Rotate faces to level eyes before cropping. '
                             'Recommended for CNN classifiers (EfficientNet/MiniFASNet). '
                             'Do NOT use for FFT/moire-based methods, since rotation + '
                             'interpolation can smear the high-frequency moire signal.')
    parser.add_argument('--min_kpt_conf', type=float, default=0.5,
                        help='Minimum eye-keypoint confidence required to align; '
                             'falls back to unaligned crop if below this')
    args = parser.parse_args()

    print('Running inference...', end=' ')
    main(
        onnx_file_path=args.onnx_model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        img_size=args.img_size,
        crop_scale=args.crop_scale,
        save_annotated=args.save_annotated,
        align=args.align,
        min_kpt_conf=args.min_kpt_conf
    )
# MiniFASNet PAD inference image: run infer.py's batch mode over a mounted
# folder of images and get back annotated images + a results CSV.
#
# Build (context must be the Presentation-Attack-Detection/ folder, since
# this needs to COPY the sibling yolov8_face/ directory too -- run from
# Presentation-Attack-Detection/):
#     docker build -f MiniFASNet/Dockerfile -t minifasnet-pad .
#
# Or from the repo root:
#     docker build -f Presentation-Attack-Detection/MiniFASNet/Dockerfile \
#         -t minifasnet-pad Presentation-Attack-Detection/
#
# Run (mount a folder of images in, get annotated images + results.csv out):
#     docker run --rm \
#         -v /path/to/images:/data/input:ro \
#         -v /path/to/output:/data/output \
#         minifasnet-pad
#
# Any infer.py flag can be appended to override the default CMD, e.g. a
# different decision threshold:
#     docker run --rm -v ...:/data/input:ro -v ...:/data/output minifasnet-pad \
#         --input_dir /data/input --output_dir /data/output --threshold 0.3
#
# The checkpoint baked into this image is artifacts/m2.v7 (see the COPY
# lines below) -- rebuild with a different --build-arg CHECKPOINT=... to
# ship a different one.

FROM python:3.8-slim

# libgl1/libglib2.0-0: opencv-python-headless still dynamically links against
# these on Debian even though it doesn't need a display.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY MiniFASNet/requirements-infer.txt Presentation-Attack-Detection/MiniFASNet/requirements-infer.txt
RUN pip install --no-cache-dir -r Presentation-Attack-Detection/MiniFASNet/requirements-infer.txt

ARG CHECKPOINT=m2.v7

# App code: MiniFASNet/ and yolov8_face/ are kept as siblings under
# Presentation-Attack-Detection/, matching the real repo layout, since
# infer.py locates yolov8_face/ via a relative "../yolov8_face" from its own
# path (see REPO_ROOT / YOLOV8_FACE_DIR in infer.py).
COPY MiniFASNet/infer.py Presentation-Attack-Detection/MiniFASNet/infer.py
COPY yolov8_face/ops.py Presentation-Attack-Detection/yolov8_face/ops.py
COPY yolov8_face/yolov8_face_detect_align.py Presentation-Attack-Detection/yolov8_face/yolov8_face_detect_align.py
COPY yolov8_face/yolov8n-face.onnx Presentation-Attack-Detection/yolov8_face/yolov8n-face.onnx

# Baked-in checkpoint, copied to infer.py's own default path
# (artifacts/model.onnx + artifacts/config.json) so no --pad_model/
# --pad_config override is needed at runtime.
COPY MiniFASNet/artifacts/${CHECKPOINT}/model.onnx Presentation-Attack-Detection/MiniFASNet/artifacts/model.onnx
COPY MiniFASNet/artifacts/${CHECKPOINT}/config.json Presentation-Attack-Detection/MiniFASNet/artifacts/config.json

WORKDIR /app/Presentation-Attack-Detection/MiniFASNet

RUN mkdir -p /data/input /data/output
VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["python", "infer.py"]
CMD ["--input_dir", "/data/input", "--output_dir", "/data/output"]

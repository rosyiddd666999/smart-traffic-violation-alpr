# syntax=docker/dockerfile:1.7
#
# Image CPU (fallback & untuk pengujian di mesin tanpa GPU).
# Untuk GPU NVIDIA gunakan Dockerfile.gpu lewat docker-compose.gpu.yml.
#
# Build:
#   docker build -t traffic-alpr:latest .
#
# Catatan desain:
# - torch CPU dipasang lebih dulu supaya ultralytics tidak menarik wheel CUDA
#   (~2,5 GB) dari PyPI.
# - Model YOLO + fast-alpr diunduh saat build (docker/warmup_models.py) agar
#   container produksi tidak butuh internet.
# - Aplikasi berjalan sebagai user non-root UID 10001.

FROM python:3.12-slim AS runtime

ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
ARG ONNXRUNTIME_VERSION=1.31.1
ARG INSTALL_EASYOCR=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Library sistem untuk OpenCV, PyTorch, dan ffmpeg bawaan imageio-ffmpeg.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Virtualenv terpisah agar site-packages bersih dan mudah di-audit.
RUN python -m venv "$VIRTUAL_ENV" \
    && pip install --upgrade pip setuptools wheel

# 1) Torch CPU (index resmi PyTorch, bukan PyPI yang membawa CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"

# 2) Dependency aplikasi (tanpa torch/onnxruntime).
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# 3) Runtime ONNX CPU. Dipasang terpisah agar tidak bertabrakan dengan
#    onnxruntime-gpu di Dockerfile.gpu.
RUN pip install "onnxruntime==${ONNXRUNTIME_VERSION}"

# 4) Opsional: fallback EasyOCR (tidak dipakai secara default).
RUN if [ "$INSTALL_EASYOCR" = "1" ]; then \
        pip install "easyocr==1.7.2"; \
    fi

# User non-root + direktori data persisten.
RUN useradd --uid 10001 --create-home --home-dir /home/appuser \
        --shell /usr/sbin/nologin appuser \
    && mkdir -p /app /data \
    && chown appuser:appuser /data

WORKDIR /app
COPY --chown=appuser:appuser . .

ENV HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache \
    YOLO_CONFIG_DIR=/home/appuser/.config/Ultralytics \
    MPLCONFIGDIR=/home/appuser/.cache/matplotlib \
    STORAGE_DIR=/data \
    DATABASE_URL=sqlite:////data/violations.db \
    DEVICE=cpu \
    OCR_GPU=false

USER appuser

# Unduh bobot YOLO + model ONNX ke cache di dalam image (butuh internet sekali).
RUN python docker/warmup_models.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)"

# WAJIB satu worker: worker pipeline in-process + SQLite.
# Jangan pernah memakai --workers > 1 atau --scale app=2.
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]

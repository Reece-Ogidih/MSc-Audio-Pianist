# Hex/Hare runtime image for the ALA Pianist DroQ continuation.
#
# This image intentionally contains dependencies only. Mount the repository at
# /app and scratch/experiment storage at /workspace when running with Hare.
#
# Local checkpoint environment used torch 2.13.0. Hex driver 570.124.06 reports
# CUDA 12.8 support, so use a CUDA 12.6 runtime rather than a CUDA 13 build.
FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/ala-pianist-venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PIP_NO_CACHE_DIR=1 \
    ALA_PIANIST_ROOT=/app \
    ALA_PIANIST_DROQ_OUTPUT_DIR=/workspace/general_one_hand/droq

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    fluidsynth \
    git \
    libasound2-dev \
    libegl1 \
    libgl1 \
    libglfw3 \
    libgles2 \
    libosmesa6 \
    libportaudio2 \
    libsndfile1 \
    portaudio19-dev \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /opt/ala-pianist-runtime
COPY requirements.txt /opt/ala-pianist-runtime/requirements.txt
COPY torch-constraints.txt /opt/ala-pianist-runtime/torch-constraints.txt
RUN python -m pip install \
      --constraint /opt/ala-pianist-runtime/torch-constraints.txt \
      -r /opt/ala-pianist-runtime/requirements.txt \
    && python - <<'PY'
import platform
import torch

print("python", platform.python_version())
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("torch_file", torch.__file__)

if not torch.__version__.startswith("2.13.0"):
    raise SystemExit(f"Unexpected torch version: {torch.__version__}")
if not str(torch.version.cuda).startswith("12.6"):
    raise SystemExit(f"Unexpected torch CUDA runtime: {torch.version.cuda}")
if "ala-pianist-venv" in str(torch.__file__):
    raise SystemExit(f"Torch was installed into the venv instead of inherited: {torch.__file__}")
PY

WORKDIR /app
CMD ["bash"]

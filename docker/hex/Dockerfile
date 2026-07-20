# Hex/Hare runtime image for the ALA Pianist DroQ continuation.
#
# This image intentionally contains dependencies only. Mount the repository at
# /app and scratch/experiment storage at /workspace when running with Hare.
#
# Local environment used for pinning:
#   Python 3.10.20, torch 2.13.0+cpu, numpy 2.2.6, mujoco 3.10.0.
# The local PyTorch build is CPU-only, so the GPU base image must be verified on
# the selected Hex node in Prompt 2 against the live NVIDIA driver.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
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
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ala-pianist-runtime
COPY requirements.txt /opt/ala-pianist-runtime/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /opt/ala-pianist-runtime/requirements.txt

WORKDIR /app
CMD ["bash"]

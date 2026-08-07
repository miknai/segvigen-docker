# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Target: Databricks Runtime 17.3 LTS (standard, NOT the ML variant -- custom
# Docker containers on GPU compute require a standard runtime). Its GPU node
# driver supports CUDA up to 12.6 (per the matching 17.3 LTS ML release
# notes), which comfortably covers CUDA 12.4 here -- kept at 12.4 rather than
# 12.6 to match the flash-attn 2.7.3 prebuilt wheels TRELLIS.2's setup.sh
# (below) pulls in for PyTorch 2.6.0. If you switch cluster runtime versions
# later, re-verify against that runtime's release notes.
# ---------------------------------------------------------------------------
ARG CUDA_TAG=12.4.1
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# Databricks Container Services contract:
# https://docs.databricks.com/aws/en/compute/custom-containers
# Requires: a JDK on PATH, bash, sudo, coreutils, procps, iproute2, Ubuntu.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        bash sudo coreutils procps iproute2 \
        ca-certificates curl git wget \
        build-essential cmake ninja-build \
        libjpeg-dev libsm6 libxrender1 libxext6 libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# ---------------------------------------------------------------------------
# Conda + TRELLIS.2 (the 3D generative model SegviGen is built on)
# ---------------------------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"
RUN conda init bash

WORKDIR /opt
RUN git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive

WORKDIR /opt/TRELLIS.2
# Architectures for common Databricks GPU node types:
# T4=7.5, A100=8.0, A10G=8.6, L4=8.9, H100=9.0. Extend if you use other SKUs.
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# setup.sh gates everything on `command -v nvidia-smi` succeeding, but never
# actually runs it or parses its output -- confirmed against the upstream
# script. No live GPU is needed to build these extensions since
# TORCH_CUDA_ARCH_LIST above already tells PyTorch's build tooling which
# architectures to target instead of querying a device. This stub only
# needs to exist on PATH for that one check, then gets removed so it can
# never shadow the real nvidia-smi injected at runtime on the GPU cluster.
RUN printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi \
    && chmod +x /usr/local/bin/nvidia-smi

# Anaconda now requires explicit Terms of Service acceptance for its
# default channels before conda will use them, even with -y.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# setup.sh's own `conda create -n trellis2 python=3.10` has no -y flag, so
# it can't get past its confirmation prompt in a non-interactive build --
# and since the script has no `set -e` anywhere, that failure is silently
# swallowed and everything installs into conda's *base* env (Python 3.14)
# instead of trellis2. Create the env ourselves first (setup.sh's own
# create call then harmlessly no-ops/fails on "already exists").
RUN conda create -y -n trellis2 python=3.10

# Bare `conda activate` also doesn't work inside a plain `bash script.sh`
# invocation without sourcing conda's shell hook first -- do that
# ourselves so every install in this one pass actually lands in trellis2.
RUN --mount=type=cache,target=/root/.cache/pip \
    /bin/bash -lc "source ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate trellis2 && python --version && chmod +x setup.sh && bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm"

RUN rm -f /usr/local/bin/nvidia-smi

# Databricks execs python/pip directly -- it does not source .bashrc or
# `conda activate`. Prepend the trellis2 env so it's the default everywhere.
ENV PATH="${CONDA_DIR}/envs/trellis2/bin:${PATH}"

# ---------------------------------------------------------------------------
# SegviGen's own extra requirements, on top of TRELLIS.2 (see README.md)
# ---------------------------------------------------------------------------
# README pins no mathutils version, so pip grabs latest (5.1.0), whose C
# source uses PyLong_AsInt -- a CPython API only added in Python 3.13. We're
# on 3.10, so that fails to compile. Pin to 3.3.0 (the release just before
# a large gap to 5.1.0 on PyPI), which predates Python 3.13 entirely and
# lines up with the bpy==4.0.0 (Blender 4.0) version pinned below.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install mathutils==3.3.0 \
    && pip install transformers==4.57.6 \
    && pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/ \
    && pip install --upgrade Pillow

# ---------------------------------------------------------------------------
# SegviGen source. Cloned last so the dependency layers above stay cached
# across ordinary code changes. This Dockerfile lives in its own repo
# (segvigen-docker), separate from SegviGen, so the source has to be fetched
# rather than COPYed from the build context.
#
# Docker can't detect upstream commits on its own, so this layer's cache
# only busts when SEGVIGEN_REF's *value* changes. Point it at a commit SHA
# (not a branch name) for reproducible builds, and/or pass
# --build-arg SEGVIGEN_REF=<sha> (or --no-cache) from CI to force a refresh.
# ---------------------------------------------------------------------------
ARG SEGVIGEN_REF=main
RUN git clone https://github.com/Nelipot-Lee/SegviGen.git /opt/SegviGen \
    && cd /opt/SegviGen \
    && git checkout "${SEGVIGEN_REF}"
WORKDIR /opt/SegviGen

# Pretrained checkpoints (huggingface.co/fenghora/SegviGen) are intentionally
# NOT baked in -- they're large and change independently of the code. Mount
# them from a Unity Catalog volume / DBFS path at runtime and pass
# --ckpt_path accordingly, or fetch them in a cluster init script.

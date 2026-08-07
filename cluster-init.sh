#!/bin/bash
# Databricks cluster-scoped init script for SegviGen on a GPU cluster,
# as a fallback while Container Services / custom Docker images aren't
# available yet.
#
# HOW THIS DIFFERS FROM THE DOCKERFILE IN THIS REPO:
# This runs directly on the cluster host's own OS (Ubuntu 24.04 on
# Databricks Runtime 17.3), not inside an isolated container -- so it's
# the same install recipe validated in the Dockerfile, but on a different
# underlying OS. It has NOT been run/tested on an actual Databricks GPU
# cluster; treat the first run on a real cluster as a test run and expect
# to iterate, the same way the Dockerfile needed several rounds of fixes.
#
# STRATEGY: build the conda env + SegviGen once, cache it as a tarball in
# persistent storage, and on every later cluster start just extract the
# cached tarball instead of rebuilding from scratch. This is what actually
# avoids repeat installs -- an init script that reran the full install on
# every cluster start would still pay the ~25+ minute setup.sh cost each
# time.
#
# PORTABILITY NOTE: conda environments bake in absolute paths, so the
# cache is only safe to reuse because it's always extracted back to this
# exact same path (/opt/conda). Do not change CONDA_DIR below without
# clearing the cache.
#
# USAGE: only use this on a SINGLE-NODE cluster. Init scripts run on every
# node; if the cache doesn't exist yet, multiple workers would race to
# build and write it simultaneously. SegviGen's inference scripts aren't
# distributed anyway, so single-node is the right shape regardless.
#
# HOW TO RUN INFERENCE AFTER THIS SCRIPT RUNS: this does not rewire
# Databricks' own notebook Python interpreter (that's a much deeper
# integration than an init script can safely do). Invoke the trellis2 env
# directly, e.g. from a notebook %sh cell or via subprocess:
#   /opt/conda/envs/trellis2/bin/python /opt/SegviGen/inference_full.py ...
#
# SETUP: upload this file to a Workspace file or DBFS/UC Volume path, then
# reference it under cluster Advanced Options -> Init Scripts.

set -uo pipefail  # deliberately not -e: see note below setup.sh call

PERSIST_DIR="/Volumes/development/team_3d_dpc/gk-segvigen/segvigen-env"
CONDA_DIR="/opt/conda"
SEGVIGEN_DIR="/opt/SegviGen"
ENV_ARCHIVE="${PERSIST_DIR}/conda-trellis2.tar.gz"
SRC_ARCHIVE="${PERSIST_DIR}/segvigen-src.tar.gz"
LOCK_FILE="${PERSIST_DIR}/.building"

log() { echo "[segvigen-init] $*"; }

mkdir -p "${PERSIST_DIR}"

if [ -f "${ENV_ARCHIVE}" ] && [ -f "${SRC_ARCHIVE}" ]; then
    log "Found cached environment, extracting (fast path)..."
    mkdir -p "${CONDA_DIR}"
    tar -xzf "${ENV_ARCHIVE}" -C "${CONDA_DIR}"
    mkdir -p "$(dirname "${SEGVIGEN_DIR}")"
    tar -xzf "${SRC_ARCHIVE}" -C "$(dirname "${SEGVIGEN_DIR}")"
    log "Done. trellis2 env at ${CONDA_DIR}/envs/trellis2, SegviGen at ${SEGVIGEN_DIR}"
    exit 0
fi

if [ -f "${LOCK_FILE}" ]; then
    log "Another node appears to be building the cache already (${LOCK_FILE} exists)."
    log "This script is only meant for single-node clusters -- exiting without building."
    exit 0
fi
touch "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

log "No cached environment found -- building from scratch. This will take ~30+ minutes."

apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build \
    libjpeg-dev libsm6 libxrender1 libxext6 libeigen3-dev \
    curl git

curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
rm /tmp/miniconda.sh

# Anaconda requires explicit Terms of Service acceptance for its default
# channels before conda will use them, even with -y.
"${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"${CONDA_DIR}/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

"${CONDA_DIR}/bin/conda" create -y -n trellis2 python=3.10

# setup.sh gates everything on `command -v nvidia-smi` succeeding but
# never actually runs it or parses its output. A real GPU is present on
# this cluster, so this stub is likely unnecessary here -- but it's kept
# as a harmless safety net in case setup.sh runs before the GPU driver is
# fully initialized this early in cluster boot.
if ! command -v nvidia-smi > /dev/null 2>&1; then
    printf '#!/bin/sh\nexit 0\n' > /usr/local/bin/nvidia-smi
    chmod +x /usr/local/bin/nvidia-smi
fi

git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive /opt/TRELLIS.2
cd /opt/TRELLIS.2

# Common Databricks GPU node types: T4=7.5, A100=8.0, A10G=8.6, L4=8.9,
# H100=9.0. Extend if you use other SKUs.
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Same fixes validated in this repo's Dockerfile: bare `conda activate`
# doesn't work in a plain `bash script.sh` invocation without sourcing
# conda's shell hook first, and setup.sh has no `set -e` so we can't fully
# trust its own exit code -- check trellis2 actually has torch afterward.
source "${CONDA_DIR}/etc/profile.d/conda.sh"
conda activate trellis2
chmod +x setup.sh
bash setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

rm -f /usr/local/bin/nvidia-smi

python -c "import torch; assert torch.__version__.startswith('2.6.0'), torch.__version__" || {
    log "ERROR: setup.sh did not leave a working torch==2.6.0 in trellis2. Aborting cache write."
    exit 1
}

# Pinned per the same issue found and fixed in the Dockerfile: unpinned
# mathutils resolves to a release whose C source needs Python 3.13+.
pip install mathutils==3.3.0
pip install transformers==4.57.6
pip install bpy==4.0.0 --extra-index-url https://download.blender.org/pypi/
pip install --upgrade Pillow

git clone https://github.com/Nelipot-Lee/SegviGen.git "${SEGVIGEN_DIR}"

log "Build succeeded -- caching to ${PERSIST_DIR} for future cluster starts..."
tar -czf "${ENV_ARCHIVE}.tmp" -C "${CONDA_DIR}" .
mv "${ENV_ARCHIVE}.tmp" "${ENV_ARCHIVE}"
tar -czf "${SRC_ARCHIVE}.tmp" -C "$(dirname "${SEGVIGEN_DIR}")" "$(basename "${SEGVIGEN_DIR}")"
mv "${SRC_ARCHIVE}.tmp" "${SRC_ARCHIVE}"

log "Done. trellis2 env at ${CONDA_DIR}/envs/trellis2, SegviGen at ${SEGVIGEN_DIR}"

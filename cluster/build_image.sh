#!/bin/bash
# Run this ONCE interactively on the cluster to build and save the container image.
# The image .tar is saved to persistent scratch; everything else stays in home.

set -euo pipefail

ZHAW_USER="geifab01"
PROJECT_DIR="/cluster/home/${ZHAW_USER}/vpt-dreamer"
IMAGE_SCRATCH="/raid/persistent_scratch/${ZHAW_USER}"
IMAGE_NAME="vpt-dreamer:v2"
IMAGE_TAR="${IMAGE_SCRATCH}/vpt-dreamer-v2.tar"

mkdir -p "${IMAGE_SCRATCH}" && [[ -d "${IMAGE_SCRATCH}" ]] || { echo "ERROR: Could not create ${IMAGE_SCRATCH} — does /raid/persistent_scratch exist?"; exit 1; }

echo "==> Building image ${IMAGE_NAME}"
podman build -f "${PROJECT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${PROJECT_DIR}"

echo "==> Saving image to ${IMAGE_TAR}"
podman save -o "${IMAGE_TAR}" "${IMAGE_NAME}"

echo "==> Done. Image saved to ${IMAGE_TAR}"

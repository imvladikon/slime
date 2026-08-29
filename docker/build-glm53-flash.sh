#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/glm53-flash.lock"

IMAGE_NAME=${IMAGE_NAME:-slime-glm53-flash:locked}
SLIME_COMMIT=$(git -C "${REPO_ROOT}" rev-parse HEAD)

if ! git -C "${REPO_ROOT}" diff --quiet || \
  ! git -C "${REPO_ROOT}" diff --cached --quiet || \
  [[ -n "$(git -C "${REPO_ROOT}" ls-files --others --exclude-standard)" ]]; then
  echo "Refusing to build an exact GLM-5.3-Flash image from a dirty Slime worktree." >&2
  exit 1
fi

docker build \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  --build-arg "SGLANG_IMAGE=${SGLANG_IMAGE}" \
  --build-arg "SLIME_REPOSITORY=${SLIME_REPOSITORY}" \
  --build-arg "SLIME_COMMIT=${SLIME_COMMIT}" \
  --build-arg "MEGATRON_REPOSITORY=${MEGATRON_REPOSITORY}" \
  --build-arg "MEGATRON_COMMIT=${MEGATRON_COMMIT}" \
  --build-arg "SGLANG_REPOSITORY=${SGLANG_REPOSITORY}" \
  --build-arg "SGLANG_COMMIT=${SGLANG_COMMIT}" \
  --build-arg "TRANSFORMER_ENGINE_CUDA_ARCHS=${TRANSFORMER_ENGINE_CUDA_ARCHS}" \
  --build-arg "DEEPEP_CUDA_ARCH_LIST=${DEEPEP_CUDA_ARCH_LIST}" \
  --build-arg "DEEPEP_PACKAGE_VERSION=${DEEPEP_PACKAGE_VERSION}" \
  --build-arg "DEEPGEMM_PACKAGE_VERSION=${DEEPGEMM_PACKAGE_VERSION}" \
  --build-arg "BUILD_MAX_JOBS=${BUILD_MAX_JOBS:-2}" \
  --build-arg "BUILD_NVCC_THREADS=${BUILD_NVCC_THREADS:-1}" \
  --build-arg ENABLE_MEGATRON_PATCH=0 \
  --build-arg ENABLE_SGLANG_PATCH=0 \
  "${REPO_ROOT}"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/glm53-flash.lock"

IMAGE_NAME=${IMAGE_NAME:-slime-glm53-flash:locked}

docker build \
  --file "${SCRIPT_DIR}/Dockerfile" \
  --tag "${IMAGE_NAME}" \
  --build-arg "SGLANG_IMAGE=${SGLANG_IMAGE}" \
  --build-arg "MEGATRON_REPOSITORY=${MEGATRON_REPOSITORY}" \
  --build-arg "MEGATRON_COMMIT=${MEGATRON_COMMIT}" \
  --build-arg "SGLANG_REPOSITORY=${SGLANG_REPOSITORY}" \
  --build-arg "SGLANG_COMMIT=${SGLANG_COMMIT}" \
  --build-arg ENABLE_MEGATRON_PATCH=0 \
  --build-arg ENABLE_SGLANG_PATCH=0 \
  --build-arg SLIME_COMMIT=local \
  "${REPO_ROOT}"

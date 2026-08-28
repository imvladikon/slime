#!/usr/bin/env bash

# Executed qualification lane for the normalized 84M GLM-5.3-Flash proxy.
# The default `all` mode runs one SFT step, reloads its HF export, then runs two
# colocated GRPO steps so the second generation consumes the first policy update.

set -euo pipefail

MODE=${1:-all}
if [[ "${MODE}" != "sft" && "${MODE}" != "rl" && "${MODE}" != "all" ]]; then
  echo "Usage: $0 [sft|rl|all]" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
# shellcheck disable=SC1091
source "${GLM53_LOCK_FILE:-${REPO_ROOT}/docker/glm53-flash.lock}"
PYTHON_BIN=${PYTHON_BIN:-python}
TINY_CHECKPOINT=${GLM53_TINY_CHECKPOINT:-}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_ROOT=${SGLANG_ROOT:-/root/sglang-source}
RUN_ID=${GLM53_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
OUTPUT_ROOT=${GLM53_OUTPUT_ROOT:-/tmp/slime-glm53-flash-${RUN_ID}}
SFT_HF=${GLM53_SFT_HF_CHECKPOINT:-${OUTPUT_ROOT}/sft-hf_0}
ALLOW_UNPINNED=${GLM53_ALLOW_UNPINNED:-0}

if [[ "${MODE}" != "rl" && ( -z "${TINY_CHECKPOINT}" || ! -f "${TINY_CHECKPOINT}/model.safetensors" ) ]]; then
  echo "Set GLM53_TINY_CHECKPOINT to the normalized tiny checkpoint directory." >&2
  exit 2
fi
if [[ ! -d "${MEGATRON_ROOT}/megatron" ]]; then
  echo "MEGATRON_ROOT does not contain Megatron-LM: ${MEGATRON_ROOT}" >&2
  exit 2
fi
if [[ "${MODE}" != "sft" && ! -d "${SGLANG_ROOT}/python/sglang" ]]; then
  echo "SGLANG_ROOT does not contain SGLang: ${SGLANG_ROOT}" >&2
  exit 2
fi

check_revision() {
  local name=$1
  local root=$2
  local expected=$3
  local actual
  actual=$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)
  if [[ "${actual}" != "${expected}" ]]; then
    if [[ "${ALLOW_UNPINNED}" = "1" ]]; then
      echo "WARNING: ${name} HEAD ${actual:-missing} does not match ${expected}." >&2
    else
      echo "${name} HEAD ${actual:-missing} does not match locked commit ${expected}." >&2
      exit 2
    fi
  fi
}

check_revision Megatron-LM "${MEGATRON_ROOT}" "${MEGATRON_COMMIT}"
if [[ "${MODE}" != "sft" ]]; then
  check_revision SGLang "${SGLANG_ROOT}" "${SGLANG_COMMIT}"
fi
if [[ "${MODE}" != "rl" ]]; then
  "${PYTHON_BIN}" - "${TINY_CHECKPOINT}" "${TINY_NORMALIZED_MODEL_SHA256}" "${TINY_MODEL_REVISION}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_hash = sys.argv[2]
expected_revision = sys.argv[3]
digest = hashlib.sha256()
with (root / "model.safetensors").open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_hash = digest.hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"normalized tiny hash {actual_hash} does not match {expected_hash}")
with (root / "contract_normalization.json").open(encoding="utf-8") as handle:
    provenance = json.load(handle)
if provenance.get("source_revision") != expected_revision:
    raise SystemExit("normalized tiny source revision does not match the lock")
if provenance.get("normalized_model_sha256") != expected_hash:
    raise SystemExit("normalized tiny provenance hash does not match the lock")
PY
fi

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_ROOT}:${SGLANG_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_DSA_FUSE_TOPK=0
export SGLANG_CACHE_DIR=${SGLANG_CACHE_DIR:-/tmp/sglang-glm53-cache}
export RAY_USE_UVLOOP=0

CUDA_RUNTIME_LIB=$(
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import re
import site
import torch

cuda_major = int(torch.version.cuda.split(".")[0])
runtime_candidates = []
for root in site.getsitepackages():
    for candidate in Path(root).glob("nvidia/cu*/lib"):
        match = re.fullmatch(r"cu(\d+)", candidate.parent.name)
        if match and int(match.group(1)) == cuda_major:
            runtime_candidates.append(candidate)
paths = []
if runtime_candidates:
    paths.append(sorted(runtime_candidates)[0])
torch_lib = Path(torch.__file__).resolve().parent / "lib"
if torch_lib.is_dir():
    paths.append(torch_lib)
print(":".join(str(path) for path in paths))
PY
)
if [[ -n "${CUDA_RUNTIME_LIB}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

TRAIN_ENV_VARS=$(
  "${PYTHON_BIN}" - <<'PY'
import json
import os

print(json.dumps({
    "PYTHONPATH": os.environ["PYTHONPATH"],
    "PATH": os.environ["PATH"],
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}))
PY
)
RUNTIME_ENV=$(
  "${PYTHON_BIN}" - <<'PY'
import json
import os

keys = ("PYTHONPATH", "PATH", "LD_LIBRARY_PATH", "SGLANG_DSA_FUSE_TOPK", "SGLANG_CACHE_DIR", "RAY_USE_UVLOOP")
print(json.dumps({"env_vars": {key: os.environ.get(key, "") for key in keys}}))
PY
)

unset RAY_ADDRESS
if ray status >/dev/null 2>&1; then
  echo "Refusing to reuse an existing Ray cluster; stop it before qualification." >&2
  exit 2
fi
ray start --head --dashboard-host=127.0.0.1 --dashboard-port=8265 --disable-usage-stats
RAY_STARTED=1
cleanup() {
  if [[ "${RAY_STARTED}" = "1" ]]; then
    ray stop --force >/dev/null
  fi
}
trap cleanup EXIT

export GLM53_FLASH_PROFILE=tiny
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/models/glm5.3-flash.sh"

COMMON_ARGS=(
  "${MODEL_ARGS[@]}"
  --train-env-vars "${TRAIN_ENV_VARS}"
  --disable-distributed-optimizer
  --no-save-optim
  --no-load-optim
  --no-load-rng
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --transformer-impl local
  --attention-backend unfused
  --moe-token-dispatcher-type alltoall
  --recompute-granularity selective
  --recompute-modules mhc
  --seq-length 128
  --max-position-embeddings 128
  --use-dynamic-batch-size
  --max-tokens-per-gpu 256
  --optimizer adam
  --lr-decay-style constant
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.95
  --bf16
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --no-persist-layer-norm
  --no-gradient-accumulation-fusion
  --seed 1234
  --rollout-seed 1234
)

submit_job() {
  local submission_id=$1
  shift
  ray job submit \
    --address http://127.0.0.1:8265 \
    --submission-id "${submission_id}" \
    --working-dir "${REPO_ROOT}" \
    --runtime-env-json "${RUNTIME_ENV}" \
    -- "${PYTHON_BIN}" train.py "$@"
}

"${PYTHON_BIN}" - "${OUTPUT_ROOT}/run-manifest.json" "${MODE}" "${RUN_ID}" \
  "${REPO_ROOT}" "${MEGATRON_ROOT}" "${SGLANG_ROOT}" "${TINY_CHECKPOINT}" \
  "${SFT_HF}" "${ALLOW_UNPINNED}" <<'PY'
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def git_state(root):
    path = Path(root)
    if not (path / ".git").exists():
        return {"path": str(path), "head": None, "dirty": None}
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True))
    return {"path": str(path), "head": head, "dirty": dirty}


output, mode, run_id, slime, megatron, sglang, tiny, sft_hf, allow_unpinned = sys.argv[1:]
packages = {}
for name in ("ray", "safetensors", "torch", "transformers"):
    packages[name] = importlib.metadata.version(name)
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "run_id": run_id,
    "lock_enforced": allow_unpinned != "1",
    "repositories": {
        "slime": git_state(slime),
        "megatron": git_state(megatron),
        "sglang": git_state(sglang),
    },
    "tiny_checkpoint": tiny or None,
    "sft_hf_checkpoint": sft_hf,
    "packages": packages,
    "seeds": {"megatron": 1234, "rollout": 1234, "sglang": 1234},
}
Path(output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if [[ "${MODE}" = "sft" || "${MODE}" = "all" ]]; then
  submit_job "glm53-sft-${RUN_ID}" \
    "${COMMON_ARGS[@]}" \
    --hf-checkpoint "${TINY_CHECKPOINT}" \
    --num-rollout 1 \
    --ref-load "${TINY_CHECKPOINT}" \
    --save "${OUTPUT_ROOT}/sft-megatron" \
    --save-hf "${OUTPUT_ROOT}/sft-hf_{rollout_id}" \
    --save-interval 1 \
    --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
    --prompt-data tests/fixtures/glm53_flash_sft.jsonl \
    --input-key messages \
    --rollout-batch-size 2 \
    --n-samples-per-prompt 1 \
    --global-batch-size 2 \
    --loss-type sft_loss \
    --calculate-per-token-loss \
    --disable-compute-advantages-and-returns \
    --debug-train-only \
    --lr 1e-5
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/verify_glm53_flash_tiny_export.py" \
    --source "${TINY_CHECKPOINT}" \
    --candidate "${SFT_HF}" \
    --json-output "${OUTPUT_ROOT}/sft-export-verification.json"
fi

if [[ "${MODE}" = "rl" || "${MODE}" = "all" ]]; then
  if [[ ! -f "${SFT_HF}/config.json" ]]; then
    echo "RL requires an SFT HF checkpoint at ${SFT_HF}." >&2
    exit 2
  fi
  submit_job "glm53-rl-${RUN_ID}" \
    "${COMMON_ARGS[@]}" \
    --hf-checkpoint "${SFT_HF}" \
    --num-rollout 2 \
    --load "${SFT_HF}" \
    --ref-load "${SFT_HF}" \
    --save "${OUTPUT_ROOT}/rl-megatron" \
    --save-hf "${OUTPUT_ROOT}/rl-hf_{rollout_id}" \
    --save-interval 1 \
    --colocate \
    --rollout-num-gpus 1 \
    --rollout-num-gpus-per-engine 1 \
    --update-weight-mode full \
    --update-weight-transport nccl \
    --update-weight-buffer-size 67108864 \
    --check-weight-update-equal \
    --prompt-data tests/fixtures/glm53_flash_rl.jsonl \
    --input-key prompt \
    --label-key label \
    --apply-chat-template \
    --custom-rm-path slime_plugins.models.glm5_next.smoke_reward.score \
    --rollout-batch-size 1 \
    --n-samples-per-prompt 2 \
    --global-batch-size 2 \
    --advantage-estimator grpo \
    --loss-type policy_loss \
    --eps-clip 0.2 \
    --eps-clip-high 0.2 \
    --kl-coef 0 \
    --kl-loss-coef 0 \
    --entropy-coef 0 \
    --rollout-max-context-len 128 \
    --rollout-max-prompt-len 96 \
    --rollout-max-response-len 8 \
    --rollout-temperature 1 \
    --sglang-mem-fraction-static 0.45 \
    --sglang-max-running-requests 2 \
    --sglang-max-total-tokens 512 \
    --sglang-chunked-prefill-size 64 \
    --sglang-page-size 64 \
    --sglang-attention-backend dsa \
    --sglang-dsa-prefill-backend torch \
    --sglang-dsa-decode-backend torch \
    --sglang-dsa-topk-backend torch \
    --sglang-kv-cache-dtype bfloat16 \
    --sglang-cuda-graph-backend-prefill disabled \
    --sglang-cuda-graph-backend-decode disabled \
    --sglang-disable-overlap-schedule \
    --sglang-random-seed 1234 \
    --no-offload-train \
    --no-offload-rollout \
    --lr 1e-6
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/verify_glm53_flash_tiny_export.py" \
    --source "${SFT_HF}" \
    --candidate "${OUTPUT_ROOT}/rl-hf_1" \
    --json-output "${OUTPUT_ROOT}/rl-export-verification.json"
fi

echo "GLM-5.3-Flash tiny lifecycle artifacts: ${OUTPUT_ROOT}"

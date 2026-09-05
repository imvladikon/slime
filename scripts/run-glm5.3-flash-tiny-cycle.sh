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
LOCK_FILE=${GLM53_LOCK_FILE:-${REPO_ROOT}/docker/glm53-flash.lock}
source "${LOCK_FILE}"
PYTHON_BIN=${PYTHON_BIN:-python}
TINY_CHECKPOINT=${GLM53_TINY_CHECKPOINT:-}
MEGATRON_ROOT=${MEGATRON_ROOT:-/root/Megatron-LM}
SGLANG_ROOT=${SGLANG_ROOT:-/root/sglang-source}
RUN_ID=${GLM53_RUN_ID:-$(date +%Y%m%d-%H%M%S)-$$}
OUTPUT_ROOT=${GLM53_OUTPUT_ROOT:-/tmp/slime-glm53-flash-${RUN_ID}}
SFT_HF_ORACLE=${OUTPUT_ROOT}/sft-hf_0
SFT_HF_OFFLINE=${OUTPUT_ROOT}/sft-hf-offline
if [[ "${MODE}" = "rl" ]]; then
  SFT_HF=${GLM53_SFT_HF_CHECKPOINT:-${SFT_HF_OFFLINE}}
else
  # An all-mode qualification must feed its freshly produced SFT export into
  # RL; accepting an external override would silently break lifecycle coverage.
  SFT_HF=${SFT_HF_OFFLINE}
fi
RL_HF_ORACLE=${OUTPUT_ROOT}/rl-hf_1
RL_HF_OFFLINE=${OUTPUT_ROOT}/rl-hf-offline
ALLOW_UNPINNED=${GLM53_ALLOW_UNPINNED:-0}
ALLOW_EXISTING_OUTPUT=${GLM53_ALLOW_EXISTING_OUTPUT:-0}
SGLANG_MEM_FRACTION_STATIC=${GLM53_SGLANG_MEM_FRACTION_STATIC:-0.45}
SGLANG_MAX_TOTAL_TOKENS=${GLM53_SGLANG_MAX_TOTAL_TOKENS:-512}

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
  local expected_repository=$3
  local expected=$4
  local actual
  local dirty
  local repository
  if [[ -e "${root}/.git" ]]; then
    actual=$(git -C "${root}" rev-parse HEAD 2>/dev/null || true)
    dirty=$(git -C "${root}" status --porcelain 2>/dev/null || true)
    repository=$(git -C "${root}" remote get-url fork 2>/dev/null \
      || git -C "${root}" remote get-url origin 2>/dev/null || true)
  elif [[ -f "${root}/.source-provenance.json" ]]; then
    read -r actual repository < <(
      "${PYTHON_BIN}" - "${root}/.source-provenance.json" <<'PY'
import json
import sys

document = json.load(open(sys.argv[1], encoding="utf-8"))
print(document.get("commit", ""), document.get("repository", ""))
PY
    )
    dirty=
  else
    actual=
    dirty=missing
    repository=
  fi
  local normalized_repository=${repository%.git}
  normalized_repository=${normalized_repository%/}
  local normalized_expected_repository=${expected_repository%.git}
  normalized_expected_repository=${normalized_expected_repository%/}
  if [[ "${actual}" != "${expected}" || -n "${dirty}" \
    || "${normalized_repository}" != "${normalized_expected_repository}" ]]; then
    if [[ "${ALLOW_UNPINNED}" = "1" ]]; then
      echo "WARNING: ${name} does not match ${expected_repository}@${expected} " \
        "(source ${repository:-missing}, HEAD ${actual:-missing}, dirty ${dirty:-false})." >&2
    else
      echo "${name} must match ${expected_repository}@${expected} " \
        "(source ${repository:-missing}, HEAD ${actual:-missing}, dirty ${dirty:-false})." >&2
      exit 2
    fi
  fi
}

if [[ -n "${GLM53_EXPECTED_SLIME_REPOSITORY:-}" || -n "${GLM53_EXPECTED_SLIME_COMMIT:-}" ]]; then
  : "${GLM53_EXPECTED_SLIME_REPOSITORY:?Set GLM53_EXPECTED_SLIME_REPOSITORY with GLM53_EXPECTED_SLIME_COMMIT}"
  : "${GLM53_EXPECTED_SLIME_COMMIT:?Set GLM53_EXPECTED_SLIME_COMMIT with GLM53_EXPECTED_SLIME_REPOSITORY}"
  check_revision Slime "${REPO_ROOT}" \
    "${GLM53_EXPECTED_SLIME_REPOSITORY}" "${GLM53_EXPECTED_SLIME_COMMIT}"
fi
check_revision Megatron-LM "${MEGATRON_ROOT}" "${MEGATRON_REPOSITORY}" "${MEGATRON_COMMIT}"
if [[ "${MODE}" != "sft" ]]; then
  check_revision SGLang "${SGLANG_ROOT}" "${SGLANG_REPOSITORY}" "${SGLANG_COMMIT}"
fi
if [[ "${MODE}" != "rl" ]]; then
  "${PYTHON_BIN}" - "${TINY_CHECKPOINT}" "${TINY_NORMALIZED_MODEL_SHA256}" \
    "${TINY_NORMALIZED_CONFIG_SHA256}" "${TINY_MODEL_REVISION}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_hash = sys.argv[2]
expected_config_hash = sys.argv[3]
expected_revision = sys.argv[4]
digest = hashlib.sha256()
with (root / "model.safetensors").open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
actual_hash = digest.hexdigest()
if actual_hash != expected_hash:
    raise SystemExit(f"normalized tiny hash {actual_hash} does not match {expected_hash}")
actual_config_hash = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
if actual_config_hash != expected_config_hash:
    raise SystemExit(
        f"normalized tiny config hash {actual_config_hash} does not match {expected_config_hash}"
    )
with (root / "contract_normalization.json").open(encoding="utf-8") as handle:
    provenance = json.load(handle)
if provenance.get("source_revision") != expected_revision:
    raise SystemExit("normalized tiny source revision does not match the lock")
if provenance.get("normalized_model_sha256") != expected_hash:
    raise SystemExit("normalized tiny provenance hash does not match the lock")
if provenance.get("normalized_config_sha256") != expected_config_hash:
    raise SystemExit("normalized tiny config provenance hash does not match the lock")
PY
fi

if [[ "${GLM53_PREFLIGHT_ONLY:-0}" = "1" ]]; then
  echo "GLM-5.3-Flash source and checkpoint preflight passed."
  exit 0
fi

if [[ -d "${OUTPUT_ROOT}" && -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  if [[ "${ALLOW_EXISTING_OUTPUT}" != "1" ]]; then
    echo "Refusing nonempty GLM53_OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
    echo "Use a new run directory, or set GLM53_ALLOW_EXISTING_OUTPUT=1 explicitly." >&2
    exit 2
  fi
  echo "WARNING: reusing nonempty output directory ${OUTPUT_ROOT}." >&2
fi
mkdir -p "${OUTPUT_ROOT}"
MANIFEST_PATH=${OUTPUT_ROOT}/run-manifest.json
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

set_manifest_status() {
  local status=$1
  local exit_code=$2
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${status}" "${exit_code}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(0)
document = json.loads(path.read_text(encoding="utf-8"))
document["status"] = sys.argv[2]
document["exit_code"] = int(sys.argv[3])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

record_checkpoint() {
  local stage=$1
  local checkpoint=$2
  "${PYTHON_BIN}" - "${MANIFEST_PATH}" "${stage}" "${checkpoint}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
stage = sys.argv[2]
root = Path(sys.argv[3]).resolve()
config = root / "config.json"
index = root / "model.safetensors.index.json"
single = root / "model.safetensors"
if not config.is_file():
    raise SystemExit(f"checkpoint config is missing: {config}")
if index.is_file():
    document = json.loads(index.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid checkpoint index: {index}")
    files = [config, index] + [root / name for name in sorted(set(weight_map.values()))]
elif single.is_file():
    files = [config, single]
else:
    raise SystemExit(f"checkpoint weights are missing: {root}")

digest = hashlib.sha256()
file_records = []
for path in files:
    if not path.is_file():
        raise SystemExit(f"checkpoint file is missing: {path}")
    relative = path.relative_to(root).as_posix()
    size = path.stat().st_size
    digest.update(relative.encode("utf-8") + b"\0" + str(size).encode("ascii") + b"\0")
    file_digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            file_digest.update(chunk)
            digest.update(chunk)
    file_records.append({"path": relative, "bytes": size, "sha256": file_digest.hexdigest()})

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest.setdefault("checkpoints", {})[stage] = {
    "path": str(root),
    "sha256": digest.hexdigest(),
    "files": file_records,
}
temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, manifest_path)
PY
}

compare_hf_exact() {
  local left=$1
  local right=$2
  "${PYTHON_BIN}" - "${left}" "${right}" <<'PY'
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        document = json.loads(index.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in document["weight_map"].items()}
    single = root / "model.safetensors"
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}


left, right = map(Path, sys.argv[1:])
maps = [weight_map(left), weight_map(right)]
if set(maps[0]) != set(maps[1]):
    missing = sorted(set(maps[0]) - set(maps[1]))
    extra = sorted(set(maps[1]) - set(maps[0]))
    raise SystemExit(f"checkpoint keys differ: missing={missing[:8]}, extra={extra[:8]}")

handles = [{}, {}]
try:
    for key in sorted(maps[0]):
        tensors = []
        for side, root in enumerate((left, right)):
            filename = maps[side][key]
            handle = handles[side].get(filename)
            if handle is None:
                handle = safe_open(root / filename, framework="pt", device="cpu")
                handles[side][filename] = handle
            tensors.append(handle.get_tensor(key))
        if tensors[0].shape != tensors[1].shape or tensors[0].dtype != tensors[1].dtype:
            raise SystemExit(
                f"{key} metadata differs: "
                f"{tensors[0].shape}/{tensors[0].dtype} != {tensors[1].shape}/{tensors[1].dtype}"
            )
        if not torch.equal(tensors[0], tensors[1]):
            raise SystemExit(f"{key} values differ")
finally:
    handles.clear()

print(f"Exact HF parity: {len(maps[0])} tensors")
PY
}

convert_torch_dist_offline() {
  local input=$1
  local origin=$2
  local output=$3
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/convert_torch_dist_to_hf_ray.py" \
    --input-dir "${input}" \
    --origin-hf-dir "${origin}" \
    --output-dir "${output}" \
    --model-name glm5next \
    --concurrency 1 \
    --task-group-bytes 67108864 \
    --max-file-bytes 16777216 \
    --no-progress
}

unset RAY_ADDRESS
if ray status >/dev/null 2>&1; then
  echo "Refusing to reuse an existing Ray cluster; stop it before qualification." >&2
  exit 2
fi
ray start --head --dashboard-host=127.0.0.1 --dashboard-port=8265 --disable-usage-stats
RAY_STARTED=1
cleanup() {
  local exit_code=$?
  if [[ "${exit_code}" != "0" ]]; then
    set +e
    set_manifest_status failed "${exit_code}"
    set -e
  fi
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
  mkdir -p "${OUTPUT_ROOT}/logs"
  ray job submit \
    --address http://127.0.0.1:8265 \
    --submission-id "${submission_id}" \
    --working-dir "${REPO_ROOT}" \
    --runtime-env-json "${RUNTIME_ENV}" \
    -- "${PYTHON_BIN}" train.py "$@" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/${submission_id}.log"
}

"${PYTHON_BIN}" - "${MANIFEST_PATH}" "${MODE}" "${RUN_ID}" \
  "${REPO_ROOT}" "${MEGATRON_ROOT}" "${SGLANG_ROOT}" "${TINY_CHECKPOINT}" \
  "${SFT_HF}" "${ALLOW_UNPINNED}" "${ALLOW_EXISTING_OUTPUT}" "${LOCK_FILE}" \
  "${SGLANG_IMAGE}" "${MEGATRON_REPOSITORY}" "${MEGATRON_COMMIT}" \
  "${SGLANG_REPOSITORY}" "${SGLANG_COMMIT}" "${FULL_MODEL_REPOSITORY}" \
  "${FULL_MODEL_REVISION}" "${FULL_CONFIG_SHA256}" "${FULL_INDEX_SHA256}" \
  "${FULL_HEADERS_SHA256}" "${TINY_MODEL_REPOSITORY}" "${TINY_MODEL_REVISION}" \
  "${TINY_NORMALIZED_MODEL_SHA256}" "${TINY_NORMALIZED_CONFIG_SHA256}" \
  "${TRANSFORMER_ENGINE_CUDA_ARCHS}" \
  "${DEEPEP_CUDA_ARCH_LIST}" "${DEEPEP_PACKAGE_VERSION}" \
  "${DEEPGEMM_PACKAGE_VERSION}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def source_state(root):
    path = Path(root)
    if (path / ".git").exists():
        head = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(path), "status", "--porcelain"], text=True
            )
        )
        repository = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "fork"],
            check=False,
            capture_output=True,
            text=True,
        )
        if repository.returncode != 0:
            repository = subprocess.run(
                ["git", "-C", str(path), "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True,
            )
        return {
            "path": str(path),
            "repository": repository.stdout.strip(),
            "commit": head,
            "dirty": dirty,
            "provenance": "git",
        }
    provenance_path = path / ".source-provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        return {
            "path": str(path),
            "repository": provenance.get("repository"),
            "commit": provenance.get("commit"),
            "dirty": False,
            "provenance": "manifest",
        }
    return {
        "path": str(path),
        "repository": None,
        "commit": None,
        "dirty": None,
        "provenance": "missing",
    }


(
    output,
    mode,
    run_id,
    slime,
    megatron,
    sglang,
    tiny,
    sft_hf,
    allow_unpinned,
    allow_existing_output,
    lock_file,
    sglang_image,
    megatron_repository,
    megatron_commit,
    sglang_repository,
    sglang_commit,
    full_model_repository,
    full_model_revision,
    full_config_sha256,
    full_index_sha256,
    full_headers_sha256,
    tiny_model_repository,
    tiny_model_revision,
    tiny_normalized_sha256,
    tiny_normalized_config_sha256,
    transformer_engine_cuda_archs,
    deepep_cuda_arch_list,
    deepep_package_version,
    deepgemm_package_version,
) = sys.argv[1:]
packages = {}
for name in ("ray", "safetensors", "torch", "transformers"):
    packages[name] = importlib.metadata.version(name)
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "run_id": run_id,
    "lock_enforced": allow_unpinned != "1",
    "existing_output_allowed": allow_existing_output == "1",
    "status": "running",
    "exit_code": None,
    "repositories": {
        "slime": source_state(slime),
        "megatron": source_state(megatron),
        "sglang": source_state(sglang),
    },
    "tiny_checkpoint": tiny or None,
    "sft_hf_checkpoint": sft_hf,
    "packages": packages,
    "seeds": {"megatron": 1234, "rollout": 1234, "sglang": 1234},
    "lock": {
        "path": str(Path(lock_file).resolve()),
        "sha256": hashlib.sha256(Path(lock_file).read_bytes()).hexdigest(),
        "values": {
            "SGLANG_IMAGE": sglang_image,
            "MEGATRON_REPOSITORY": megatron_repository,
            "MEGATRON_COMMIT": megatron_commit,
            "SGLANG_REPOSITORY": sglang_repository,
            "SGLANG_COMMIT": sglang_commit,
            "FULL_MODEL_REPOSITORY": full_model_repository,
            "FULL_MODEL_REVISION": full_model_revision,
            "FULL_CONFIG_SHA256": full_config_sha256,
            "FULL_INDEX_SHA256": full_index_sha256,
            "FULL_HEADERS_SHA256": full_headers_sha256,
            "TINY_MODEL_REPOSITORY": tiny_model_repository,
            "TINY_MODEL_REVISION": tiny_model_revision,
            "TINY_NORMALIZED_MODEL_SHA256": tiny_normalized_sha256,
            "TINY_NORMALIZED_CONFIG_SHA256": tiny_normalized_config_sha256,
            "TRANSFORMER_ENGINE_CUDA_ARCHS": transformer_engine_cuda_archs,
            "DEEPEP_CUDA_ARCH_LIST": deepep_cuda_arch_list,
            "DEEPEP_PACKAGE_VERSION": deepep_package_version,
            "DEEPGEMM_PACKAGE_VERSION": deepgemm_package_version,
        },
    },
}
path = Path(output)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY

if [[ "${MODE}" = "rl" ]]; then
  record_checkpoint rl_input_sft "${SFT_HF}"
fi

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
    --candidate "${SFT_HF_ORACLE}" \
    --json-output "${OUTPUT_ROOT}/sft-live-export-verification.json"
  convert_torch_dist_offline \
    "${OUTPUT_ROOT}/sft-megatron/iter_0000000" \
    "${TINY_CHECKPOINT}" \
    "${SFT_HF_OFFLINE}"
  compare_hf_exact "${SFT_HF_ORACLE}" "${SFT_HF_OFFLINE}"
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/verify_glm53_flash_tiny_export.py" \
    --source "${TINY_CHECKPOINT}" \
    --candidate "${SFT_HF_OFFLINE}" \
    --json-output "${OUTPUT_ROOT}/sft-offline-export-verification.json"
  record_checkpoint sft_live_oracle "${SFT_HF_ORACLE}"
  record_checkpoint sft_output "${SFT_HF_OFFLINE}"
fi

if [[ "${MODE}" = "rl" || "${MODE}" = "all" ]]; then
  if [[ ! -f "${SFT_HF}/config.json" ]]; then
    echo "RL requires an SFT HF checkpoint at ${SFT_HF}." >&2
    exit 2
  fi
  if [[ "${MODE}" = "all" ]]; then
    record_checkpoint rl_input_sft "${SFT_HF}"
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
    --use-rollout-routing-replay \
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
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
    --sglang-max-running-requests 2 \
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}" \
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
    --candidate "${RL_HF_ORACLE}" \
    --json-output "${OUTPUT_ROOT}/rl-live-export-verification.json"
  convert_torch_dist_offline \
    "${OUTPUT_ROOT}/rl-megatron/iter_0000001" \
    "${SFT_HF}" \
    "${RL_HF_OFFLINE}"
  compare_hf_exact "${RL_HF_ORACLE}" "${RL_HF_OFFLINE}"
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/verify_glm53_flash_tiny_export.py" \
    --source "${SFT_HF}" \
    --candidate "${RL_HF_OFFLINE}" \
    --json-output "${OUTPUT_ROOT}/rl-offline-export-verification.json"
  record_checkpoint rl_live_oracle "${RL_HF_ORACLE}"
  record_checkpoint rl_output "${RL_HF_OFFLINE}"
fi

set_manifest_status succeeded 0

echo "GLM-5.3-Flash tiny lifecycle artifacts: ${OUTPUT_ROOT}"

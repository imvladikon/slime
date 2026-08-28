# GLM-5.3-Flash lifecycle qualification

GLM-5.3-Flash is the `glm5_next` architecture. It is not interchangeable with
GLM-5.3 or the older `glm_moe_dsa` GLM-5/5.2 adapters. This lane trains the
45-layer backbone with Megatron, uses SGLang for rollout, keeps the released
MTP layer disabled, and freezes the DSA indexer exactly as the reference
training implementation does.

## Locked source lane

The dependency lock is `docker/glm53-flash.lock`:

- CUDA 13 base image: `lmsysorg/sglang:glm-5.3-flash@sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6`
- Megatron-LM: `imvladikon/Megatron-LM@630d81f0d3ffe2e178973470bfe7ff0d3dc45333`
- SGLang: `imvladikon/sglang@9e89492422bdca00ac643e78d7c62f5df8a59e49`
- full model metadata: `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`
- tiny source: `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B@7c3a6d3dc51732dd8ab230888e06ba8c93a381ac`

Build the current Slime checkout with those exact dependency commits:

```bash
IMAGE_NAME=slime-glm53-flash:locked docker/build-glm53-flash.sh
```

The build disables both legacy patch stacks. The locked SGLang fork already
contains the qualified compact-DSA path and scoped weight checker, while the
locked Megatron fork already contains the Slime runtime seams plus the required
GLM mHC/recompute fixes. Those older patches do not apply cleanly to these
commits and must not be layered on top.

This is a core GPU/source lock, not a bit-reproducible lock of every secondary
package. The upstream Dockerfile still contains mutable apt packages, TileLang
nightlies, and several broad Python requirements. The build nevertheless fails
closed on the CUDA 13/Torch 2.13 ABI, SGLang import, and `pip check`; archive the
resulting image digest before a production run.

`imvladikon/verl` is not a dependency of this lane. It remains an alternative
FSDP/VERL workflow; the primary lifecycle here is Slime's Megatron actor plus
SGLang rollout.

## Model contract

The released checkpoint contains:

- 45 backbone layers: 34 KDA and 11 DSA;
- three dense layers followed by 42 MoE layers;
- 288 routed experts with top-8 routing and one shared expert;
- mHC width 4 and IndexPool4;
- a 24-layer vision tower;
- one additional MTP layer after the backbone.

`scripts/models/glm5.3-flash.sh` validates this config when the model provider
starts. Training intentionally sets `--mtp-num-layers 0`; speculative rollout
must therefore remain disabled until MTP import, gradients, synchronization,
export, and reload have their own lifecycle gate.

The public tiny checkpoint has a stale contract: its config advertises MTP
weights that are absent, declares the wrong QK width, and contains incorrect
dtypes. Loading it with Transformers and calling `save_pretrained()` preserves
those semantics; it does not infer the missing architecture. The tiny lifecycle
requires an explicitly normalized fixture with `qk_head_dim=64`, the exact
KDA/DSA schedule, no MTP layer, BF16 KDA convolution weights, and FP32 mHC
base/scale state.

## Executed tiny lifecycle

The following command is the single-GPU qualification used on an RTX 3070 with
8 GiB. It performs one real SFT optimizer step and checkpoint, then two real
GRPO iterations through SGLang. The second generation therefore consumes the
policy produced by the first Megatron backward/optimizer step. Every post-step
sync is destructively reset, repeated, and compared before the next rollout:

```bash
export GLM53_TINY_CHECKPOINT=/models/GLM-5.3-Flash-0.1B-A0.1B-contract
export MEGATRON_ROOT=/root/Megatron-LM
export SGLANG_ROOT=/root/sglang-source
scripts/run-glm5.3-flash-tiny-cycle.sh all
```

The local capacity profile deliberately uses:

```text
--no-offload-train
--no-offload-rollout
--sglang-max-total-tokens 512
--weight-checker-skip-prefix visual.
```

The offload flags avoid the observed torch-memory-saver VMM allocator versus
CUDA IPC incompatibility. More importantly, SGLang discards main weights while
sleeping unless its CPU backup is enabled; because frozen vision weights are not
part of language synchronization, rollout offload would silently destroy that
tower. The token cap bounds KV allocation on the 8 GiB card. The
checker prefix excludes the frozen vision tower from the destructive sync test;
all synchronized language state, including KDA, DSA, mHC, MoE, embeddings, and
LM head, is still reset and compared.

The launcher qualification produced a nonzero gradient norm (`14.8053`), generated two
samples with raw mean reward `0.5`, synchronized three weight buckets before and
after the policy step, and exported 223/223 expected tensors. Exactly 160
language tensors changed, while all 39 vision tensors stayed bitwise unchanged.
Standard Transformers reloaded the result as
`Glm5NextForConditionalGeneration` with 84,361,950 parameters.

## Metadata-only full-scale preflight

Download only config and index metadata. Do not download any full
`.safetensors` payload:

```bash
mkdir -p /shared/GLM-5.3-Flash-metadata
hf download zai-org/GLM-5.3-Flash \
  config.json model.safetensors.index.json \
  --revision 04c4e9e95c5da8862dced7e5056455116f83a7e0 \
  --local-dir /shared/GLM-5.3-Flash-metadata
```

The preflight can fetch only the bounded JSON header range from every shard. It
rejects any server that returns a full response and refuses to run if a weight
payload appears in the metadata directory:

```bash
tools/glm53_flash_preflight.py \
  --config /shared/GLM-5.3-Flash-metadata/config.json \
  --index /shared/GLM-5.3-Flash-metadata/model.safetensors.index.json \
  --headers /shared/GLM-5.3-Flash-metadata/safetensors_headers.json \
  --fetch-missing-headers \
  --mode rl \
  --tp 8 --pp 1 --cp 1 --ep 72 --etp 1 --dp 72 \
  --rollout-num-gpus 576 --rollout-tp 8 --rollout-ep 8 \
  --rollout-moe-dp 1 --rollout-gpus-per-engine 8 \
  --no-offload-train --no-offload-rollout \
  --gpu-memory-gib 141 \
  --actor-runtime-reserve-gib 24 \
  --rollout-runtime-reserve-gib 16 \
  --json-output glm53-flash-preflight.json
```

For that analytical H200 example, the tool derives from all 76,108 tensor
headers:

| Per actor rank | GiB |
| --- | ---: |
| BF16/FP32 parameters, including replicated vision | 11.48 |
| FP32 gradient buffer | 20.56 |
| distributed Adam state | 6.11 |
| actor persistent floor | 38.15 |
| SGLang runtime model floor | 38.74 |
| conservative phase peak with explicit reserve | 100.90 |
| 90% of a 141 GiB GPU | 126.90 |

This is a lower-bound gate, not an OOM guarantee. The 24/16 GiB reserves must
be replaced with measured activation, CUDA graph, grouped-GEMM, DSA, KV, and
allocator peaks on the target image. The same topology correctly fails the
default gate on an 80 GiB H100.

The preflight also exposes scale costs that parameter-only estimates hide:

- train-side backbone synchronization source: 583.62 GiB;
- rollout runtime target per replica: 297.82 GiB;
- largest BF16 source tensor: 1.1816 GiB;
- minimum flattened CUDA IPC transient for that tensor: 2.3633 GiB;
- 72 colocated rollout replicas on 576 actor GPUs: 20.94 TiB of aggregate target
  writes per policy update.

The calculation includes 1.05 GiB/rank for the fully replicated frozen vision
tower, 8.40 GiB/node of vision loader staging, and exactly 74,514,944 bytes/rank
of SGLang FP32 upcasts for mHC, KDA convolution, and DSA indexer state. A rollout
CPU weight backup, if explicitly enabled, is charged separately to host memory.

Current RL gates fail closed for PP greater than one, CP greater than one,
actor ETP greater than one, unsafe rollout/train offload, and non-colocated full
sync. They also require Slime's derived TP to equal GPUs/engine and require
`GPUs/engine = SGLang EP * MoE-DP`, with EPLB, redundant experts, and elastic EP
disabled. The remote NCCL path broadcasts unsharded full tensors to rollout
ranks; it is not a production transport for this model.

The analytical candidate is 72 nodes x 8 H200, actor TP8/EP72/ETP1/PP1/CP1,
and 576 colocated rollout GPUs arranged as 72 engines with TP8/EP8/MoE-DP1:

```text
--actor-num-nodes 72 --actor-num-gpus-per-node 8
--tensor-model-parallel-size 8 --expert-model-parallel-size 72
--expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1
--context-parallel-size 1
--rollout-num-gpus 576 --rollout-num-gpus-per-engine 8
--sglang-ep-size 8 --sglang-moe-dp-size 1
--no-offload-train --no-offload-rollout --mtp-num-layers 0
```

This remains an analytical candidate, not a qualified production topology,
until a multi-rank expert-routing test (minimum TP4/EP4) and target-H200 peak
measurements pass.

## Full export gate

The 321B post-training export is checked without reloading the full model. The
verifier compares the candidate shard headers against all expected backbone
keys, shapes, and training dtypes, proves that MTP and FP8 scale metadata were
removed, reloads only `AutoConfig`, and can optionally stream the 347 frozen
vision tensors one at a time when source weights already exist locally:

```bash
tools/verify_glm53_flash_full_export.py \
  --source-config /shared/GLM-5.3-Flash-metadata/config.json \
  --source-index /shared/GLM-5.3-Flash-metadata/model.safetensors.index.json \
  --source-headers /shared/GLM-5.3-Flash-metadata/safetensors_headers.json \
  --candidate /checkpoints/glm53-flash-hf \
  --json-output glm53-flash-export-verification.json
```

Passing `--source-weights /shared/GLM-5.3-Flash` adds the streamed frozen-value
comparison. The tool never downloads weights. A standard full-model
Transformers reload is intentionally marked `NOT_RUN_BY_DESIGN`; it would need
hundreds of GiB and is replaced by exact header/config validation plus the
runtime SGLang load gate on the target cluster.

## Lifecycle ownership

```text
pinned HF config/index/headers
  -> strict glm5_next contract and topology preflight
  -> HF-to-Megatron FP8 dequantization into BF16/FP32 training state
  -> Megatron SFT or policy backward and optimizer step
  -> shard-aware colocated tensor routing and optional FP8 requantization
  -> SGLang DSA/KPool rollout with MTP disabled
  -> scoped destructive equality check
  -> Megatron torch_dist checkpoint and normalized HF export
  -> tiny: standard Transformers reload
  -> full: exact header/config gate and target-cluster SGLang load
```

The tiny lane owns key/shape/dtype/value checks at every transition. The full
lane owns config/header checks without the weight payload; frozen vision values
are checked only when `--source-weights` is supplied. A rollout tolerance must
not be widened to compensate for a checkpoint, mapping, or synchronization
failure.

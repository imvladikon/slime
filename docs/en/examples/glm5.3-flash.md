# GLM-5.3-Flash lifecycle qualification

GLM-5.3-Flash is the `glm5_next` architecture. It is not interchangeable with
GLM-5.3 or the older `glm_moe_dsa` GLM-5/5.2 adapters. This lane trains the
45-layer backbone with Megatron, uses SGLang for rollout, keeps the released
MTP layer disabled, and freezes the DSA indexer exactly as the reference
training implementation does.

## Locked source lane

The dependency lock is `docker/glm53-flash.lock`:

- CUDA 13 base image: `lmsysorg/sglang:glm-5.3-flash@sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6`
- Megatron-LM: `imvladikon/Megatron-LM@15e885973f94d72a1ae2365761b4ec4f32176e1a`
- SGLang: `imvladikon/sglang@a935cba7215930d551b416e963a7bb14872e5af6`
- full model metadata: `zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0`
- tiny source: `inference-optimization/GLM-5.3-Flash-0.1B-A0.1B@7c3a6d3dc51732dd8ab230888e06ba8c93a381ac`

The SGLang cookbook linked by Z.ai is the official recommended serving baseline
for an existing HF checkpoint. It does not cover Megatron training,
Megatron-to-HF export, live policy synchronization, or R3 routing replay. The
pinned SGLang engine already implements GLM-5.3-Flash serving; this lane adds
those training seams. In particular, the Docker build compiles
`sglang-router` from the same pinned SGLang source because the older prebuilt
gateway silently discarded native `/generate` extensions such as
`return_routed_experts`.

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

The purpose image's `sgl-deep-ep==0.1.2` is retained as the sole owner of the
`deep_ep` import tree. Its CUDA-13 extension was inspected with `cuobjdump`
13.3 and contains native `sm_90`, `sm_100`, and `sm_103` cubins. The Docker
build repeats the metadata-owner, exact-version, and all three architecture
checks without importing DeepEP (its import-time prerequisite gate requires a
visible GPU).

`imvladikon/verl` is not a dependency of this lane. It remains an alternative
FSDP/VERL workflow; the primary lifecycle here is Slime's Megatron actor plus
SGLang rollout.

The Slime base already includes GLM-5.2 train/rollout alignment PR #2262,
including the pinned batch-invariant DeepGEMM and aligned DeepEP kernels. Those
primitives are useful evidence and a small-topology diagnostic, but they are
not blindly enabled for GLM-5.3-Flash: the accepted gate is GLM-5.2 EP8, while
the analytical full actor uses EP72 and a different `glm5_next` architecture.
The full profile therefore uses Megatron all-to-all for training and DeepEP
only inside each SGLang EP8 rollout engine.

The qualified RL algorithm is GRPO/policy-only. Slime's current PPO path forces
critic-related training offload, which this GLM VLM runtime rejects; PPO needs a
separately qualified non-offloaded or separate-GPU critic design.

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

Create that fixture from the locked small source checkpoint, never from the
full release:

```bash
tools/normalize_glm53_flash_tiny.py \
  --source /models/GLM-5.3-Flash-0.1B-A0.1B \
  --output /models/GLM-5.3-Flash-0.1B-A0.1B-contract
```

The tool writes `contract_normalization.json`. The lifecycle launcher verifies
its source revision, model hash
`c8859ffe8b82f4e7346f49abaef98b52378f95a32df2c32e18b5156890857d84`,
and config hash
`ffe59c74dc9f3bcb9eec45129e1d0d57891c183769462035924aa42968d7e34f`
before starting Ray, and records all repository revisions, package versions,
checkpoint tree digests, and seeds in `run-manifest.json`.

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
--weight-checker-frozen-prefix visual.
```

The offload flags avoid the observed torch-memory-saver VMM allocator versus
CUDA IPC incompatibility. More importantly, SGLang discards main weights while
sleeping unless its CPU backup is enabled; because frozen vision weights are not
part of language synchronization, rollout offload would silently destroy that
tower. The token cap bounds KV allocation on the 8 GiB card. The
checker prefix excludes the frozen vision tower from the destructive sync test;
all synchronized language state, including KDA, DSA, mHC, MoE, embeddings, and
LM head, is still reset and compared. The separate frozen prefix computes a
non-destructive SGLang checksum before the first rollout and after the last one,
so the excluded vision tower is still required to stay bitwise unchanged.

All framework and rollout seeds are fixed. SGLang's global
`--enable-deterministic-inference` is intentionally not used because the pinned
runtime rejects it with the DSA attention backend; exact token reproducibility
is therefore not claimed for DSA kernels.

The final clean run `final-13ffd738` was executed from Slime commit
`13ffd738912c4653e12cbc5d2901c49eea5a568f`. Its SFT step produced loss
`10.755826` and gradient norm `85.069893`. The two RL iterations then produced
gradient norms `14.812646` and `12.890700`, entropies `2.901259` and `2.911986`,
and policy/reference log-probability absolute differences `0.001825` and
`0.004316`; each iteration generated two samples with raw mean reward `0.5`.
The first policy update advanced the rollout version from 1 to 2, reset and
resynchronized it as version 3, and the second generation consumed version 3.
The second update similarly advanced through versions 4 and 5. Both post-step
destructive comparisons passed, as did the independent `visual.*` checksum
across the complete two-rollout lifecycle.

The strict final export gate found all 223 expected tensors. Exactly 160
language tensors changed, all 39 vision tensors and all seven frozen DSA indexer
tensors stayed bitwise unchanged, and every floating tensor was finite.
Standard Transformers reloaded both trained results as
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
  --host-memory-gib 512 \
  --actor-runtime-reserve-gib 24 \
  --rollout-runtime-reserve-gib 16 \
  --rollout-cache-reserve-gib 16 \
  --sglang-mem-fraction-static 0.54 \
  --sglang-moe-a2a-backend deepep \
  --sglang-deepep-mode auto \
  --update-weight-buffer-bytes 536870912 \
  --use-rollout-routing-replay \
  --rollout-batch-size 8 --n-samples-per-prompt 8 \
  --rollout-max-response-len 65536 \
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
| memory visible to SGLang before model load | 102.85 |
| SGLang static model/KV/KDA allocation (54%) | 55.54 |
| KV/KDA cache headroom above the rollout model | 16.79 |
| non-static SGLang activation/workspace slack | 47.31 |
| colocated actor phase peak | 117.69 |
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
- the 512 MiB expert update buffer caps BF16 staging plus FP8/IPC copies at an
  analytical 1 GiB/rank; each rollout EP rank receives 1.6875 GiB per MoE
  layer, requiring at least four batches per layer and 168 per policy update;
- 72 colocated rollout replicas on 576 actor GPUs: 21.20 TiB of aggregate target
  device writes per policy update;
- rank-local expert transfer before FP8 conversion: at most 39.87 TiB of BF16
  P2P traffic per update (39.31 TiB if each source skips one local target).
- R3 logical expert routes cost 1,440 bytes per generated token; the configured
  8 prompts x 8 samples x 65,536-token upper bound is 5.625 GiB of raw route
  metadata per rollout before serialization or allocator overhead.

Expert source ownership is byte-balanced across all eight physical Megatron
replicas instead of selecting the first owner. Under the candidate topology the
analytical upper bound is 70.88 GiB egress per actor rank, or 567 GiB per
eight-GPU node, before dense collectives and protocol overhead. Runtime metrics
report the actual maximum planned source bytes; target-cluster acceptance must
reject material skew from this balanced plan.

The calculation includes 1.05 GiB/rank for the fully replicated frozen vision
tower, 8.40 GiB/node of vision loader staging, and exactly 74,514,944 bytes/rank
of SGLang FP32 upcasts for mHC, KDA convolution, and DSA indexer state. A rollout
CPU weight backup, if explicitly enabled, is charged separately to host memory.

Current RL gates fail closed for PP greater than one, CP greater than one,
actor ETP greater than one, unsafe rollout/train offload, and non-colocated full
sync. They also require Slime's derived TP to equal GPUs/engine and require
`GPUs/engine = SGLang EP * MoE-DP`, with EPLB, redundant experts, and elastic EP
disabled. The remote NCCL path broadcasts unsharded full tensors to rollout
ranks; it is not a production transport for this model. The full model profile
also sets `--require-rank-local-expert-update`, so a missing or invalid expert
transfer plan aborts instead of silently falling back to full expert broadcast.
Connection-time runtime metrics expose whether that plan was active and its
planned BF16 source/target bytes; they are not measured post-FP8 wire traffic.

The analytical candidate is 72 nodes x 8 H200, actor TP8/EP72/ETP1/PP1/CP1,
and 576 colocated rollout GPUs arranged as 72 engines with TP8/EP8/MoE-DP1:

```text
--actor-num-nodes 72 --actor-num-gpus-per-node 8
--tensor-model-parallel-size 8 --expert-model-parallel-size 72
--expert-tensor-parallel-size 1 --pipeline-model-parallel-size 1
--context-parallel-size 1
--transformer-impl transformer_engine
--recompute-granularity selective --recompute-modules mhc moe_act
--rollout-num-gpus 576 --rollout-num-gpus-per-engine 8
--sglang-ep-size 8 --sglang-moe-dp-size 1
--sglang-moe-a2a-backend deepep --sglang-deepep-mode auto
--sglang-dsa-prefill-backend tilelang
--sglang-dsa-decode-backend tilelang
--sglang-kv-cache-dtype bfloat16
--sglang-moe-runner-backend deep_gemm
--sglang-disable-shared-experts-fusion
--sglang-mem-fraction-static 0.54
--use-rollout-routing-replay
--sglang-weight-loader-prefetch-checkpoints
--require-rank-local-expert-update
--update-weight-buffer-size 536870912
--moe-token-dispatcher-type alltoall
--no-offload-train --no-offload-rollout --mtp-num-layers 0
```

This is intentionally larger than AutoModel's validated 72-GPU full-SFT
recipe (TP1/EP72/CP2): that recipe does not keep an SGLang replica colocated
with the actor.  Applying the same TP1/EP72 shape to the qualified CP1
colocated RL lane yields 25.54 GiB parameters, 48.68 GiB FP32 gradients, and
48.62 GiB distributed-Adam state per rank.  Its projected 156.65 GiB phase
peak exceeds the 126.90 GiB safety budget before a usable rollout cache is
available.  TP8 reduces the actor persistent floor to 38.15 GiB/rank and is
therefore a memory requirement of this candidate, not a throughput-only
choice.

The BF16 KV cache, TileLang DSA pair, DeepGEMM runner, and disabled shared
expert fusion follow the Z.ai-linked SGLang H200 serving cell. DeepEP here is
the separate MoE all-to-all transport used by the rollout engine. The cookbook
is an official recommended serving reference, but its verification badge is
hardware-and-command specific: the linked GB300 Low Latency FP8/TRT-LLM cell
is `Verified`, while the H200 cell from which these settings are taken is
currently `Not Verified`. Consequently, our modified colocated H200 command
still requires target-cluster acceptance and is not presented as a production
claim.

`mem_fraction_static` is applied by SGLang to memory visible before rollout
weights load, not to total H200 capacity. In the colocated plan that is the GPU
capacity minus the resident actor floor. The preflight therefore budgets the
static model/KV/KDA pool, cache headroom, and non-static activation/workspace
slack separately.

This remains an analytical candidate, not a qualified production topology,
until a multi-rank expert-routing test (minimum TP4/EP4) and target-H200 peak
measurements pass. Rank-partitioned checkpoint prefetch reduces the analytical
72-replica startup read from a 172.00 TiB per-rank-iteration upper bound toward
the 21.50 TiB physical floor. Shared-filesystem throughput, engine staggering,
and actual bytes read remain target-cluster gates.

## Full export gate

Do not pass `--save-hf` for the full model. The synchronous saver reconstructs
complete TP/EP tensors on every participating rank; distributing only the file
writes does not make that reconstruction scalable. The full profile rejects
this path. Save the authoritative Megatron `torch_dist` checkpoint, then run the
key/chunk-targeted Ray converter as a separate job on a shared filesystem:

```bash
python tools/convert_torch_dist_to_hf_ray.py \
  --input-dir /checkpoints/glm53-flash/iter_0000020 \
  --origin-hf-dir /models/GLM-5.3-Flash \
  --output-dir /checkpoints/glm53-flash-hf \
  --model-name glm5next \
  --concurrency 32 \
  --task-group-bytes 2147483648 \
  --max-file-bytes 5368709120
```

Workers read only their planned DCP keys or physical expert slices and write
their own safetensor shards; there are no NCCL gathers. Stateful q/kv pairs and
both mHC scale triples stay on one worker and incomplete groups fail closed.
Official FP8 export reserves one Ray GPU per worker, stages one bounded group to
that GPU, writes the converted result back to CPU, and then releases the group.
Frozen vision weights are streamed from the pinned HF source because they are
intentionally absent from the language-only training checkpoint. The converter
writes `conversion-manifest.json` with source hashes, task/I/O totals, actors,
and the output-index hash.

The converter was run against the real tiny SFT `torch_dist` artifact (current
DCP `common_state`, no `common.pt`). Its 223 output tensors matched the live
tiny oracle bit-for-bit, the strict changed/frozen-state verifier passed, and
Transformers reloaded `Glm5NextForConditionalGeneration` with 84,361,950
parameters. The tiny lifecycle now performs this comparison after both SFT and
RL and feeds the offline SFT export into RL.

The metadata preflight budgets 583.62 GiB for a language-only BF16/FP32 DCP,
298.80 GiB for the serving FP8 output, and 882.42 GiB while both coexist. A full
Adam resume checkpoint is approximately 3.989 TiB (14 bytes per trainable
language parameter plus frozen language weights), before metadata and
filesystem overhead. Per-worker RSS, GPU peak, aggregate storage bytes, and
throughput remain mandatory measurements on the target cluster.

The resulting 321B export is checked without reloading the full model. The
verifier compares the candidate shard headers against all expected backbone
keys, shapes, and serving dtypes, proves that MTP was removed while the official
FP8 quantization contract and all 36,467 backbone scale tensors were preserved,
reloads only `AutoConfig`, and can optionally stream all 424 frozen tensors
(347 vision plus 77 DSA indexer tensors) one at a time when source weights
already exist locally:

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
  -> authoritative Megatron torch_dist checkpoint
  -> separate key/chunk-targeted Ray HF export
  -> tiny: standard Transformers reload
  -> full: exact header/config gate and target-cluster SGLang load
```

The tiny lane owns key/shape/dtype/value checks at every transition. The full
lane owns config/header checks without the weight payload; frozen vision values
and DSA indexer values are checked only when `--source-weights` is supplied. A rollout tolerance must
not be widened to compensate for a checkpoint, mapping, or synchronization
failure.

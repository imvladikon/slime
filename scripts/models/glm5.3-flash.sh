#!/usr/bin/env bash

# GLM-5.3-Flash model arguments.  The full profile matches
# zai-org/GLM-5.3-Flash; the tiny profile matches the normalized local contract
# checkpoint used by the lifecycle smoke tests.

GLM53_FLASH_PROFILE=${GLM53_FLASH_PROFILE:-full}

case "${GLM53_FLASH_PROFILE}" in
  full)
    GLM53_NUM_LAYERS=45
    GLM53_NUM_DENSE_LAYERS=3
    GLM53_HIDDEN_SIZE=4096
    GLM53_FFN_HIDDEN_SIZE=12288
    GLM53_NUM_ATTENTION_HEADS=64
    GLM53_Q_LORA_RANK=1536
    GLM53_KV_LORA_RANK=512
    GLM53_QK_HEAD_DIM=256
    GLM53_V_HEAD_DIM=256
    GLM53_NUM_EXPERTS=288
    GLM53_EXPERT_TOPK=8
    GLM53_MOE_FFN_HIDDEN_SIZE=2048
    GLM53_INDEX_HEADS=32
    GLM53_INDEX_HEAD_DIM=128
    GLM53_INDEX_TOPK=2048
    GLM53_OPTIMIZED_MOE_ARGS=(--moe-grouped-gemm --moe-permute-fusion)
    ;;
  tiny)
    GLM53_NUM_LAYERS=5
    GLM53_NUM_DENSE_LAYERS=3
    GLM53_HIDDEN_SIZE=256
    GLM53_FFN_HIDDEN_SIZE=256
    GLM53_NUM_ATTENTION_HEADS=4
    GLM53_Q_LORA_RANK=128
    GLM53_KV_LORA_RANK=64
    GLM53_QK_HEAD_DIM=64
    GLM53_V_HEAD_DIM=64
    GLM53_NUM_EXPERTS=8
    GLM53_EXPERT_TOPK=4
    GLM53_MOE_FFN_HIDDEN_SIZE=128
    GLM53_INDEX_HEADS=4
    GLM53_INDEX_HEAD_DIM=64
    GLM53_INDEX_TOPK=64
    GLM53_OPTIMIZED_MOE_ARGS=()
    ;;
  *)
    echo "Unsupported GLM53_FLASH_PROFILE=${GLM53_FLASH_PROFILE}; expected full or tiny" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

GLM53_NUM_MOE_LAYERS=$((GLM53_NUM_LAYERS - GLM53_NUM_DENSE_LAYERS))

MODEL_ARGS=(
  --spec "slime_plugins.models.glm5_next.glm5_next" "get_glm5_next_spec"

  --num-layers "${GLM53_NUM_LAYERS}"
  --hidden-size "${GLM53_HIDDEN_SIZE}"
  --ffn-hidden-size "${GLM53_FFN_HIDDEN_SIZE}"
  --num-attention-heads "${GLM53_NUM_ATTENTION_HEADS}"
  --layernorm-epsilon 1e-5
  --disable-bias-linear
  --swiglu
  --activation-func-clamp-value 10
  --untie-embeddings-and-output-weights
  --vocab-size 154880
  --make-vocab-size-divisible-by 16
  --position-embedding-type rope
  --no-position-embedding
  --normalization RMSNorm
  --qk-layernorm
  --enable-mhc-connections

  --multi-latent-attention
  --q-lora-rank "${GLM53_Q_LORA_RANK}"
  --kv-lora-rank "${GLM53_KV_LORA_RANK}"
  --qk-head-dim "${GLM53_QK_HEAD_DIM}"
  --qk-pos-emb-head-dim 0
  --v-head-dim "${GLM53_V_HEAD_DIM}"
  --kv-channels "${GLM53_QK_HEAD_DIM}"

  --moe-layer-freq "[0]*${GLM53_NUM_DENSE_LAYERS}+[1]*${GLM53_NUM_MOE_LAYERS}"
  --num-experts "${GLM53_NUM_EXPERTS}"
  --moe-ffn-hidden-size "${GLM53_MOE_FFN_HIDDEN_SIZE}"
  --moe-shared-expert-intermediate-size "${GLM53_MOE_FFN_HIDDEN_SIZE}"
  --moe-router-topk "${GLM53_EXPERT_TOPK}"
  --moe-router-score-function sigmoid
  --moe-router-pre-softmax
  --moe-router-enable-expert-bias
  --moe-router-bias-update-rate 0.001
  --moe-router-load-balancing-type none
  --moe-router-topk-scaling-factor 2.5
  --moe-router-num-groups 1
  --moe-router-group-topk 1
  --moe-router-dtype fp32
  --moe-aux-loss-coeff 0
  "${GLM53_OPTIMIZED_MOE_ARGS[@]}"

  --mtp-num-layers 0
  --freeze-indexer
  --weight-checker-skip-prefix visual.
  --enable-experimental
)

# The provider reads the released config and validates the actual KDA/DSA
# schedule, mHC-4 contract, KPool-4 compression, and frozen DSA indexer.  These
# variables are exported for launch-time memory and topology validation.
export GLM53_NUM_LAYERS GLM53_NUM_DENSE_LAYERS GLM53_NUM_MOE_LAYERS
export GLM53_HIDDEN_SIZE GLM53_NUM_EXPERTS GLM53_INDEX_HEADS
export GLM53_INDEX_HEAD_DIM GLM53_INDEX_TOPK

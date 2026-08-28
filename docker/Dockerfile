ARG SGLANG_IMAGE=lmsysorg/sglang:glm-5.3-flash@sha256:e6f5482505e7502f791fe4615ad1fbec118cbbd6b44e98f2479b16b98b985ad6
FROM ${SGLANG_IMAGE} AS sglang

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_REPOSITORY=https://github.com/NVIDIA/Megatron-LM.git
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG SGLANG_REPOSITORY=
ARG SGLANG_COMMIT=
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPGEMM_PACKAGE_VERSION=0.1.5.post3
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_PACKAGE_VERSION=0.1.2
ARG DEEPEP_CUDA_ARCH_LIST=
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=90;100a;103a
ARG TMS_COMMIT=8d30c59ca12a68d9deccbc9c6599076a1218cbc5

ARG TMS_CUDA_MAJOR=

# ======================================== Setup =============================================

WORKDIR /root/

# An exact SGLang source checkout can replace the Python package shipped by the
# base image.  Native extensions and CUDA dependencies continue to come from
# the matching SGLang image, while the editable checkout supplies model and
# scheduler fixes that must move in lockstep with slime.
RUN if [ -n "${SGLANG_REPOSITORY}" ] && [ -n "${SGLANG_COMMIT}" ]; then \
      mkdir -p /root/sglang-source && \
      git -C /root/sglang-source init && \
      git -C /root/sglang-source remote add origin "${SGLANG_REPOSITORY}" && \
      git -C /root/sglang-source fetch --depth 1 origin "${SGLANG_COMMIT}" && \
      git -C /root/sglang-source checkout --detach FETCH_HEAD && \
      test "$(git -C /root/sglang-source rev-parse HEAD)" = "${SGLANG_COMMIT}" && \
      pip install -e /root/sglang-source/python --no-deps; \
    fi
ENV PYTHONPATH=/root/sglang-source/python

# ======================================== Apt dependencies =============================================

RUN apt update
RUN apt install -y nvtop rsync dnsutils prometheus protobuf-compiler

# ====================================== Python dependencies ============================================

# CUDA 12 training uses FA2/FA3.  On CUDA 13, keep the FA4 package supplied by
# SGLang instead of replacing its Blackwell inference kernels with FA2.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip uninstall -y flash-attn-4 flash_attn_4 || true; \
      MAX_JOBS=64 pip -v install flash-attn==2.8.3 --no-build-isolation; \
      git clone https://github.com/Dao-AILab/flash-attention.git && \
      cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
      cd hopper/ && \
      FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
      cd /root/ && rm -rf flash-attention/; \
    fi

RUN pip install flash-linear-attention==0.4.2
# FlashQLA currently requires TileLang 0.1.9, while SGLang 0.5.15.post1 on
# CUDA 13 requires 0.1.11.  Keep the existing CUDA 12 installation and use
# the default FLA backend on CUDA 13 instead of downgrading SGLang's runtime.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip install \
        git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} \
        --no-build-isolation; \
      pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/; \
    fi

# The cu13 TE wheel is built against a newer cuBLAS than SGLang's CUDA 13.0
# toolkit.  Build TE against the base image's libraries on CUDA 13; retain the
# validated wheel path for CUDA 12.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "13" ]; then \
      NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" MAX_JOBS=64 \
        pip -v install --no-build-isolation \
        git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_COMMIT}; \
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.16.1"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone "${MEGATRON_REPOSITORY}" Megatron-LM --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT} && \
    test "$(git rev-parse HEAD)" = "${MEGATRON_COMMIT}" && \
    pip install -e .

# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels
# (DeepGEMM batch-invariant selection and the DeepEP low-latency fork).
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall

# The GLM CUDA 13 purpose image already contains SGLang's exact DeepGEMM
# distribution.  The older custom source publishes version 0.0.0, which cannot
# satisfy the locked SGLang metadata, and its batch-invariant API is unused by
# the DSA qualification lane.  Retain that custom build only for legacy CUDA 12.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "13" ]; then \
      python -c "from importlib.metadata import version; import deep_gemm, sys; assert version('sgl-deep-gemm') == sys.argv[1]; assert hasattr(deep_gemm, 'tf32_hc_prenorm_gemm')" "${DEEPGEMM_PACKAGE_VERSION}"; \
    else \
      git clone https://github.com/zhuzilin/DeepGEMM.git --recursive && \
      cd DeepGEMM && git checkout ${DEEPGEMM_COMMIT} && \
      git submodule update --init --recursive && \
      bash build_sgl_deep_gemm.sh && \
      pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl && \
      cd /root/ && rm -rf DeepGEMM; \
    fi

# Keep the CUDA 13 purpose image's exact sgl-deep-ep distribution. Installing
# the older `deep-ep` fork over it leaves two metadata owners for the same
# import tree, so `pip check` cannot detect a mixed installation. The public
# 0.1.2 wheel contains native sm_90, sm_100, and sm_103 cubins; validate Hopper
# statically because importing DeepEP during an image build requires a GPU.
# Legacy CUDA 12 retains the custom low-latency build.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "13" ]; then \
      DEEPEP_EXTENSION="$(python -c "from importlib.metadata import PackageNotFoundError, distribution, version; import sys; assert version('sgl-deep-ep') == sys.argv[1]; d = distribution('sgl-deep-ep'); files = [p for p in (d.files or ()) if str(p).startswith('deep_ep/') and str(p).endswith('.so')]; assert len(files) == 1, files; print(d.locate_file(files[0]))" "${DEEPEP_PACKAGE_VERSION}")" && \
      python -c "from importlib.metadata import distributions; names = {str(d.metadata['Name']).lower().replace('_', '-') for d in distributions()}; assert 'deep-ep' not in names, 'deep-ep and sgl-deep-ep both own the deep_ep import tree'" && \
      test -f "${DEEPEP_EXTENSION}" && \
      for arch in sm_90 sm_100 sm_103; do \
        /usr/local/cuda/bin/cuobjdump --list-elf "${DEEPEP_EXTENSION}" | grep -Eq "${arch}([.]cubin)?$"; \
      done; \
    else \
      git clone https://github.com/zhuzilin/DeepEP.git /root/DeepEP && \
      cd /root/DeepEP && git checkout ${DEEPEP_COMMIT} && \
      CUDA_ARCHS="${DEEPEP_CUDA_ARCH_LIST:-9.0}" && \
      if [ -d /usr/local/cuda/include/cccl ]; then \
        export CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}"; \
        export NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl ${NVCC_PREPEND_FLAGS:-}"; \
      fi && \
      TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MAX_JOBS=64 python setup.py bdist_wheel && \
      pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
      cd /root/ && rm -rf DeepEP; \
    fi

RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

# Megatron currently requires NumPy 1.x, while SciPy 1.18+ requires NumPy 2.x.
# Pin both packages together so downgrading NumPy does not leave an incompatible SciPy.
RUN pip install "numpy==1.26.4" "scipy==1.17.1"

# Build the router binding from the same pinned SGLang source as the engine.
# A prebuilt router with an older typed GenerateRequest silently drops newer
# native fields such as return_routed_experts, which breaks R3 training.
RUN pip install 'maturin[patchelf]>=1.0,<2.0' && \
    command -v cargo && command -v rustc && command -v protoc && \
    cd /root/sglang-source/sgl-model-gateway/bindings/python && \
    maturin build --release --out /tmp/sglang-router-wheel && \
    pip install --force-reinstall --no-deps /tmp/sglang-router-wheel/sglang_router-*.whl && \
    python -c "from importlib.metadata import version; import sglang_router.sglang_router_rs; assert version('sglang-router') == '0.3.2'" && \
    rm -rf /root/sglang-source/sgl-model-gateway/target /tmp/sglang-router-wheel && \
    pip uninstall -y maturin patchelf

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

ARG ENABLE_MEGATRON_PATCH=1
COPY docker/patch/${PATCH_VERSION}/megatron*.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    if [ "${ENABLE_MEGATRON_PATCH}" = "1" ]; then \
      git update-index --refresh && \
      git apply megatron.patch --3way && \
      if grep -R -n '^<<<<<<< ' .; then \
        echo "Patch failed to apply cleanly. Please resolve conflicts." && \
        exit 1; \
      fi && \
      if [ -f megatron-sglang-aligned.patch ]; then \
        git apply megatron-sglang-aligned.patch --3way && \
        if grep -R -n '^<<<<<<< ' .; then \
          echo "megatron-sglang-aligned.patch failed to apply cleanly. Please resolve conflicts." && \
          exit 1; \
        fi; \
      fi; \
    fi && \
    rm -f megatron*.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own sglang version). should add back later.
ARG ENABLE_SGLANG_PATCH=1
COPY docker/patch/${PATCH_VERSION}/sglang*.patch /sgl-workspace/sglang/
RUN if [ "$ENABLE_SGLANG_PATCH" = "1" ]; then \
  cd /sgl-workspace/sglang && \
  for patch in sglang.patch sglang-top_p.patch sglang-release_hicache.patch sglang-pull_weights.patch sglang-deterministic.patch; do \
    if [ ! -f "$patch" ]; then \
      continue; \
    fi; \
    if ! git apply --check "$patch"; then \
      echo "Patch $patch does not apply to the SGLang source in the base image."; \
      exit 1; \
    fi; \
    git apply "$patch" || exit 1; \
  done && \
  rm sglang*.patch; \
fi

# ====================================== Install main package ============================================

ARG SLIME_REPOSITORY=https://github.com/THUDM/slime.git
ARG SLIME_COMMIT=main
COPY . /tmp/slime-source
RUN if [ "${SLIME_COMMIT}" = "local" ]; then \
      rm -rf /root/slime && \
      mv /tmp/slime-source /root/slime; \
    else \
      rm -rf /tmp/slime-source && \
      git clone "${SLIME_REPOSITORY}" /root/slime && \
      cd /root/slime && \
      git checkout "${SLIME_COMMIT}"; \
    fi && \
    pip install -e /root/slime --no-deps

RUN cd /root/slime/slime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

RUN python -c "import torch, sglang, transformer_engine.pytorch; assert torch.version.cuda and torch.version.cuda.split('.')[0] == '13'; assert torch.__version__.split('+')[0].startswith('2.13.')" && \
    pip check

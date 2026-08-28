import hashlib
from argparse import Namespace
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.types import ParamInfo

from ..megatron_to_hf import conversion_cache_scope, convert_to_hf
from ..sglang import monkey_patch_torch_reductions
from .common import all_gather_params_async, named_params_and_buffers


class HfWeightIteratorDirect:
    def __init__(self, args, model, model_name, quantization_config, transform_ue8m0=True):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.transform_ue8m0 = transform_ue8m0
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    def get_hf_weight_chunks(
        self,
        megatron_local_weights,
        progress_desc: str = "Update weights",
        should_convert_chunk: Callable[[int], bool] | None = None,
        param_info_buckets: Sequence[Sequence[ParamInfo]] | None = None,
    ):
        rank = dist.get_rank()
        param_info_buckets = (
            self.megatron_local_param_info_buckets if param_info_buckets is None else param_info_buckets
        )

        with conversion_cache_scope():
            for chunk_idx, megatron_local_param_infos in enumerate(
                tqdm(param_info_buckets, disable=rank != 0, desc=progress_desc)
            ):
                megatron_full_params = _get_megatron_full_params(
                    megatron_local_param_infos, megatron_local_weights
                )
                if should_convert_chunk is None or should_convert_chunk(chunk_idx):
                    hf_named_tensors = self._convert_to_hf_named_tensors(
                        megatron_full_params, megatron_local_param_infos
                    )
                else:
                    hf_named_tensors = []
                try:
                    yield hf_named_tensors
                finally:
                    del hf_named_tensors, megatron_full_params

    def _convert_to_hf_named_tensors(
        self,
        megatron_full_params: Sequence[torch.Tensor],
        param_infos: Sequence[ParamInfo],
    ):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    info.name,
                    param,
                    self.quantization_config,
                    transform_ue8m0=self.transform_ue8m0,
                )
            )
        return hf_named_tensors


def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=accelerator.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=accelerator.current_device()))
    accelerator.synchronize()

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    param_infos = _get_megatron_local_param_infos(args, model)
    return pack_param_info_buckets(param_infos, args.update_weight_buffer_size)


def pack_param_info_buckets(
    param_infos: Sequence[ParamInfo],
    update_weight_buffer_size: int,
) -> list[list[ParamInfo]]:
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Logical metadata is identical across ranks; src_rank remains
    the global physical owner and may differ for replicated dense parameters.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos: dict[str, ParamInfo] = {}
    local_parameters: dict[str, torch.Tensor] = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        previous = local_parameters.get(name)
        if previous is not None and previous is not param:
            raise RuntimeError(f"Distinct local parameters canonicalize to the same name: {name}")
        local_parameters[name] = param
        info = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )
        _merge_param_info(param_infos, info, reject_distinct_owner=False)

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                _merge_param_info(param_infos, info, reject_distinct_owner=False)

    if ep_size > 1:
        local_expert_infos = {name: info for name, info in param_infos.items() if ".experts." in name}
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, local_expert_infos),
            object_list=param_infos_list,
            group=mpu.get_expert_model_parallel_group(),
        )
        for _src_rank, infos in param_infos_list:
            for info in infos.values():
                _merge_param_info(param_infos, info, reject_distinct_owner=True)

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    _validate_param_infos_consistent(param_infos)

    return param_infos


def _param_info_metadata(info: ParamInfo) -> tuple:
    attrs = tuple(
        (str(key), type(value).__qualname__, repr(value))
        for key, value in sorted(info.attrs.items(), key=lambda item: str(item[0]))
    )
    return info.name, str(info.dtype), tuple(info.shape), info.size, attrs


def _merge_param_info(
    param_infos: dict[str, ParamInfo],
    info: ParamInfo,
    *,
    reject_distinct_owner: bool,
) -> None:
    """Merge logical metadata while preserving global physical source ranks."""
    old_info = param_infos.get(info.name)
    if old_info is None:
        param_infos[info.name] = info
        return
    if _param_info_metadata(old_info) != _param_info_metadata(info):
        raise RuntimeError(
            f"Parameter metadata mismatch for {info.name}: "
            f"owner {old_info.src_rank} != owner {info.src_rank}"
        )
    if reject_distinct_owner and old_info.src_rank != info.src_rank:
        raise RuntimeError(
            f"Routed expert {info.name} has multiple physical owners: "
            f"{old_info.src_rank} and {info.src_rank}"
        )
    if info.src_rank < old_info.src_rank:
        param_infos[info.name] = info


def _param_info_fingerprint(param_infos: Sequence[ParamInfo]) -> tuple[int, str]:
    """Return constant-size metadata for the cross-world consistency check."""
    digest = hashlib.sha256()
    for info in param_infos:
        for value in _param_info_metadata(info):
            value = repr(value)
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return len(param_infos), digest.hexdigest()


def _validate_param_infos_consistent(param_infos: Sequence[ParamInfo]) -> None:
    """Check global metadata without replicating every ParamInfo on every rank."""
    local = _param_info_fingerprint(param_infos)
    fingerprints = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=local,
        object_list=fingerprints,
        group=get_gloo_group(),
    )
    if any(fingerprint != local for fingerprint in fingerprints):
        raise RuntimeError(f"Parameter metadata mismatch across ranks: local={local}, gathered={fingerprints}")

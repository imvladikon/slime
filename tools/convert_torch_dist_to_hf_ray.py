r"""Ray-based Megatron torch_dist to HuggingFace conversion.

This is a high-throughput converter for large Kimi/DeepSeek-style checkpoints.
The core idea is to parallelize conversion work using distributed processes on Ray.
Conversion is an embarrassingly parallel problem, so it exhibits strong scaling with the number of nodes.

System flow:

+-------------------+
| prepare_runtime   |
|-------------------|
| init Ray          |
| validate output   |
| stage HF config   |
| create staging    |
+---------+---------+
          |
          v
+------------------------+
| read_metadata_and_plan |
|------------------------|
| read DCP metadata/args |
| read DCP metadata      |
| build task plan        |
| publish metadata ref   |
+-----------+------------+
            |
            v
+-------------------------------------------------------------+
| dispatch_conversion_tasks                                   |
|-------------------------------------------------------------|
| create pinned Ray actors                                    |
| submit/refill tasks up to --concurrency                     |
|                                                             |
|  +------------------+   +------------------+   +---------+  |
|  | actor 0          |   | actor 1          |   | actor N |  |
|  |------------------|   |------------------|   |---------|  |
|  | DCP load         |   | DCP load         |   | ...     |  |
|  | convert to HF    |   | convert to HF    |   | ...     |  |
|  | optional quant   |   | optional quant   |   | ...     |  |
|  | write shards     |   | write shards     |   | ...     |  |
|  | return manifest  |   | return manifest  |   | ...     |  |
|  +------------------+   +------------------+   +---------+  |
+-----------------------------+-------------------------------+
                              |
                              v
+----------------------------+
| finalize_conversion_output |
|----------------------------|
| merge manifests            |
| assign final shard names   |
| publish shards/assets      |
| write final index          |
+----------------------------+

Examples:

The converter expects filesystem paths. Stage remote checkpoints and HF assets
onto local or shared storage before running it.

INPUT_DIR=/mnt/checkpoints/kimi-k26/iter_0000400
ORIGIN_HF_DIR=/mnt/hf/Kimi-K2.6-fp8-configs-only

# 8-layer smoke test
SOURCE_KEY_REGEX="^language_model\.decoder\.layers\.([0-7])\."
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /tmp/$USER/$RUN_ID/hf_ray_l0_7 --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --source-key-regex $SOURCE_KEY_REGEX --max-file-bytes 21474836480 --concurrency 16 --progress-interval-seconds 10 -f

# Full conversion on 1 node
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /mnt/outputs/kimi-ray/$RUN_ID/ray1_hf --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --max-file-bytes 21474836480 --concurrency 16 --progress-interval-seconds 10 -f

# Full conversion on 4 nodes
python tools/convert_torch_dist_to_hf_ray.py --input-dir $INPUT_DIR --output-dir /mnt/outputs/kimi-ray/$RUN_ID/ray4_hf --origin-hf-dir $ORIGIN_HF_DIR --model-name kimi_k25 --max-file-bytes 21474836480 --concurrency 64 --progress-interval-seconds 10 -f
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import ray
import safetensors.torch
import torch
import torch.distributed.checkpoint as dist_cp
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import MetadataIndex
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
from torch.distributed.checkpoint.utils import _create_file_view
from torch.futures import Future
from tqdm.auto import tqdm
from transformers import AutoConfig
from typing_extensions import override

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slime.backends.megatron_utils import megatron_to_hf as m2hf

DEFAULT_DIRECT_MOE_GROUP_SIZE = 2 * 1024**3


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name, name):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


@dataclass(frozen=True)
class Args:
    input_dir: str
    output_dir: str
    origin_hf_dir: str | None
    model_name: str | None
    force: bool
    max_file_bytes: int
    concurrency: int | None
    task_group_bytes: int
    source_key_regex: str | None
    dry_run_plan: bool
    progress: bool
    progress_interval_seconds: float


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    keys: tuple[str, ...]
    estimated_source_bytes: int
    moe_blocks: tuple[MoeBlockSpec, ...] = ()


@dataclass(frozen=True)
class MoeBlockSpec:
    source_key: str
    relative_path: str
    storage_indices: tuple[MetadataIndex, ...]
    layer_idx: int
    linear_name: str
    hf_prefix: str


@dataclass(frozen=True)
class PreparedTensorGroup:
    source_name: str
    tensors: tuple[tuple[str, torch.Tensor], ...]


@dataclass(frozen=True)
class TaskLoadStats:
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PreparedTaskTensors:
    groups: tuple[PreparedTensorGroup, ...]
    load_stats: TaskLoadStats


@dataclass(frozen=True)
class ShardManifest:
    temp_filename: str
    final_filename: str | None
    weight_keys: tuple[str, ...]
    bytes: int
    tensors: int


@dataclass(frozen=True)
class TaskResult:
    task_id: int
    actor_id: int
    node: str
    pid: int
    shards: tuple[ShardManifest, ...]
    source_bytes: int
    output_bytes: int
    weights: int
    source_keys: tuple[str, ...]
    dcp_read_items: int
    dcp_files: int
    dcp_storage_bytes: int
    cuda_device_id: int | None
    ray_node_id: str


@dataclass(frozen=True)
class DcpLoadResult:
    state_dict: dict[str, torch.Tensor]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class DirectMoeLoadResult:
    tensor_groups: tuple[PreparedTensorGroup, ...]
    read_items: int
    files: int
    storage_bytes: int


@dataclass(frozen=True)
class PlannedShard:
    temp_filename: str
    final_filename: str
    weight_keys: tuple[str, ...]
    bytes: int


class ProgressReporter:
    def __init__(
        self,
        tasks: list[TaskSpec],
        enabled: bool,
        interval_seconds: float,
        stream: Any = sys.stderr,
    ) -> None:
        self.enabled = enabled
        self.interval_seconds = max(interval_seconds, 0.1)
        self.total_tasks = len(tasks)
        self.total_bytes = sum(task.estimated_source_bytes for task in tasks)
        self.completed_tasks = 0
        self.last_refresh_time = 0.0
        self.progress = tqdm(
            total=self.total_bytes or self.total_tasks,
            desc="Converting",
            unit="B" if self.total_bytes else "task",
            unit_scale=bool(self.total_bytes),
            unit_divisor=1000,
            mininterval=self.interval_seconds,
            disable=not enabled,
            file=stream,
        )
        self._set_postfix()

    def complete(self, result: TaskResult) -> None:
        self.completed_tasks += 1
        self.progress.update(result.source_bytes if self.total_bytes else 1)
        self._set_postfix()

    def tick(self) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_refresh_time >= self.interval_seconds:
            self.progress.refresh()
            self.last_refresh_time = now

    def finish(self) -> None:
        self._set_postfix()
        self.progress.close()

    def _set_postfix(self) -> None:
        if not self.enabled:
            return
        self.progress.set_postfix_str(f"tasks={self.completed_tasks}/{self.total_tasks}", refresh=False)


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = make_storage_meta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class ChunkedStateDictLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    def __init__(self, keys_to_load: set[str]):
        super().__init__()
        self.keys_to_load = keys_to_load

    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        if metadata is None:
            raise ValueError("DCP metadata is required")
        for key, value in metadata.state_dict_metadata.items():
            if key not in self.keys_to_load:
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype)  # type: ignore[assignment]
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


class MeteredStorageReader(WrappedStorageReader):
    def __init__(self, path: str):
        super().__init__(path)
        self.read_items = 0
        self.files = 0
        self.storage_bytes = 0

    @override
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        self.read_items, self.files, self.storage_bytes = compute_dcp_load_accounting(self.storage_data, plan)
        per_file: dict[str, list[Any]] = {}
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            per_file.setdefault(item_md.relative_path, []).append(read_item)

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        planner.load_bytes(req, read_bytes)
                        continue

                    seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
                    seekable.seek(0)
                    tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()
                    if target_tensor.size() != tensor.size():
                        raise AssertionError(
                            f"DCP tensor size mismatch for {req.storage_index}: "
                            f"{target_tensor.size()} vs {tensor.size()}"
                        )
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

        fut: Future[None] = Future()
        fut.set_result(None)
        return fut


def make_storage_meta():
    storage_meta = getattr(dist_cp, "StorageMeta", None)
    if storage_meta is not None:
        return storage_meta()
    return dist_cp.metadata.StorageMeta()


def compute_dcp_load_accounting(storage_data: dict[Any, Any], plan: LoadPlan) -> tuple[int, int, int]:
    files = set()
    storage_bytes = 0
    for read_item in plan.items:
        item_md = storage_data[read_item.storage_index]
        files.add(item_md.relative_path)
        storage_bytes += int(item_md.length)
    return len(plan.items), len(files), storage_bytes


def prepare_cached_metadata_for_reader(
    metadata: dist_cp.metadata.Metadata,
    storage_reader: WrappedStorageReader,
) -> dist_cp.metadata.Metadata:
    if getattr(metadata, "storage_meta", None) is None:
        metadata.storage_meta = make_storage_meta()
    metadata.storage_meta.load_id = storage_reader.load_id
    if metadata.planner_data is None:
        metadata.planner_data = {}
    return metadata


def load_tensor_chunk(
    input_dir: str,
    keys_to_load: set[str],
    metadata: dist_cp.metadata.Metadata,
) -> DcpLoadResult:
    state_dict: dict[str, torch.Tensor] = {}
    storage_reader = MeteredStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    planner = ChunkedStateDictLoadPlanner(keys_to_load)
    planner.set_up_planner(state_dict, metadata, is_coordinator=True)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    local_plan = planner.create_local_plan()
    local_plan = storage_reader.prepare_local_plan(local_plan)
    global_plan = planner.create_global_plan([local_plan])
    global_plan = storage_reader.prepare_global_plan(global_plan)
    final_local_plan = planner.finish_plan(global_plan[0])
    storage_reader.read_data(final_local_plan, planner).wait()
    return DcpLoadResult(
        state_dict=state_dict,
        read_items=storage_reader.read_items,
        files=storage_reader.files,
        storage_bytes=storage_reader.storage_bytes,
    )


def get_expert_param(args: Any, name: str, param: torch.Tensor):
    if ".experts." not in name:
        yield name, param
        return

    num_experts = args.num_experts
    match = re.search(r"mlp.experts\.(.+)\.weight(\d+)", name)
    if not match:
        if param.shape[0] != num_experts:
            raise AssertionError(f"Expected {num_experts} experts for {name}, got {param.shape}")
        for expert_id in range(num_experts):
            expert_name = name.replace(".experts.experts.", ".experts.") + str(expert_id)
            yield expert_name, param[expert_id]
    else:
        yield name, param


def get_layer_param(args: Any, name: str, param: torch.Tensor):
    if ".layers." not in name:
        yield name, param
        return

    num_layers = args.num_layers
    match = re.search(r"\.layers\.(\d+)\.", name)
    if not match:
        if param.shape[0] != num_layers:
            raise AssertionError(f"Expected {num_layers} layers for {name}, got {param.shape}")
        for layer_id in range(num_layers):
            layer_name = name.replace(".layers.", f".layers.{layer_id}.")
            yield from get_expert_param(args, layer_name, param[layer_id])
    else:
        yield from get_expert_param(args, name, param)


def get_named_params(args: Any, state_dict: dict[str, torch.Tensor]):
    for name, param in state_dict.items():
        yield from get_layer_param(args, f"module.module.{name}", param)


_MOE_EXPERT_KEY_RE = re.compile(
    r"^(?P<prefix>language_model\.)?decoder\.layers\.(?P<layer>\d+)\."
    r"mlp\.experts\.experts\.(?P<linear>linear_fc[12])\.weight$"
)


def parse_moe_expert_key(source_key: str) -> tuple[int, str, str] | None:
    match = _MOE_EXPERT_KEY_RE.match(source_key)
    if not match:
        return None
    hf_prefix = "language_model." if match.group("prefix") else ""
    return int(match.group("layer")), match.group("linear"), hf_prefix


def is_supported_moe_read_item(read_item: Any, source_key: str, tensor_size: torch.Size) -> bool:
    parsed = parse_moe_expert_key(source_key)
    if parsed is None or len(tensor_size) != 3:
        return False
    _, linear_name, _ = parsed

    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    if len(offsets) != 3 or len(lengths) != 3:
        return False

    expert_offset, ffn_offset, hidden_offset = offsets
    expert_count, ffn_length, hidden_length = lengths
    num_experts, ffn_size, hidden_size = (int(value) for value in tensor_size)
    if expert_offset < 0 or expert_offset + expert_count > num_experts:
        return False
    if hidden_offset != 0 or hidden_length != hidden_size:
        return False
    if linear_name == "linear_fc2":
        return ffn_offset == 0 and ffn_length == ffn_size
    if ffn_size % 2 != 0:
        return False
    half_ffn = ffn_size // 2
    return (ffn_offset == 0 and ffn_length == ffn_size) or (ffn_offset in {0, half_ffn} and ffn_length == half_ffn)


def create_full_tensor_read_items(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[Any]:
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{source_key} is not tensor metadata")
    return create_read_items_for_chunk_list(source_key, md, list(md.chunks))


def create_direct_moe_read_items(block: MoeBlockSpec, md: dist_cp.metadata.TensorStorageMetadata) -> list[ReadItem]:
    chunk_by_index = {
        MetadataIndex(block.source_key, chunk.offsets, idx): chunk for idx, chunk in enumerate(md.chunks)
    }
    read_items: list[ReadItem] = []
    for storage_index in block.storage_indices:
        chunk = chunk_by_index[storage_index]
        read_items.append(
            ReadItem(
                type=LoadItemType.TENSOR,
                dest_index=MetadataIndex(block.source_key),
                dest_offsets=chunk.offsets,
                storage_index=storage_index,
                storage_offsets=torch.Size([0 for _ in chunk.offsets]),
                lengths=chunk.sizes,
            )
        )
    read_items.sort(key=lambda item: tuple(item.storage_index.offset))
    return read_items


def create_moe_block_specs(source_key: str, metadata: dist_cp.metadata.Metadata) -> list[MoeBlockSpec] | None:
    parsed = parse_moe_expert_key(source_key)
    if parsed is None:
        return None
    layer_idx, linear_name, hf_prefix = parsed
    md = metadata.state_dict_metadata[source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        return None
    read_items = create_full_tensor_read_items(source_key, metadata)
    if not read_items or any(not is_supported_moe_read_item(item, source_key, md.size) for item in read_items):
        return None
    if metadata.storage_data is None:
        return None

    by_file: dict[str, list[MetadataIndex]] = {}
    for read_item in read_items:
        item_md = metadata.storage_data[read_item.storage_index]
        by_file.setdefault(item_md.relative_path, []).append(read_item.storage_index)

    return [
        MoeBlockSpec(
            source_key=source_key,
            relative_path=relative_path,
            storage_indices=tuple(sorted(indices, key=lambda index: tuple(index.offset))),
            layer_idx=layer_idx,
            linear_name=linear_name,
            hf_prefix=hf_prefix,
        )
        for relative_path, indices in sorted(by_file.items())
    ]


def expert_source_name(source_key: str, expert_id: int) -> str:
    return "module.module." + source_key.replace(".experts.experts.", ".experts.") + str(expert_id)


def hf_expert_weight_name(
    hf_prefix: str,
    layer_idx: int,
    expert_id: int,
    projection: str,
    model_name: str,
) -> str:
    normalized_model_name = model_name.lower().replace("_", "").replace("-", "")
    if "glm5next" in normalized_model_name:
        return f"model.language_model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"
    return f"{hf_prefix}model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"


def contiguous_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def converted_moe_tensors_from_chunk(
    source_key: str,
    layer_idx: int,
    linear_name: str,
    hf_prefix: str,
    read_item: Any,
    tensor: torch.Tensor,
    tensor_size: torch.Size,
    model_name: str,
) -> list[PreparedTensorGroup]:
    offsets = tuple(int(value) for value in read_item.storage_index.offset)
    lengths = tuple(int(value) for value in read_item.lengths)
    expert_offset = offsets[0]
    expert_count = lengths[0]
    if tensor.shape[0] != expert_count:
        raise AssertionError(f"Expected {expert_count} experts in {source_key} chunk, got {tensor.shape}")

    groups: list[PreparedTensorGroup] = []
    if linear_name == "linear_fc2":
        for local_expert_idx in range(expert_count):
            expert_id = expert_offset + local_expert_idx
            groups.append(
                PreparedTensorGroup(
                    source_name=expert_source_name(source_key, expert_id),
                    tensors=(
                        (
                            hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "down_proj", model_name),
                            contiguous_if_needed(tensor[local_expert_idx]),
                        ),
                    ),
                )
            )
        return groups

    half_ffn = int(tensor_size[1]) // 2
    second_dim_offset = offsets[1]
    second_dim_length = lengths[1]
    for local_expert_idx in range(expert_count):
        expert_id = expert_offset + local_expert_idx
        expert_tensor = tensor[local_expert_idx]
        if second_dim_offset == 0 and second_dim_length == int(tensor_size[1]):
            gate_weight, up_weight = expert_tensor.chunk(2, dim=0)
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj", model_name),
                    contiguous_if_needed(gate_weight),
                ),
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj", model_name),
                    contiguous_if_needed(up_weight),
                ),
            )
        elif second_dim_offset == 0 and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "gate_proj", model_name),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        elif second_dim_offset == half_ffn and second_dim_length == half_ffn:
            named_tensors = (
                (
                    hf_expert_weight_name(hf_prefix, layer_idx, expert_id, "up_proj", model_name),
                    contiguous_if_needed(expert_tensor),
                ),
            )
        else:
            raise AssertionError(
                f"Unsupported {linear_name} chunk for {source_key}: "
                f"offsets={offsets}, lengths={lengths}, size={tuple(tensor_size)}"
            )
        groups.append(
            PreparedTensorGroup(source_name=expert_source_name(source_key, expert_id), tensors=named_tensors)
        )
    return groups


def load_moe_block_direct(
    input_dir: str,
    block: MoeBlockSpec,
    metadata: dist_cp.metadata.Metadata,
    model_name: str,
) -> DirectMoeLoadResult:
    md = metadata.state_dict_metadata[block.source_key]
    if not isinstance(md, dist_cp.metadata.TensorStorageMetadata):
        raise TypeError(f"{block.source_key} is not tensor metadata")

    storage_reader = MeteredStorageReader(input_dir)
    metadata = prepare_cached_metadata_for_reader(metadata, storage_reader)
    storage_reader.set_up_storage_reader(metadata, is_coordinator=True)
    read_items = create_direct_moe_read_items(block, md)
    read_count, file_count, storage_bytes = compute_dcp_load_accounting(
        storage_reader.storage_data, LoadPlan(read_items)
    )

    tensor_groups: list[PreparedTensorGroup] = []
    new_path = storage_reader.fs.concat_path(storage_reader.path, block.relative_path)
    with storage_reader.fs.create_stream(new_path, "rb") as stream:
        for read_item in read_items:
            item_md = storage_reader.storage_data[read_item.storage_index]
            file_slice = cast(io.IOBase, _create_file_view(stream, item_md.offset, item_md.length))
            transform_from = storage_reader.transforms.transform_load_stream(
                read_item,
                item_md.transform_descriptors or (),
                file_slice,
            )
            seekable = transform_from if transform_from.seekable() else io.BytesIO(transform_from.read(-1))
            seekable.seek(0)
            tensor = cast(torch.Tensor, torch.load(seekable, map_location="cpu", weights_only=True))
            tensor = narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
            tensor_groups.extend(
                converted_moe_tensors_from_chunk(
                    block.source_key,
                    block.layer_idx,
                    block.linear_name,
                    block.hf_prefix,
                    read_item,
                    tensor,
                    md.size,
                    model_name,
                )
            )

    return DirectMoeLoadResult(tuple(tensor_groups), read_count, file_count, storage_bytes)


def tensor_metadata_from_checkpoint_metadata(
    metadata: dist_cp.metadata.Metadata,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    tensor_metadata = {}
    for key, value in metadata.state_dict_metadata.items():
        if "optimizer" in key or "_state" in key:
            continue
        if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
            tensor_metadata[key] = (value.size, value.properties.dtype)
    return tensor_metadata


def tensor_nbytes(shape: torch.Size, dtype: torch.dtype) -> int:
    element_bits = torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
    return shape.numel() * (element_bits // 8)


def filter_tensor_metadata(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    source_key_regex: str | None,
) -> dict[str, tuple[torch.Size, torch.dtype]]:
    if not source_key_regex:
        return tensor_metadata
    pattern = re.compile(source_key_regex)
    return {key: value for key, value in tensor_metadata.items() if pattern.search(key)}


def _mla_pair_group(key: str) -> str:
    return key.replace("self_attention.linear_q_down_proj.weight", "self_attention.MLA_A_PAIR.weight").replace(
        "self_attention.linear_kv_down_proj.weight",
        "self_attention.MLA_A_PAIR.weight",
    )


_MHC_COMPONENT_RE = re.compile(
    r"^(?P<prefix>(?:language_model\.)?decoder\.layers\.\d+\.)"
    r"(?P<site>self_attention_hyper_connection|mlp_hyper_connection)\."
    r"alpha_(?:pre|post|res)$"
)


def _atomic_conversion_group(key: str, q_lora_rank: int | None) -> str:
    """Keep stateful Megatron-to-HF conversions inside one Ray task."""
    if q_lora_rank is not None:
        paired = _mla_pair_group(key)
        if paired != key:
            return paired
    mhc = _MHC_COMPONENT_RE.fullmatch(key)
    if mhc:
        return f"{mhc.group('prefix')}{mhc.group('site')}.MHC_SCALE"
    return key


def _validate_atomic_conversion_groups(grouped: dict[str, list[str]]) -> None:
    for group, keys in grouped.items():
        if group.endswith("MLA_A_PAIR.weight"):
            required = {
                group.replace("MLA_A_PAIR.weight", "linear_q_down_proj.weight"),
                group.replace("MLA_A_PAIR.weight", "linear_kv_down_proj.weight"),
            }
            if set(keys) != required:
                raise ValueError(f"Incomplete MLA projection pair {group}: {sorted(keys)}")
        elif group.endswith(".MHC_SCALE"):
            prefix = group.removesuffix("MHC_SCALE")
            required = {prefix + f"alpha_{name}" for name in ("pre", "post", "res")}
            if set(keys) != required:
                raise ValueError(f"Incomplete mHC scale group {group}: {sorted(keys)}")


def group_small_tasks(
    atomic_tasks: list[tuple[int, tuple[str, ...]]],
    task_group_bytes: int,
) -> list[tuple[int, tuple[str, ...]]]:
    if task_group_bytes <= 0:
        return atomic_tasks

    large_tasks = [task for task in atomic_tasks if task[0] >= task_group_bytes]
    small_tasks = [task for task in atomic_tasks if task[0] < task_group_bytes]
    small_tasks.sort(key=lambda item: (-item[0], item[1]))
    grouped_tasks: list[tuple[int, tuple[str, ...]]] = []
    current_keys: list[str] = []
    current_bytes = 0
    for estimated_bytes, keys in small_tasks:
        if current_keys and current_bytes + estimated_bytes > task_group_bytes:
            grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
            current_keys = []
            current_bytes = 0
        current_keys.extend(keys)
        current_bytes += estimated_bytes
    if current_keys:
        grouped_tasks.append((current_bytes, tuple(sorted(current_keys))))
    return large_tasks + grouped_tasks


def plan_whole_source_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    grouped: dict[str, list[str]] = {}
    for key in tensor_metadata:
        group = _atomic_conversion_group(key, q_lora_rank)
        grouped.setdefault(group, []).append(key)
    _validate_atomic_conversion_groups(grouped)

    atomic_tasks = []
    for keys in grouped.values():
        sorted_keys = tuple(sorted(keys))
        estimated_bytes = sum(tensor_nbytes(tensor_metadata[key][0], tensor_metadata[key][1]) for key in sorted_keys)
        atomic_tasks.append((estimated_bytes, sorted_keys))
    raw_tasks = group_small_tasks(atomic_tasks, task_group_bytes)
    raw_tasks.sort(key=lambda item: (-item[0], item[1]))
    return [TaskSpec(idx, keys, estimated_bytes) for idx, (estimated_bytes, keys) in enumerate(raw_tasks)]


def collect_moe_blocks_by_file(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
) -> tuple[list[tuple[int, MoeBlockSpec]], dict[str, tuple[torch.Size, torch.dtype]]]:
    moe_blocks: list[tuple[int, MoeBlockSpec]] = []
    whole_source_metadata = dict(tensor_metadata)
    if metadata.storage_data is None:
        return moe_blocks, whole_source_metadata

    for source_key in sorted(tensor_metadata):
        block_specs = create_moe_block_specs(source_key, metadata)
        if block_specs is None:
            continue
        for block in block_specs:
            estimated_bytes = sum(int(metadata.storage_data[index].length) for index in block.storage_indices)
            moe_blocks.append((estimated_bytes, block))
        del whole_source_metadata[source_key]
    return moe_blocks, whole_source_metadata


def group_moe_block_tasks(moe_blocks: list[tuple[int, MoeBlockSpec]], task_group_bytes: int) -> list[TaskSpec]:
    if not moe_blocks:
        return []
    target_bytes = task_group_bytes or DEFAULT_DIRECT_MOE_GROUP_SIZE
    moe_blocks.sort(
        key=lambda item: (
            item[1].source_key,
            item[1].relative_path,
            tuple(item[1].storage_indices[0].offset) if item[1].storage_indices else (),
        )
    )

    tasks: list[TaskSpec] = []
    current_blocks: list[MoeBlockSpec] = []
    current_bytes = 0

    def flush_current() -> None:
        nonlocal current_blocks, current_bytes
        if not current_blocks:
            return
        tasks.append(
            TaskSpec(
                task_id=-1,
                keys=tuple(sorted({block.source_key for block in current_blocks})),
                estimated_source_bytes=current_bytes,
                moe_blocks=tuple(current_blocks),
            )
        )
        current_blocks = []
        current_bytes = 0

    for estimated_bytes, block in moe_blocks:
        if current_blocks and current_bytes + estimated_bytes > target_bytes:
            flush_current()
        current_blocks.append(block)
        current_bytes += estimated_bytes
    flush_current()
    return tasks


def plan_conversion_tasks(
    tensor_metadata: dict[str, tuple[torch.Size, torch.dtype]],
    metadata: dist_cp.metadata.Metadata,
    q_lora_rank: int | None,
    task_group_bytes: int,
) -> list[TaskSpec]:
    moe_blocks, whole_source_metadata = collect_moe_blocks_by_file(tensor_metadata, metadata)
    tasks = group_moe_block_tasks(moe_blocks, task_group_bytes)
    tasks.extend(
        TaskSpec(-1, task.keys, task.estimated_source_bytes)
        for task in plan_whole_source_tasks(whole_source_metadata, q_lora_rank, task_group_bytes)
    )
    tasks.sort(
        key=lambda task: (
            -task.estimated_source_bytes,
            task.moe_blocks[0].source_key if task.moe_blocks else task.keys[0],
            task.moe_blocks[0].relative_path if task.moe_blocks else "",
            task.keys,
        )
    )
    return [TaskSpec(idx, task.keys, task.estimated_source_bytes, task.moe_blocks) for idx, task in enumerate(tasks)]


def summarize_plan(tasks: list[TaskSpec], model_name: str, concurrency: int, output_dir: str) -> dict[str, Any]:
    source_bytes = [task.estimated_source_bytes for task in tasks]
    direct_moe_tasks = [task for task in tasks if task.moe_blocks]
    return {
        "model_name": model_name,
        "concurrency": concurrency,
        "output_dir": output_dir,
        "tasks": len(tasks),
        "source_keys": sum(len(task.keys) for task in tasks),
        "direct_moe_tasks": len(direct_moe_tasks),
        "direct_moe_blocks": sum(len(task.moe_blocks) for task in direct_moe_tasks),
        "estimated_source_bytes": sum(source_bytes),
        "largest_task_bytes": max(source_bytes, default=0),
        "smallest_task_bytes": min(source_bytes, default=0),
        "top_tasks": [
            {
                "task_id": task.task_id,
                "keys": task.keys,
                "estimated_source_bytes": task.estimated_source_bytes,
                "moe_blocks": len(task.moe_blocks),
            }
            for task in tasks[:10]
        ],
    }


def load_hf_config(origin_hf_dir: str | None) -> Any | None:
    if origin_hf_dir is None:
        return None
    return AutoConfig.from_pretrained(origin_hf_dir, trust_remote_code=True)


def load_quantization_config(hf_config: Any | None) -> dict[str, Any] | None:
    if hf_config is None:
        return None
    quantization_config = getattr(hf_config, "quantization_config", None)
    if quantization_config is not None:
        return dict(quantization_config)
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        nested = getattr(text_config, "quantization_config", None)
        if nested is not None:
            return dict(nested)
    return None


def get_hf_vocab_size(hf_config: Any | None) -> int | None:
    if hf_config is None:
        return None
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    vocab_size = getattr(hf_config, "vocab_size", None)
    return int(vocab_size) if vocab_size is not None else None


def copy_assets(origin_hf_dir: str | None, output_dir: str) -> None:
    if origin_hf_dir is None:
        return
    for filename in os.listdir(origin_hf_dir):
        if filename == "model.safetensors.index.json" or filename.endswith(".safetensors"):
            continue
        src = os.path.join(origin_hf_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, filename))


def is_glm5_next_model(model_name: str) -> bool:
    return "glm5next" in model_name.lower().replace("_", "").replace("-", "")


def source_weight_map(origin_hf_dir: str) -> dict[str, str]:
    index_path = os.path.join(origin_hf_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as handle:
            weight_map = json.load(handle).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid Hugging Face weight index: {index_path}")
        return {str(key): str(filename) for key, filename in weight_map.items()}

    single = os.path.join(origin_hf_dir, "model.safetensors")
    if os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: "model.safetensors" for key in handle.keys()}
    raise FileNotFoundError(f"No safetensors weights found under {origin_hf_dir}")


def copy_frozen_source_tensors(
    origin_hf_dir: str | None,
    staging_dir: str,
    task_id: int,
    max_file_bytes: int,
    *,
    prefix: str,
) -> TaskResult:
    """Stream frozen tensors that are intentionally absent from Megatron state."""
    if origin_hf_dir is None:
        raise ValueError(f"--origin-hf-dir is required to preserve frozen {prefix} tensors")
    from safetensors import safe_open

    weight_map = source_weight_map(origin_hf_dir)
    selected = {key: filename for key, filename in weight_map.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"The source checkpoint contains no frozen tensors with prefix {prefix}")

    keys_by_file: dict[str, list[str]] = {}
    for key, filename in selected.items():
        keys_by_file.setdefault(filename, []).append(key)

    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    shards: list[ShardManifest] = []
    total_size = 0
    for filename in sorted(keys_by_file):
        source_path = os.path.join(origin_hf_dir, filename)
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            for key in sorted(keys_by_file[filename]):
                tensor = handle.get_tensor(key)
                tensor_size = tensor.numel() * tensor.element_size()
                if current_tensors and current_size + tensor_size > max_file_bytes:
                    shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))
                    shard_idx += 1
                    current_tensors = {}
                    current_size = 0
                current_tensors[key] = tensor.contiguous()
                current_size += tensor_size
                total_size += tensor_size
    if current_tensors:
        shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))

    return TaskResult(
        task_id=task_id,
        actor_id=-1,
        node=socket.gethostname(),
        pid=os.getpid(),
        shards=tuple(shards),
        source_bytes=total_size,
        output_bytes=total_size,
        weights=len(selected),
        source_keys=tuple(sorted(selected)),
        dcp_read_items=0,
        dcp_files=len(keys_by_file),
        dcp_storage_bytes=0,
        cuda_device_id=None,
        ray_node_id="driver",
    )


def _flush_shard(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    tensors: dict[str, torch.Tensor],
    current_size: int,
) -> ShardManifest:
    filename = f"worker-{socket.gethostname()}-task-{task_id:05d}-shard-{shard_idx:05d}.safetensors"
    safetensors.torch.save_file(tensors, os.path.join(staging_dir, filename))
    return ShardManifest(filename, None, tuple(tensors.keys()), current_size, len(tensors))


def append_to_shards(
    staging_dir: str,
    task_id: int,
    shard_idx: int,
    current_tensors: dict[str, torch.Tensor],
    current_size: int,
    converted_named_tensors: tuple[tuple[str, torch.Tensor], ...] | list[tuple[str, torch.Tensor]],
    max_file_bytes: int,
    shards: list[ShardManifest],
) -> tuple[int, int, int]:
    total_size = 0
    for converted_name, converted_param in converted_named_tensors:
        tensor_size = converted_param.numel() * converted_param.element_size()
        if tensor_size + current_size > max_file_bytes and current_tensors:
            shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))
            shard_idx += 1
            current_tensors.clear()
            current_size = 0
        current_tensors[converted_name] = converted_param
        current_size += tensor_size
        total_size += tensor_size
    return shard_idx, current_size, total_size


def prepare_moe_block_task_tensors(
    task: TaskSpec,
    input_dir: str,
    metadata: dist_cp.metadata.Metadata,
    model_name: str,
) -> PreparedTaskTensors:
    groups: list[PreparedTensorGroup] = []
    total_read_items = 0
    total_files = 0
    total_storage_bytes = 0
    for block in task.moe_blocks:
        result = load_moe_block_direct(input_dir, block, metadata, model_name)
        groups.extend(result.tensor_groups)
        total_read_items += result.read_items
        total_files += result.files
        total_storage_bytes += result.storage_bytes
    return PreparedTaskTensors(tuple(groups), TaskLoadStats(total_read_items, total_files, total_storage_bytes))


def prepare_whole_source_task_tensors(
    task: TaskSpec,
    input_dir: str,
    megatron_args: Any,
    model_name: str,
    metadata: dist_cp.metadata.Metadata,
) -> PreparedTaskTensors:
    load_result = load_tensor_chunk(input_dir, set(task.keys), metadata)
    state_dict = load_result.state_dict

    groups: list[PreparedTensorGroup] = []
    try:
        with m2hf.conversion_cache_scope():
            for name, param in get_named_params(megatron_args, state_dict):
                if getattr(megatron_args, "vocab_size", None) is not None:
                    param = m2hf.remove_padding(name, param, megatron_args.vocab_size)
                converted_named_tensors = m2hf._convert_to_hf_core(megatron_args, model_name, name, param)
                groups.append(PreparedTensorGroup(name, tuple(converted_named_tensors)))
        return PreparedTaskTensors(
            tuple(groups),
            TaskLoadStats(load_result.read_items, load_result.files, load_result.storage_bytes),
        )
    finally:
        del state_dict


def write_prepared_tensor_groups(
    staging_dir: str,
    task_id: int,
    groups: tuple[PreparedTensorGroup, ...],
    megatron_args: Any,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    cuda_device_id: int | None,
) -> tuple[tuple[ShardManifest, ...], int]:
    current_tensors: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_idx = 0
    shards: list[ShardManifest] = []
    total_size = 0

    for group in groups:
        converted_named_tensors = group.tensors
        if quantization_config is not None:
            if cuda_device_id is None:
                raise RuntimeError("Quantized HF export requires a CUDA device per conversion worker")
            torch.cuda.set_device(cuda_device_id)
            device = torch.device("cuda", cuda_device_id)
            converted_named_tensors = tuple(
                (name, tensor.to(device=device, non_blocking=False)) for name, tensor in converted_named_tensors
            )
            converted_named_tensors = tuple(
                m2hf.quantize_params(
                    megatron_args,
                    group.source_name,
                    list(converted_named_tensors),
                    quantization_config,
                    transform_ue8m0=False,
                )
            )
            converted_named_tensors = tuple(
                (name, tensor.detach().cpu().contiguous()) for name, tensor in converted_named_tensors
            )
        shard_idx, current_size, added_size = append_to_shards(
            staging_dir,
            task_id,
            shard_idx,
            current_tensors,
            current_size,
            converted_named_tensors,
            max_file_bytes,
            shards,
        )
        total_size += added_size

    if current_tensors:
        shards.append(_flush_shard(staging_dir, task_id, shard_idx, current_tensors, current_size))
    return tuple(shards), total_size


def assign_cuda_device_id(actor_id: int, device_count: int) -> int:
    if device_count <= 0:
        raise RuntimeError("Quantization requires CUDA, but no CUDA devices are visible")
    return actor_id % device_count


def initialize_worker_cuda_device(actor_id: int, quantization_config: dict[str, Any] | None) -> int | None:
    if quantization_config is None:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("Quantization requires CUDA, but CUDA is unavailable")
    cuda_device_id = assign_cuda_device_id(actor_id, torch.cuda.device_count())
    torch.cuda.set_device(cuda_device_id)
    return cuda_device_id


class ConversionWorker:
    def __init__(
        self,
        actor_id: int,
        input_dir: str,
        staging_dir: str,
        megatron_args: Any,
        model_name: str,
        quantization_config: dict[str, Any] | None,
        max_file_bytes: int,
        metadata_ref: Any,
    ) -> None:
        self.actor_id = actor_id
        self.ray_node_id = ray.get_runtime_context().get_node_id()
        self.cuda_device_id = initialize_worker_cuda_device(actor_id, quantization_config)
        self.input_dir = input_dir
        self.staging_dir = staging_dir
        self.megatron_args = megatron_args
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.max_file_bytes = max_file_bytes
        self.metadata = metadata_ref if isinstance(metadata_ref, dist_cp.metadata.Metadata) else ray.get(metadata_ref)

    def convert(self, task: TaskSpec) -> TaskResult:
        node = socket.gethostname()
        pid = os.getpid()
        if task.moe_blocks:
            prepared = prepare_moe_block_task_tensors(task, self.input_dir, self.metadata, self.model_name)
        else:
            prepared = prepare_whole_source_task_tensors(
                task, self.input_dir, self.megatron_args, self.model_name, self.metadata
            )
        shards, total_size = write_prepared_tensor_groups(
            self.staging_dir,
            task.task_id,
            prepared.groups,
            self.megatron_args,
            self.quantization_config,
            self.max_file_bytes,
            self.cuda_device_id,
        )
        return TaskResult(
            task_id=task.task_id,
            actor_id=self.actor_id,
            node=node,
            pid=pid,
            shards=shards,
            source_bytes=task.estimated_source_bytes,
            output_bytes=total_size,
            weights=sum(len(shard.weight_keys) for shard in shards),
            source_keys=task.keys,
            dcp_read_items=prepared.load_stats.read_items,
            dcp_files=prepared.load_stats.files,
            dcp_storage_bytes=prepared.load_stats.storage_bytes,
            cuda_device_id=self.cuda_device_id,
            ray_node_id=self.ray_node_id,
        )


def initialize_ray() -> None:
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if ray.is_initialized():
        return
    try:
        ray.init(address="auto", ignore_reinit_error=True)
    except ConnectionError:
        ray.init(ignore_reinit_error=True)


def live_ray_node_ids(*, requires_gpu: bool = False) -> list[str]:
    node_ids = [
        str(node["NodeID"])
        for node in ray.nodes()
        if node.get("Alive", False)
        and (not requires_gpu or float(node.get("Resources", {}).get("GPU", 0)) >= 1)
    ]
    node_ids.sort()
    if not node_ids:
        resource = "GPU-capable " if requires_gpu else ""
        raise RuntimeError(f"Ray has no live {resource}nodes")
    return node_ids


def make_conversion_actor(*, requires_gpu: bool):
    return ray.remote(
        num_cpus=1,
        num_gpus=1 if requires_gpu else 0,
        runtime_env={
            "env_vars": {
                "PYTHONUNBUFFERED": "1",
                "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
            }
        },
    )(ConversionWorker)


def collect_ray_results(
    tasks: list[TaskSpec],
    input_dir: str,
    staging_dir: str,
    megatron_args: Any,
    model_name: str,
    quantization_config: dict[str, Any] | None,
    max_file_bytes: int,
    concurrency: int,
    metadata_ref: Any,
    progress: bool,
    progress_interval_seconds: float,
) -> list[TaskResult]:
    resource_name = "GPU" if quantization_config is not None else "CPU"
    available_workers = int(ray.cluster_resources().get(resource_name, 0))
    if available_workers < 1:
        raise RuntimeError(f"Ray has no available {resource_name} resources for conversion workers")
    worker_count = min(concurrency, len(tasks), available_workers)
    if worker_count < 1:
        return []

    worker_cls = make_conversion_actor(requires_gpu=quantization_config is not None)
    node_ids = live_ray_node_ids(requires_gpu=quantization_config is not None)
    workers = []
    for actor_id in range(worker_count):
        node_id = node_ids[actor_id % len(node_ids)]
        actor_cls = worker_cls.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False))
        workers.append(
            actor_cls.remote(
                actor_id,
                input_dir,
                staging_dir,
                megatron_args,
                model_name,
                quantization_config,
                max_file_bytes,
                metadata_ref,
            )
        )

    pending: dict[Any, int] = {}
    submitted = 0
    results: list[TaskResult] = []
    progress_reporter = ProgressReporter(tasks, progress, progress_interval_seconds)
    progress_reporter.tick()
    for worker_idx, worker in enumerate(workers):
        if submitted >= len(tasks):
            break
        pending[worker.convert.remote(tasks[submitted])] = worker_idx
        submitted += 1

    while pending:
        ready, _ = ray.wait(list(pending), num_returns=1, timeout=progress_reporter.interval_seconds)
        if not ready:
            progress_reporter.tick()
            continue
        ready_ref = ready[0]
        worker_idx = pending.pop(ready_ref)
        result = ray.get(ready_ref)
        results.append(result)
        progress_reporter.complete(result)
        if not progress:
            print(
                f"task {result.task_id} finished on {result.node}: "
                f"{result.output_bytes / 1e9:.2f} GB output, {result.weights} tensors"
            )
        if submitted < len(tasks):
            pending[workers[worker_idx].convert.remote(tasks[submitted])] = worker_idx
            submitted += 1

    progress_reporter.finish()
    results.sort(key=lambda result: result.task_id)
    return results


def plan_global_shards(task_results: list[TaskResult]) -> tuple[tuple[PlannedShard, ...], dict[str, Any]]:
    all_shards = [shard for result in sorted(task_results, key=lambda item: item.task_id) for shard in result.shards]
    if not all_shards:
        raise ValueError("No HF tensor shards were emitted")
    weight_map: dict[str, str] = {}
    planned_shards: list[PlannedShard] = []
    total_size = 0
    for idx, shard in enumerate(all_shards):
        final_filename = f"model-{idx:05d}-of-{len(all_shards):05d}.safetensors"
        for key in shard.weight_keys:
            if key in weight_map:
                raise ValueError(f"Duplicate HF tensor emitted during finalization: {key}")
            weight_map[key] = final_filename
        planned_shards.append(PlannedShard(shard.temp_filename, final_filename, shard.weight_keys, shard.bytes))
        total_size += shard.bytes
    return tuple(planned_shards), {"metadata": {"total_size": total_size}, "weight_map": weight_map}


def finalize_output(
    staging_dir: str,
    output_dir: str,
    origin_hf_dir: str | None,
    task_results: list[TaskResult],
    model_name: str,
) -> None:
    planned_shards, index = plan_global_shards(task_results)
    copy_assets(origin_hf_dir, output_dir)
    for shard in planned_shards:
        os.replace(os.path.join(staging_dir, shard.temp_filename), os.path.join(output_dir, shard.final_filename))
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    if is_glm5_next_model(model_name):
        from slime.backends.megatron_utils.hf_checkpoint_saver import _normalize_glm5_next_training_config

        _normalize_glm5_next_training_config(Path(output_dir))
    shutil.rmtree(staging_dir)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_conversion_manifest(
    args: Args,
    model_name: str,
    plan: dict[str, Any],
    task_results: list[TaskResult],
    elapsed_seconds: float,
) -> None:
    index_path = os.path.join(args.output_dir, "model.safetensors.index.json")
    with open(index_path, encoding="utf-8") as handle:
        index = json.load(handle)

    source_assets: dict[str, dict[str, Any]] = {}
    if args.origin_hf_dir is not None:
        for filename in ("config.json", "model.safetensors.index.json"):
            path = os.path.join(args.origin_hf_dir, filename)
            if os.path.isfile(path):
                source_assets[filename] = {
                    "path": os.path.abspath(path),
                    "sha256": sha256_file(path),
                    "bytes": os.path.getsize(path),
                }

    actors = sorted(
        {
            (result.ray_node_id, result.node, result.actor_id, result.cuda_device_id)
            for result in task_results
            if result.actor_id >= 0
        }
    )
    manifest = {
        "format_version": 1,
        "model_name": model_name,
        "input": {
            "torch_dist": os.path.abspath(args.input_dir),
            "metadata_sha256": sha256_file(os.path.join(args.input_dir, ".metadata")),
            "origin_hf": os.path.abspath(args.origin_hf_dir) if args.origin_hf_dir else None,
            "source_assets": source_assets,
        },
        "settings": {
            "concurrency": plan["concurrency"],
            "task_group_bytes": args.task_group_bytes,
            "max_file_bytes": args.max_file_bytes,
            "source_key_regex": args.source_key_regex,
            "quantization_uses_one_gpu_per_worker": any(
                result.cuda_device_id is not None for result in task_results
            ),
        },
        "plan": plan,
        "execution": {
            "elapsed_seconds": elapsed_seconds,
            "tasks": len(task_results),
            "actors": [
                {
                    "ray_node_id": node_id,
                    "hostname": hostname,
                    "actor_id": actor_id,
                    "cuda_device_id": cuda_device_id,
                }
                for node_id, hostname, actor_id, cuda_device_id in actors
            ],
            "source_bytes": sum(result.source_bytes for result in task_results),
            "output_bytes": sum(result.output_bytes for result in task_results),
            "dcp_read_items": sum(result.dcp_read_items for result in task_results),
            "dcp_storage_bytes": sum(result.dcp_storage_bytes for result in task_results),
            "dcp_files_touched_sum": sum(result.dcp_files for result in task_results),
        },
        "output": {
            "path": os.path.abspath(args.output_dir),
            "index_sha256": sha256_file(index_path),
            "weights": len(index["weight_map"]),
            "shards": len(set(index["weight_map"].values())),
            "total_size": index["metadata"]["total_size"],
        },
    }
    manifest_path = os.path.join(args.output_dir, "conversion-manifest.json")
    temporary_path = f"{manifest_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_path, manifest_path)


def reject_cloud_path(path: str, label: str) -> None:
    if "://" in path:
        raise ValueError(f"{label} must be a local filesystem path, got {path}")


def validate_path_layout(
    input_dir: str,
    output_dir: str,
    origin_hf_dir: str | None,
) -> None:
    """Reject broad or overlapping output paths before --force can delete data."""
    output = Path(output_dir).resolve()
    if output in {Path(output.anchor), Path.home().resolve()}:
        raise ValueError(f"refusing unsafe output_dir: {output}")

    sources = {"input_dir": Path(input_dir).resolve()}
    if origin_hf_dir is not None:
        sources["origin_hf_dir"] = Path(origin_hf_dir).resolve()
    for label, source in sources.items():
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError(f"output_dir must not overlap {label}: {output} vs {source}")


def prepare_output_dir(output_dir: str, force: bool) -> str:
    reject_cloud_path(output_dir, "output_dir")
    if os.path.exists(output_dir):
        if not force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    staging_dir = os.path.join(output_dir, ".ray-convert-staging")
    os.makedirs(staging_dir, exist_ok=True)
    return staging_dir


def _find_megatron_args(value: Any) -> Any | None:
    if isinstance(value, dict):
        args = value.get("args")
        if args is not None:
            return args
        for nested in value.values():
            result = _find_megatron_args(nested)
            if result is not None:
                return result
    elif isinstance(value, (list, tuple)):
        for nested in value:
            result = _find_megatron_args(nested)
            if result is not None:
                return result
    return None


def _load_megatron_args_from_dcp(input_dir: str, metadata: dist_cp.metadata.Metadata) -> Any:
    if metadata.storage_data is None:
        raise ValueError("DCP metadata does not contain storage_data for common_state")
    candidates = [
        (index, info)
        for index, info in metadata.storage_data.items()
        if index.fqn.startswith("common_state/")
    ]
    for _index, info in sorted(candidates, key=lambda item: item[0].fqn):
        descriptors = getattr(info, "transform_descriptors", None) or ()
        if descriptors:
            raise ValueError("Transformed DCP common_state is not supported by the offline exporter")
        checkpoint_file = os.path.join(input_dir, info.relative_path)
        with open(checkpoint_file, "rb") as stream:
            stream.seek(info.offset)
            payload = torch.load(io.BytesIO(stream.read(info.length)), map_location="cpu", weights_only=False)
        megatron_args = _find_megatron_args(payload)
        if megatron_args is not None:
            return megatron_args
    raise ValueError("DCP checkpoint does not contain Megatron args in common_state")


def load_megatron_args(
    input_dir: str,
    metadata: dist_cp.metadata.Metadata,
    model_name_override: str | None,
    vocab_size: int | None,
) -> tuple[Any, str]:
    common_pt = os.path.join(input_dir, "common.pt")
    if os.path.exists(common_pt):
        common = torch.load(common_pt, weights_only=False, map_location="cpu")
        megatron_args = common["args"]
    else:
        megatron_args = _load_megatron_args_from_dcp(input_dir, metadata)
    model_name = model_name_override or getattr(megatron_args, "original_hf_model_name", None)
    if model_name is None:
        raise ValueError("Model name is required when the checkpoint does not record original_hf_model_name")
    if vocab_size is not None:
        megatron_args.vocab_size = vocab_size
    if not hasattr(megatron_args, "sglang_enable_ep_moe"):
        megatron_args.sglang_enable_ep_moe = False
    return megatron_args, model_name


def read_metadata_and_plan(
    args: Args,
    megatron_args: Any,
    metadata: dist_cp.metadata.Metadata,
) -> list[TaskSpec]:
    tensor_metadata = tensor_metadata_from_checkpoint_metadata(metadata)
    tensor_metadata = filter_tensor_metadata(tensor_metadata, args.source_key_regex)
    if args.source_key_regex and not tensor_metadata:
        raise ValueError(f"No checkpoint keys matched {args.source_key_regex}")
    tasks = plan_conversion_tasks(
        tensor_metadata, metadata, getattr(megatron_args, "q_lora_rank", None), args.task_group_bytes
    )
    return tasks


def convert_torch_dist_to_hf_ray(args: Args) -> str:
    started_at = time.monotonic()
    reject_cloud_path(args.input_dir, "input_dir")
    if args.origin_hf_dir is not None:
        reject_cloud_path(args.origin_hf_dir, "origin_hf_dir")
    validate_path_layout(args.input_dir, args.output_dir, args.origin_hf_dir)
    metadata_path = os.path.join(args.input_dir, ".metadata")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Expected {metadata_path}")

    hf_config = load_hf_config(args.origin_hf_dir)
    vocab_size = get_hf_vocab_size(hf_config)
    metadata = WrappedStorageReader(args.input_dir).read_metadata()
    megatron_args, model_name = load_megatron_args(
        args.input_dir,
        metadata,
        args.model_name,
        vocab_size,
    )
    quantization_config = load_quantization_config(hf_config)

    tasks = read_metadata_and_plan(args, megatron_args, metadata)
    concurrency = args.concurrency or min(max(len(tasks), 1), 16)
    plan = summarize_plan(tasks, model_name, concurrency, args.output_dir)
    print(json.dumps(plan, indent=2, default=str))
    if args.dry_run_plan:
        return args.output_dir
    if not tasks:
        raise ValueError("No checkpoint tensor tasks were planned")

    staging_dir = prepare_output_dir(args.output_dir, args.force)
    initialize_ray()
    metadata_ref = ray.put(metadata)
    task_results = collect_ray_results(
        tasks,
        args.input_dir,
        staging_dir,
        megatron_args,
        model_name,
        quantization_config,
        args.max_file_bytes,
        concurrency,
        metadata_ref,
        args.progress,
        args.progress_interval_seconds,
    )
    if is_glm5_next_model(model_name):
        task_results.append(
            copy_frozen_source_tensors(
                args.origin_hf_dir,
                staging_dir,
                task_id=len(tasks),
                max_file_bytes=args.max_file_bytes,
                prefix="model.visual.",
            )
        )
    finalize_output(staging_dir, args.output_dir, args.origin_hf_dir, task_results, model_name)
    write_conversion_manifest(args, model_name, plan, task_results, time.monotonic() - started_at)
    return args.output_dir


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--origin-hf-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--max-file-bytes", type=int, default=5 * 1024**3)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--task-group-bytes", type=int, default=0)
    parser.add_argument("--source-key-regex", default=None)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.set_defaults(progress=True)
    ns = parser.parse_args()
    return Args(
        input_dir=ns.input_dir,
        output_dir=ns.output_dir,
        origin_hf_dir=ns.origin_hf_dir,
        model_name=ns.model_name,
        force=ns.force,
        max_file_bytes=ns.max_file_bytes,
        concurrency=ns.concurrency,
        task_group_bytes=ns.task_group_bytes,
        source_key_regex=ns.source_key_regex,
        dry_run_plan=ns.dry_run_plan,
        progress=ns.progress,
        progress_interval_seconds=ns.progress_interval_seconds,
    )


def main() -> None:
    convert_torch_dist_to_hf_ray(parse_args())


if __name__ == "__main__":
    main()

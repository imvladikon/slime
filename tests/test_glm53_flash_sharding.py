import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from slime.backends.megatron_utils.hf_to_megatron.common import _tensor_parallel_shard

NUM_GPUS = 0
TP_SIZE = 4
HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 256
MOE_INTERMEDIATE_SIZE = 128
KDA_KERNEL_SIZE = 4


def _gather(shard):
    gathered = [torch.empty_like(shard) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, shard)
    return gathered


def _run_tp4_contract(rank, init_method):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=TP_SIZE)
    try:
        dense_fc1 = torch.arange(2 * INTERMEDIATE_SIZE * HIDDEN_SIZE).reshape(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
        dense_shard = _tensor_parallel_shard(
            "module.module.decoder.layers.0.mlp.linear_fc1.weight",
            dense_fc1,
            parallel_size=TP_SIZE,
            parallel_rank=rank,
            partition_dim=0,
            partition_stride=1,
        )
        assert dense_shard.is_contiguous()
        assert dense_shard.numel() * TP_SIZE == dense_fc1.numel()
        dense_parts = [part.chunk(2, dim=0) for part in _gather(dense_shard)]
        dense_rebuilt = torch.cat(
            (
                torch.cat([part[0] for part in dense_parts], dim=0),
                torch.cat([part[1] for part in dense_parts], dim=0),
            ),
            dim=0,
        )
        assert torch.equal(dense_rebuilt, dense_fc1)

        expert_fc2 = torch.arange(HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE).reshape(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
        expert_shard = _tensor_parallel_shard(
            "module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
            expert_fc2,
            parallel_size=TP_SIZE,
            parallel_rank=rank,
            partition_dim=0,
            partition_stride=1,
        )
        assert expert_shard.is_contiguous()
        assert expert_shard.numel() * TP_SIZE == expert_fc2.numel()
        assert torch.equal(torch.cat(_gather(expert_shard), dim=1), expert_fc2)

        kda_conv = torch.arange(3 * HIDDEN_SIZE * KDA_KERNEL_SIZE).reshape(3 * HIDDEN_SIZE, KDA_KERNEL_SIZE)
        kda_shard = _tensor_parallel_shard(
            "module.module.decoder.layers.0.self_attention.kda.conv1d.weight",
            kda_conv,
            parallel_size=TP_SIZE,
            parallel_rank=rank,
            partition_dim=0,
            partition_stride=1,
        )
        assert kda_shard.is_contiguous()
        assert kda_shard.numel() * TP_SIZE == kda_conv.numel()
        kda_parts = [part.chunk(3, dim=0) for part in _gather(kda_shard)]
        kda_rebuilt = torch.cat(
            [torch.cat([part[axis] for part in kda_parts], dim=0) for axis in range(3)],
            dim=0,
        )
        assert torch.equal(kda_rebuilt, kda_conv)
    finally:
        dist.destroy_process_group()


def test_real_gloo_tp4_glm53_tensor_shards_round_trip(tmp_path):
    init_method = f"file://{tmp_path / 'tp4-init'}"
    mp.spawn(_run_tp4_contract, args=(init_method,), nprocs=TP_SIZE, join=True)

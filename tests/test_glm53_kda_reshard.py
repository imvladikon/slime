"""The KDA convolution has to reshard, and it has to keep its optimizer state.

Two separate contracts are checked against real Megatron-Core, both of which a
slice-based export breaks:

* Optimizer identity. ``get_param_id_to_sharded_param_map`` matches the model to
  the optimizer by ``id(ten.data)``. Exporting ``weight[section]`` hands it a
  fresh tensor, so the convolution silently loses its Adam moments -- the mapper
  only writes a debug line. Measured before the fix: unmapped at TP2/4/8.

* One logical schema. Sections exported as separate storage keys with a
  prepended axis gave ``[96, 3]`` at TP1 and ``[3, 32, 3]`` at TP>1 for the same
  key, so a checkpoint written at one TP could not be read at another.

Run:

    PYTHONPATH=<megatron> CUDA_VISIBLE_DEVICES=0,1 \
        torchrun --nproc_per_node=2 test_glm53_kda_reshard.py <kda.py> save
    PYTHONPATH=<megatron> CUDA_VISIBLE_DEVICES=0 \
        torchrun --nproc_per_node=1 test_glm53_kda_reshard.py <kda.py> load

The convolution is filled so that query, key and value differ and carry their
global row index: a permutation cannot hide behind equal values.
"""
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.optimizer import (
    get_param_id_to_sharded_param_map,
    make_sharded_optimizer_tensor,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

HEADS, HEAD_DIM, HIDDEN, KERNEL = 8, 4, 16, 3
CKPT = os.environ.get("CKPT", "/tmp/kda_reshard_ckpt")


def load_kda_methods(path):
    """Read the shipped implementation so the gate cannot drift from the code."""
    src = open(path).read()
    start = src.index("    def sharded_state_dict(")
    end = src.index("class Glm5NextKDAAttention")
    body = "from megatron.core import parallel_state\n\nclass _M:\n" + src[start:end]
    namespace = {}
    exec(compile(body, path, "exec"), namespace)
    return namespace["_M"].sharded_state_dict, namespace["_M"]._sharded_conv1d


class Conv(nn.Module):
    def __init__(self, rows, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, kernel))


class KdaLike(nn.Module):
    """Module mirror: the real block needs the FLA kernels, the export does not."""

    def __init__(self, config):
        super().__init__()
        tp = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = tp
        self.local_heads = HEADS // tp
        self.proj = self.local_heads * HEAD_DIM

        def column(inp, out):
            return ColumnParallelLinear(
                inp, out, config=config, init_method=config.init_method, bias=False, gather_output=False
            )

        for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "g_b_proj"):
            setattr(self, name, column(HIDDEN, HEADS * HEAD_DIM))
        self.b_proj = column(HIDDEN, HEADS)
        self.o_proj = RowParallelLinear(
            HEADS * HEAD_DIM, HIDDEN, config=config, init_method=config.output_layer_init_method,
            bias=False, input_is_parallel=True, skip_bias_add=False,
        )
        self.conv1d = Conv(3 * self.proj, KERNEL)
        self.A_log = nn.Parameter(torch.zeros(self.local_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.proj))
        self.f_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.g_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.o_norm = nn.LayerNorm(HEAD_DIM)
        if tp > 1:
            for param in (self.conv1d.weight, self.A_log, self.dt_bias):
                tensor_parallel.set_tensor_model_parallel_attributes(param, True, 0, 1)


def expected_conv(tp_rank, tp_size):
    """Global [Q | K | V], each row tagged by section and global row index.

    Values stay below 256 so every one of them is exact in bf16: the gate is
    about ordering, and rounding must not be able to look like a permutation.
    """
    per_section = HEADS * HEAD_DIM
    local = per_section // tp_size
    rows = [
        float(section * per_section + tp_rank * local + index)
        for section in range(3)
        for index in range(local)
    ]
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(1).repeat(1, KERNEL)


def build(kda_path, tp):
    config = TransformerConfig(
        num_layers=1, hidden_size=HIDDEN, num_attention_heads=HEADS, tensor_model_parallel_size=tp
    )
    module = KdaLike(config).cuda()
    sharded_state_dict, sharded_conv1d = load_kda_methods(kda_path)
    module.sharded_state_dict = sharded_state_dict.__get__(module)
    module._sharded_conv1d = sharded_conv1d.__get__(module)
    return module


def optimizer_state(module, sharded):
    """Sharded optimizer state, built the way MCore builds it for a real run."""
    optimizer = torch.optim.AdamW(list(module.parameters()), lr=1e-3)
    sum(param.float().pow(2).sum() for param in module.parameters()).backward()
    optimizer.step()
    id_map = get_param_id_to_sharded_param_map(sharded, module.parameters())
    order = [name for name, _ in module.named_parameters()]
    unmapped = [order[i] for i in range(len(order)) if i not in id_map]
    if unmapped:
        raise SystemExit(f"параметры без связи с оптимизатором: {unmapped}")
    state = {}
    for param_id, model_param in id_map.items():
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor = optimizer.state[list(module.parameters())[param_id]][moment]
            state[f"optimizer.{moment}.{order[param_id]}"] = make_sharded_optimizer_tensor(
                model_param, tensor, f"optimizer.{moment}."
            )
    return state


def main():
    kda_path, action = sys.argv[1], sys.argv[2]
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group("nccl")
    tp = dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(17)

    module = build(kda_path, tp)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()

    if action == "save":
        if dist.get_rank() == 0:
            os.makedirs(CKPT, exist_ok=True)
        dist.barrier()
        sharded = module.sharded_state_dict(prefix="")
        # The optimizer step has to happen before the pattern is written, or it
        # moves the very weights the comparison is about.
        sharded.update(optimizer_state(module, sharded))
        with torch.no_grad():
            module.conv1d.weight.copy_(expected_conv(tp_rank, tp).to(module.conv1d.weight.dtype).cuda())
        dist_checkpointing.save(sharded, CKPT)
        if dist.get_rank() == 0:
            print(f"сохранено при TP={tp}")
    else:
        sharded = module.sharded_state_dict(prefix="")
        sharded.update(optimizer_state(module, sharded))
        loaded = dist_checkpointing.load(sharded, CKPT)
        got = loaded["conv1d.weight"].cuda() if torch.is_tensor(loaded["conv1d.weight"]) else module.conv1d.weight
        want = expected_conv(tp_rank, tp).cuda()
        if not torch.equal(got.float(), want.float()):
            raise SystemExit(
                f"свёртка после решардинга не совпала.\nожидалось {want[:, 0].tolist()}\nполучено {got[:, 0].tolist()}"
            )
        moments = [key for key in loaded if key.startswith("optimizer.") and "conv1d" in key]
        if len(moments) != 2:
            raise SystemExit(f"состояние Adam для свёртки не восстановлено: {moments}")
        if dist.get_rank() == 0:
            print(f"загружено при TP={tp}: свёртка и моменты Adam совпали")

    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

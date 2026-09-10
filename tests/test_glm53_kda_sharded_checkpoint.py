"""KDA must describe its TP shards, or the checkpoint keeps 1/TP of the weights.

Glm5NextKDA is a plain nn.Module, so without its own sharded_state_dict MCore
takes the generic path: a recursive state_dict and an empty TP-axis map. Every
tensor is then written as replicated, each rank claims the whole parameter, and
they collide on the same global position. Marking parameters with
tensor_model_parallel does not help — the generic helper never reads it.

Run it with two ranks against the module mirror below:

    PYTHONPATH=<megatron> CUDA_VISIBLE_DEVICES=1,2 TP=2 FIXED=1 \
        torchrun --nproc_per_node=2 test_glm53_kda_sharded_checkpoint.py <kda.py>

FIXED=0 reproduces the old behaviour, which MCore's own validator rejects with
"Invalid access pattern". FIXED=1 has to round-trip every tensor exactly.

The sharded_state_dict under test is read out of the shipped kda.py rather than
copied here, so the test cannot drift away from the code it guards. FSDP and the
distributed checkpoint are real; only the module structure is a mirror, because
constructing the real block needs the FLA kernels.
"""
import os, sys, shutil
import torch, torch.nn as nn
import torch.distributed as dist

from megatron.core import parallel_state, tensor_parallel, dist_checkpointing


from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

TP = int(os.environ.get("TP", "2"))
HEADS, HEAD_DIM, HIDDEN, KERNEL = 8, 4, 16, 3
CKPT = os.environ.get("CKPT", "/tmp/f1_ckpt")
FIXED = os.environ.get("FIXED", "1") == "1"


def load_kda_methods(path):
    src = open(path).read()
    start = src.index("    def sharded_state_dict(")
    end = src.index("class Glm5NextKDAAttention")
    body = "from megatron.core import parallel_state\n\nclass _M:\n" + src[start:end]
    ns = {}
    exec(compile(body, path, "exec"), ns)
    return ns["_M"].sharded_state_dict, ns["_M"]._sharded_conv1d


class Conv(nn.Module):
    def __init__(self, rows, kernel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(rows, kernel))


class KdaLike(nn.Module):
    def __init__(self, config):
        super().__init__()
        tp = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_size = tp
        self.local_heads = HEADS // tp
        self.proj = self.local_heads * HEAD_DIM

        def column(i, o):
            return ColumnParallelLinear(i, o, config=config, init_method=config.init_method,
                                        bias=False, gather_output=False)

        for name in ("q_proj", "k_proj", "v_proj", "f_b_proj", "g_b_proj"):
            setattr(self, name, column(HIDDEN, HEADS * HEAD_DIM))
        self.b_proj = column(HIDDEN, HEADS)
        self.o_proj = RowParallelLinear(HEADS * HEAD_DIM, HIDDEN, config=config,
                                        init_method=config.output_layer_init_method, bias=False,
                                        input_is_parallel=True, skip_bias_add=False)
        self.conv1d = Conv(3 * self.proj, KERNEL)
        self.A_log = nn.Parameter(torch.zeros(self.local_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.proj))
        self.f_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.g_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias=False)
        self.o_norm = nn.LayerNorm(HEAD_DIM)
        if tp > 1:
            tensor_parallel.set_tensor_model_parallel_attributes(self.conv1d.weight, True, 0, 1)
            tensor_parallel.set_tensor_model_parallel_attributes(self.A_log, True, 0, 1)
            tensor_parallel.set_tensor_model_parallel_attributes(self.dt_bias, True, 0, 1)


def build(config, kda_path):
    module = KdaLike(config)
    if torch.cuda.is_available():
        module = module.cuda()
    if FIXED:
        ssd, conv = load_kda_methods(kda_path)
        KdaLike.sharded_state_dict = ssd
        KdaLike._sharded_conv1d = conv
    else:
        # Поведение до фикса: у класса нет собственного метода, и MCore идёт
        # обобщённым путём. Атрибут именно снимается, а не подменяется.
        if "sharded_state_dict" in KdaLike.__dict__:
            delattr(KdaLike, "sharded_state_dict")
    return module


def main():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    rank = dist.get_rank()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=TP)
    if torch.cuda.is_available():
        # ColumnParallelLinear инициализируется под model-parallel RNG.
        tensor_parallel.model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(num_layers=1, hidden_size=HIDDEN, num_attention_heads=HEADS,
                               tensor_model_parallel_size=TP, use_cpu_initialization=not torch.cuda.is_available(),
                               pipeline_model_parallel_size=1)
    kda_path = sys.argv[1]

    saved = build(config, kda_path)
    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    # Реплицированные параметры обязаны совпадать на всех рангах — это их
    # определение. Метить их рангом бессмысленно: чекпойнт законно хранит одну
    # копию. Рангом метятся только шардированные.
    REPLICATED = ("f_a_proj", "g_a_proj", "o_norm")
    with torch.no_grad():
        for name, p in saved.named_parameters():
            replicated = any(name.startswith(r) for r in REPLICATED)
            p.fill_(9.0 if replicated else float(tp_rank + 1))
    before = {n: p.detach().clone() for n, p in saved.named_parameters()}

    if rank == 0:
        if os.path.isdir(CKPT):
            shutil.rmtree(CKPT)
        os.makedirs(CKPT, exist_ok=True)
    dist.barrier()
    from megatron.core.transformer.utils import sharded_state_dict_default
    def describe(module):
        if "sharded_state_dict" in type(module).__dict__:
            return module.sharded_state_dict(prefix="kda.")
        return sharded_state_dict_default(module, "kda.")
    dist_checkpointing.save(describe(saved), CKPT)
    dist.barrier()

    loaded = build(config, kda_path)
    with torch.no_grad():
        for p in loaded.parameters():
            p.fill_(-1.0)
    state = dist_checkpointing.load(describe(loaded), CKPT)
    # load отдаёт обычные тензоры по тем же ключам; кладём обратно в модуль.
    for name, p in loaded.named_parameters():
        key = "kda." + name
        if key in state:
            with torch.no_grad():
                p.copy_(state[key])

    bad = []
    for name, p in loaded.named_parameters():
        if not torch.equal(p, before[name]):
            bad.append((name, float(before[name].flatten()[0]), float(p.flatten()[0])))
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, (rank, tp_rank, bad))
    if rank == 0:
        mode = "С ФИКСОМ" if FIXED else "БЕЗ ФИКСА"
        print(f"=== {mode}, TP={TP}: round-trip save -> load")
        total = 0
        for r, tpr, items in counts:
            total += len(items)
            status = "все тензоры вернулись" if not items else f"ИСПОРЧЕНО {len(items)}"
            print(f"  rank {r} (tp {tpr}): {status}")
            for name, want, got in items[:4]:
                print(f"      {name:24s} записывали {want}, прочитали {got}")
        print(f"ИТОГО испорченных тензоров: {total}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

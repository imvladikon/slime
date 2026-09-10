#!/usr/bin/env python3
"""Whole Slime KDA vs unsharded oracle with real FLA and MCore collectives.

Source-only import from --slime-root; no package installation or source patching.
Launch with torchrun and explicit empty GPU UUIDs. The optional negative-control
flag expects the already-localized duplicate SUM, not a successful fixed block.
"""
import argparse
from datetime import timedelta
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--slime-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--expect-double-reduce', action='store_true')
    p.add_argument('--expect-no-sp-rejection', action='store_true')
    p.add_argument('--sequence-parallel', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[19, 713])
    p.set_defaults(dtype='bfloat16')
    a = p.parse_args()
    rank, local_rank, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    uuids = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if world not in (1, 2) or len(uuids) != world or not all(s.startswith('GPU-') for s in uuids):
        p.error('explicit GPU UUIDs and TP1/TP2 required')
    import pynvml
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByUUID(uuids[local_rank])
    if pynvml.nvmlDeviceGetComputeRunningProcesses(gpu):
        p.error('occupied GPU; no action taken')
    if pynvml.nvmlDeviceGetMemoryInfo(gpu).free < 12 * 1024**3:
        p.error('less than 12 GiB free; no action taken')
    pynvml.nvmlShutdown()
    sys.path.insert(0, str(a.slime_root.resolve()))
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear
    from megatron.core.transformer.transformer_config import TransformerConfig
    from slime_plugins.models.glm5_next import kda as source

    required_kernels = ('ShortConvolution', 'FusedRMSNormGated', 'chunk_kda', 'fused_kda_gate')
    if any(getattr(source, name) is None for name in required_kernels):
        raise ImportError('This environment lacks the FLA KDA imports; no kernel qualification ran')

    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, a.dtype)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(minutes=8))
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    began = time.monotonic()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    evidence = (a.output_dir / f'rank-{rank}.jsonl').open('x', buffering=1)

    def shard(name, value):
        if name == 'conv1d.weight':
            return torch.cat([t.chunk(world, dim=0)[rank] for t in value.chunk(3, dim=0)], dim=0)
        if name.startswith(('q_proj.', 'k_proj.', 'v_proj.', 'b_proj.', 'f_b_proj.', 'g_b_proj.')) or name in ('A_log', 'dt_bias'):
            return value.chunk(world, dim=0)[rank]
        if name.startswith('o_proj.'):
            return value.chunk(world, dim=1)[rank]
        if name.startswith(('f_a_proj.', 'g_a_proj.', 'o_norm.')):
            return value
        raise AssertionError(f'unclassified parameter: {name}')

    def metrics(x, y):
        x, y = x.detach().float(), y.detach().float()
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
        return dict(max_abs=float((x-y).abs().max()),
                    rel_l2=float((x-y).norm() / y.norm().clamp_min(1e-15)),
                    projected_scale=float((x*y).sum() / y.square().sum().clamp_min(1e-30)))

    try:
        config = TransformerConfig(
            num_layers=1, hidden_size=256, num_attention_heads=4,
            tensor_model_parallel_size=world,
            params_dtype=dtype, bf16=a.dtype == 'bfloat16',
            sequence_parallel=a.sequence_parallel and world > 1,
            gradient_accumulation_fusion=False, perform_initialization=False,
            use_cpu_initialization=False,
        )
        if a.expect_no_sp_rejection:
            assert world > 1 and not config.sequence_parallel
            try:
                source.Glm5NextKDA(256, 4, 128, 4, -5.0, 1e-6, device, torch.bfloat16, config=config)
            except ValueError as error:
                assert 'shared o_norm gradient' in str(error), str(error)
                evidence.write(json.dumps(dict(event='PASS', scope='unsupported TP without SP rejected',
                    kda_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()))+'\n')
                dist.barrier()
                return
            raise AssertionError('unsupported TP without SP was accepted')
        evidence.write(json.dumps(dict(event='source', kda_sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(),
            source_path=source.__file__, torch=torch.__version__, tp=world,
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            mcore_layers_sha256=hashlib.sha256(Path(inspect.getfile(ColumnParallelLinear)).read_bytes()).hexdigest(),
            kernels={name: inspect.getfile(inspect.unwrap(getattr(source, name))) for name in
                     required_kernels})) + '\n')
        for seed in a.seeds:
            for freeze_b in (False, True):
                for boundaries in ([0, 65], [0, 17, 65]):
                    torch.manual_seed(seed)
                    reference = source.Glm5NextKDA(256, 4, 128, 4, -5.0, 1e-6, device, dtype)
                    actual = source.Glm5NextKDA(256, 4, 128, 4, -5.0, 1e-6, device, dtype, config=config)
                    ref_params, params = dict(reference.named_parameters()), dict(actual.named_parameters())
                    assert ref_params.keys() == params.keys()
                    with torch.no_grad():
                        for name, param in params.items():
                            target = shard(name, ref_params[name])
                            assert param.shape == target.shape, name
                            param.copy_(target)
                    if freeze_b:
                        for model in (reference, actual):
                            model.f_b_proj.weight.requires_grad_(False)
                            model.g_b_proj.weight.requires_grad_(False)
                    torch.manual_seed(seed + 1)
                    initial = torch.randn(1, 65, 256, device=device, dtype=dtype) * 0.2
                    weights = torch.randn_like(initial).float()
                    xr, xa = [initial.detach().clone().requires_grad_() for _ in range(2)]
                    cu = torch.tensor(boundaries, dtype=torch.int32, device=device)
                    torch.cuda.reset_peak_memory_stats()
                    case_started = time.monotonic()
                    yr = reference(xr, cu)
                    (yr.float() * weights).mean().backward()
                    ya = actual(xa, cu)
                    (ya.float() * weights).mean().backward()
                    # Real MCore finalizer, with FP32 main_grad buffers as in
                    # ordinary BF16 Megatron training. No manual norm reduction.
                    actual.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
                    for param in actual.parameters():
                        param.main_grad = None if param.grad is None else param.grad.float().clone()
                    _allreduce_non_tensor_model_parallel_grads([actual], config)
                    output, input_grad = metrics(ya, yr), metrics(xa.grad, xr.grad)
                    gradients, updates = {}, {}
                    for name, param in params.items():
                        ref_param = ref_params[name]
                        assert (param.grad is None) == (ref_param.grad is None), name
                        if param.grad is None:
                            continue
                        expected_grad = shard(name, ref_param.grad)
                        gradients[name] = metrics(param.main_grad, expected_grad)
                        # Actual SGD on FP32 master weights avoids BF16 writeback
                        # rounding hiding a wrong derivative on small updates.
                        master = torch.nn.Parameter(param.detach().float().clone())
                        master.grad = param.main_grad.clone()
                        before = master.detach().clone()
                        torch.optim.SGD([master], lr=0.5).step()
                        updates[name] = metrics(master.detach() - before, -0.5 * expected_grad.float())
                    row = dict(event='case', seed=seed, freeze_b=freeze_b, boundaries=boundaries,
                               tp=world, sequence_parallel=config.sequence_parallel, dtype=a.dtype,
                               main_grad_dtype='float32',
                               expect_double_reduce=a.expect_double_reduce,
                               output=output, input_grad=input_grad, gradients=gradients, updates=updates,
                               cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                               wall_seconds=time.monotonic()-case_started)
                    evidence.write(json.dumps(row, allow_nan=False)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k not in ('gradients','updates')}), flush=True)
                    assert output['rel_l2'] < 0.025, output
                    for name, result in gradients.items():
                        factor = world if a.expect_double_reduce and name.startswith(('f_a_proj.', 'g_a_proj.')) else 1
                        # BF16 GEMM/head layouts differ; require a small aggregate
                        # residual after the explicitly expected negative control.
                        scaled = metrics(params[name].main_grad, shard(name, ref_params[name].grad).float()*factor)
                        assert scaled['rel_l2'] < (0.04 if a.dtype == 'bfloat16' else 2e-4), (name, scaled)
                        assert abs(result['projected_scale'] - factor) < 0.04*factor, (name, result)
                        update = updates[name]
                        assert abs(update['projected_scale'] - factor) < 0.04*factor, (name, update)
                    if not a.expect_double_reduce or world == 1:
                        assert input_grad['rel_l2'] < 0.025, input_grad
                    del reference, actual, ref_params, params, xr, xa, yr, ya
        evidence.write(json.dumps(dict(event='PASS', scope='whole KDA block with real FLA/MCore; not full model SFT/RL',
                                       wall_seconds=time.monotonic()-began))+'\n')
        dist.barrier()
    finally:
        evidence.close()
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

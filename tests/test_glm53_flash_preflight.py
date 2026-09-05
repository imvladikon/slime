import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "glm53_flash_preflight.py"
SPEC = importlib.util.spec_from_file_location("glm53_flash_preflight", MODULE_PATH)
preflight = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def _stats(numel, training_bytes, checkpoint_bytes, *, rollout_bytes=None, bf16=0, fp32=0):
    return preflight.CategoryStats(
        parameter_numel=numel,
        training_bytes=training_bytes,
        checkpoint_bytes=checkpoint_bytes,
        rollout_bytes=checkpoint_bytes if rollout_bytes is None else rollout_bytes,
        training_numel_by_dtype={"BF16": bf16, "F32": fp32},
    )


def _official_categories():
    return {
        "regular_tp": _stats(
            8_625_539_200,
            17_251_639_808,
            14_727_285_248,
            rollout_bytes=14_733_969_920,
            bf16=8_625_258_496,
            fp32=280_704,
        ),
        "trainable_replicated": _stats(
            213_262_974,
            426_530_808,
            334_278_648,
            rollout_bytes=405_057_528,
            bf16=213_260_544,
            fp32=2_430,
        ),
        "routed_experts": _stats(
            304_405_807_104,
            608_811_614_208,
            304_480_124_928,
            bf16=304_405_807_104,
        ),
        "frozen_replicated": _stats(
            82_202_688,
            164_429_568,
            164_429_568,
            rollout_bytes=167_330_048,
        ),
        "vision": _stats(563_627_008, 1_127_254_016, 1_127_254_016),
    }


def _args(**overrides):
    values = dict(
        mode="rl",
        tp=8,
        pp=1,
        cp=1,
        ep=72,
        etp=1,
        dp=72,
        distributed_optimizer=True,
        bf16_optimizer_bytes=12,
        fp32_optimizer_bytes=8,
        optimizer_offload=False,
        offload_train=False,
        offload_rollout=False,
        rollout_weights_cpu_backup=False,
        colocate=True,
        allow_unsharded_remote_sync=False,
        rollout_tp=8,
        rollout_ep=8,
        rollout_moe_dp=1,
        rollout_num_gpus=576,
        rollout_gpus_per_engine=8,
        rollout_replicas=1,
        rollout_batch_size=8,
        n_samples_per_prompt=8,
        rollout_max_response_len=65536,
        use_rollout_routing_replay=True,
        actor_runtime_reserve_gib=24.0,
        rollout_runtime_reserve_gib=16.0,
        rollout_cache_reserve_gib=16.0,
        sglang_mem_fraction_static=0.54,
        sglang_moe_a2a_backend="deepep",
        sglang_deepep_mode="auto",
        gpu_memory_gib=141.0,
        memory_safety_fraction=0.9,
        backup_tags=1,
        gpus_per_node=8,
        host_memory_gib=512.0,
        aggregate_sync_bandwidth_gib_s=0.0,
        update_weight_buffer_bytes=512 * 1024**2,
        sglang_eplb=False,
        sglang_redundant_experts=False,
        sglang_elastic_ep=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _config():
    return {
        "text_config": {
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
        }
    }


def _inventory():
    size = 1_268_776_960
    return {
        "largest_training_tensor": {
            "name": "lm_head.weight",
            "bytes": size,
            "gib": preflight.gib(size),
        },
        "native_checkpoint_bytes": 328_326_771_576,
    }


def test_parameter_classifier_distinguishes_flash_state_families():
    assert preflight.classify_parameter("model.language_model.layers.3.mlp.experts.7.gate_proj.weight") == "routed_experts"
    assert preflight.classify_parameter("model.language_model.layers.3.self_attn.indexer.wk.weight") == "frozen_replicated"
    assert preflight.classify_parameter("model.language_model.layers.3.hc_attn_fn") == "trainable_replicated"
    assert preflight.classify_parameter("model.language_model.layers.3.self_attn.q_proj.weight") == "regular_tp"
    assert preflight.classify_parameter("model.visual.blocks.0.attn.proj.weight") == "vision"
    assert preflight.training_dtype("model.language_model.layers.0.hc_attn_fn", "BF16") == "BF16"
    assert preflight.rollout_dtype("model.language_model.layers.0.hc_attn_fn", "BF16") == "F32"
    assert preflight.training_dtype("model.language_model.layers.4.mlp.experts.0.up_proj.weight", "F8_E4M3") == "BF16"


def test_h200_colocated_floor_matches_official_metadata_calculation():
    report = preflight.calculate(_args(), _config(), _official_categories(), _inventory())

    assert report["verdict"] == "PASS_WITH_RUNTIME_GATES"
    assert report["actor"]["world_size"] == 576
    assert report["actor"]["expert_data_parallel_size"] == 8
    assert report["actor"]["parameter_gib_per_rank"] == pytest.approx(11.4835705)
    assert report["actor"]["optimizer_gpu_gib_per_rank"] == pytest.approx(6.1067084)
    assert report["actor"]["persistent_gpu_floor_gib_per_rank"] == pytest.approx(38.15132)
    assert report["topology"]["rollout"]["model_floor_gib_per_rank"] == pytest.approx(38.74433)
    assert report["topology"]["rollout"]["pre_model_available_gib_per_rank"] == pytest.approx(102.84868)
    assert report["topology"]["rollout"]["static_allocation_gib_per_rank"] == pytest.approx(55.53829)
    assert report["topology"]["rollout"]["cache_headroom_gib_per_rank"] == pytest.approx(16.79396)
    assert report["topology"]["rollout"]["runtime_slack_gib_per_rank"] == pytest.approx(47.31039)
    assert report["topology"]["rollout"]["r3"]["bytes_per_generated_token"] == 1440
    assert report["topology"]["rollout"]["r3"]["raw_upper_gib_per_rollout"] == pytest.approx(5.625)
    assert report["memory"]["projected_phase_peak_gib"] == pytest.approx(117.68961)
    assert report["sync"]["aggregate_target_write_tib_per_update"] == pytest.approx(21.203150)
    assert report["sync"]["expert_bf16_p2p_upper_tib_per_update"] == pytest.approx(39.8671875)
    assert report["sync"]["expert_bf16_p2p_if_one_local_target_tib_per_update"] == pytest.approx(39.3134766)
    assert report["sync"]["balanced_expert_egress_upper_gib_per_rank"] == pytest.approx(70.875)
    assert report["sync"]["balanced_expert_egress_upper_gib_per_node"] == pytest.approx(567.0)
    assert report["sync"]["dense_ipc_transient_gib_per_sending_rank"] == pytest.approx(2.36328125)
    assert report["sync"]["expert_staging_and_ipc_upper_gib_per_rank"] == pytest.approx(1.0)
    assert report["sync"]["expert_target_gib_per_layer_per_rank"] == pytest.approx(1.6875)
    assert report["sync"]["minimum_expert_transfer_batches_per_layer"] == 4
    assert report["sync"]["minimum_expert_transfer_batches_per_update"] == 168
    assert report["export"]["output_checkpoint_gib"] == pytest.approx(298.7994)
    assert report["export"]["model_only_torch_dist_source_gib"] == pytest.approx(583.6172)
    assert report["export"]["source_plus_output_disk_gib"] == pytest.approx(882.4166)
    assert report["export"]["full_adam_resume_estimate_tib"] == pytest.approx(3.98867)
    assert report["export"]["full_rank_zero_weight_gather"] is False
    assert report["export"]["nccl_collectives"] is False
    assert report["export"]["live_full_hf_export_supported"] is False
    assert report["rollout_initial_load"]["ideal_physical_read_tib"] == pytest.approx(21.500025)
    assert report["rollout_initial_load"]["unpartitioned_per_rank_read_upper_tib"] == pytest.approx(172.0002)


def test_h100_and_unqualified_distributed_seams_fail_closed():
    h100_report = preflight.calculate(
        _args(gpu_memory_gib=80.0), _config(), _official_categories(), _inventory()
    )
    report = preflight.calculate(
        _args(
            gpu_memory_gib=80.0,
            pp=2,
            cp=2,
            etp=2,
            colocate=False,
        ),
        _config(),
        _official_categories(),
        _inventory(),
    )

    errors = [gate["message"] for gate in report["gates"] if gate["severity"] == "error"]
    assert report["verdict"] == "FAIL"
    assert any("PP=1" in message for message in errors)
    assert any("CP=1" in message for message in errors)
    assert any("ETP=1" in message for message in errors)
    assert any("unsharded tensors" in message for message in errors)
    assert any(
        "safety budget" in gate["message"]
        for gate in h100_report["gates"]
        if gate["severity"] == "error"
    )


def test_colocated_train_offload_fails_before_cuda_ipc():
    report = preflight.calculate(
        _args(offload_train=True), _config(), _official_categories(), _inventory()
    )

    errors = [gate["message"] for gate in report["gates"] if gate["severity"] == "error"]
    assert report["verdict"] == "FAIL"
    assert any("torch-memory-saver" in message for message in errors)


def test_rollout_offload_and_expert_routing_fail_closed():
    report = preflight.calculate(
        _args(
            offload_rollout=True,
            rollout_ep=1,
            rollout_moe_dp=8,
            sglang_eplb=True,
        ),
        _config(),
        _official_categories(),
        _inventory(),
    )

    errors = [gate["message"] for gate in report["gates"] if gate["severity"] == "error"]
    assert any("frozen vision" in message for message in errors)
    assert any("EP>1" in message for message in errors)
    assert any("EPLB" in message for message in errors)


def test_rollout_static_pool_and_deepep_backend_fail_closed():
    report = preflight.calculate(
        _args(
            sglang_mem_fraction_static=0.30,
            sglang_moe_a2a_backend="none",
            sglang_deepep_mode="normal",
            use_rollout_routing_replay=False,
        ),
        _config(),
        _official_categories(),
        _inventory(),
    )

    errors = [gate["message"] for gate in report["gates"] if gate["severity"] == "error"]
    assert any("DeepEP MoE A2A" in message for message in errors)
    assert any("DeepEP auto or low_latency" in message for message in errors)
    assert any("R3 route-replay" in message for message in errors)
    assert any("rollout model floor" in message or "cache headroom" in message for message in errors)

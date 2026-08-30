from types import SimpleNamespace

from slime.backends.megatron_utils import initialize


def test_megatron_init_accepts_numpy_2(monkeypatch):
    calls = []
    monkeypatch.setattr(initialize.np, "__version__", "2.3.5")
    monkeypatch.setattr(initialize, "set_args", lambda _args: calls.append("set_args"))
    monkeypatch.setattr(initialize, "_initialize_distributed", lambda _args: calls.append("distributed"))
    monkeypatch.setattr(initialize, "_set_random_seed", lambda *_args: calls.append("seed"))
    monkeypatch.setattr(initialize, "_build_tokenizer", lambda _args: calls.append("tokenizer"))
    monkeypatch.setattr(
        initialize,
        "init_num_microbatches_calculator",
        lambda *_args: calls.append("microbatches"),
    )

    args = SimpleNamespace(
        enable_experimental=False,
        rank=0,
        seed=1234,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        rampup_batch_size=None,
        global_batch_size=2,
        micro_batch_size=1,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
        deterministic_mode=False,
        tp_comm_overlap=False,
        custom_megatron_init_path=None,
    )

    initialize.init(args)

    assert calls == ["set_args", "distributed", "seed", "tokenizer", "microbatches"]

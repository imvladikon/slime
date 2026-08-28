import torch

from slime_plugins.models.glm5_next.dsa import Glm5NextDSAAttention

NUM_GPUS = 0


def test_indexer_key_layernorm_preserves_bf16_parameter_contract():
    attention = Glm5NextDSAAttention.__new__(Glm5NextDSAAttention)
    torch.nn.Module.__init__(attention)
    attention.k_norm = torch.nn.LayerNorm(64, eps=1.0e-6, dtype=torch.bfloat16)

    index_k = torch.randn(7, 1, 64, dtype=torch.bfloat16)
    actual = attention._normalize_index_k(index_k)
    expected = torch.nn.functional.layer_norm(
        index_k.squeeze(1),
        attention.k_norm.normalized_shape,
        attention.k_norm.weight,
        attention.k_norm.bias,
        attention.k_norm.eps,
    )

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

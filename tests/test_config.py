import pytest
from entropix.config import ModelParams, LLAMA_1B_PARAMS

def test_model_params_structure():
    params = ModelParams(
        n_layers=12,
        n_local_heads=32,
        n_local_kv_heads=4,
        head_dim=64,
        max_seq_len=2048,
        rope_theta=10000.0,
        use_scaled_rope=True
    )
    assert params.n_layers == 12
    assert params.n_local_heads == 32
    assert params.n_local_kv_heads == 4
    assert params.head_dim == 64
    assert params.max_seq_len == 2048
    assert params.rope_theta == 10000.0
    assert params.use_scaled_rope == True

def test_llama_1b_params():
    assert isinstance(LLAMA_1B_PARAMS, ModelParams)
    assert LLAMA_1B_PARAMS.n_layers == 16
    assert LLAMA_1B_PARAMS.n_local_heads == 32
    assert LLAMA_1B_PARAMS.n_local_kv_heads == 8
    assert LLAMA_1B_PARAMS.head_dim == 64  # 2048 // 32
    assert LLAMA_1B_PARAMS.max_seq_len == 4096
    assert LLAMA_1B_PARAMS.rope_theta == 500000.0
    assert LLAMA_1B_PARAMS.use_scaled_rope == True
import pytest
import jax
import jax.numpy as jnp
from entropix.model import rms_norm, apply_rotary_emb, attention, feed_forward, xfmr
from entropix.config import LLAMA_1B_PARAMS
from entropix.weights import XfmrWeights, LayerWeights
from entropix.kvcache import KVCache

@pytest.fixture
def dummy_weights():
    return XfmrWeights(
        tok_embeddings=jnp.ones((100, 32)),
        norm=jnp.ones(32),
        output=jnp.ones((100, 32)),
        layer_weights=[LayerWeights(
            wq=jnp.ones((32, 32)),
            wk=jnp.ones((32, 32)),
            wv=jnp.ones((32, 32)),
            wo=jnp.ones((32, 32)),
            w1=jnp.ones((32, 32)),
            w2=jnp.ones((32, 32)),
            w3=jnp.ones((32, 32)),
            ffn_norm=jnp.ones(32),
            attention_norm=jnp.ones(32)
        )]
    )

def test_rms_norm():
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = jnp.array([0.1, 0.2, 0.3])
    result = rms_norm(x, w)
    assert result.shape == x.shape

def test_apply_rotary_emb():
    xq = jnp.ones((1, 1, 4, 32))
    xk = jnp.ones((1, 1, 4, 32))
    freqs_cis = jnp.ones((1, 16), dtype=jnp.complex64)
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape

def test_attention(dummy_weights):
    x = jnp.ones((1, 4, 32))
    freqs_cis = jnp.ones((4, 16), dtype=jnp.complex64)
    kvcache = KVCache.new(1, 1, 4, 1, 32)
    out, new_kvcache = attention(x, dummy_weights.layer_weights[0], LLAMA_1B_PARAMS, 0, 0, freqs_cis, kvcache)
    assert out.shape == x.shape
    assert isinstance(new_kvcache, KVCache)

def test_feed_forward(dummy_weights):
    x = jnp.ones((1, 4, 32))
    out = feed_forward(x, dummy_weights.layer_weights[0])
    assert out.shape == x.shape

def test_xfmr(dummy_weights):
    tokens = jnp.array([[1, 2, 3, 4]])
    freqs_cis = jnp.ones((4, 16), dtype=jnp.complex64)
    kvcache = KVCache.new(1, 1, 4, 1, 32)
    logits, new_kvcache = xfmr(dummy_weights, LLAMA_1B_PARAMS, tokens, 0, freqs_cis, kvcache)
    assert logits.shape == (1, 4, 100)  # Assuming vocab size of 100
    assert isinstance(new_kvcache, KVCache)
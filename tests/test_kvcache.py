import pytest
import jax.numpy as jnp
from entropix.kvcache import KVCache

def test_kvcache_new():
    kvcache = KVCache.new(layers=2, bsz=1, max_seq_len=10, kv_heads=4, head_dim=32)
    assert kvcache.k.shape == (2, 1, 10, 4, 32)
    assert kvcache.v.shape == (2, 1, 10, 4, 32)
    assert kvcache.k.dtype == jnp.bfloat16
    assert kvcache.v.dtype == jnp.bfloat16

def test_kvcache_update():
    kvcache = KVCache.new(layers=2, bsz=1, max_seq_len=10, kv_heads=4, head_dim=32)
    xk = jnp.ones((1, 1, 4, 32), dtype=jnp.float32)
    xv = jnp.ones((1, 1, 4, 32), dtype=jnp.float32)
    
    keys, values, new_kvcache = kvcache.update(xk, xv, layer_idx=0, cur_pos=0, n_rep=2)
    
    assert keys.shape == (1, 1, 8, 32)  # n_rep = 2, so 4 * 2 = 8
    assert values.shape == (1, 1, 8, 32)
    assert new_kvcache.k.shape == kvcache.k.shape
    assert new_kvcache.v.shape == kvcache.v.shape
    assert jnp.any(new_kvcache.k != kvcache.k)  # Check if the cache was updated
    assert jnp.any(new_kvcache.v != kvcache.v)

def test_kvcache_update_multiple_positions():
    kvcache = KVCache.new(layers=2, bsz=1, max_seq_len=10, kv_heads=4, head_dim=32)
    xk = jnp.ones((1, 2, 4, 32), dtype=jnp.float32)
    xv = jnp.ones((1, 2, 4, 32), dtype=jnp.float32)
    
    keys, values, new_kvcache = kvcache.update(xk, xv, layer_idx=0, cur_pos=0, n_rep=2)
    keys, values, final_kvcache = new_kvcache.update(xk, xv, layer_idx=0, cur_pos=2, n_rep=2)
    
    assert keys.shape == (1, 4, 8, 32)  # 2 positions, n_rep = 2, so 4 * 2 = 8
    assert values.shape == (1, 4, 8, 32)
    assert jnp.any(final_kvcache.k != kvcache.k)
    assert jnp.any(final_kvcache.v != kvcache.v)
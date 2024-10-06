import jax
import jax.numpy as jnp
from entropix.config import ModelParams
from typing import NamedTuple


class KVCache(NamedTuple):
    k: jax.Array
    v: jax.Array
    is_new: bool

    @classmethod
    def new(cls, model_params: ModelParams, bsz: int) -> 'KVCache':
      kv_heads = model_params.n_local_kv_heads
      layers = model_params.n_layers
      head_dim = model_params.head_dim
      max_seq_len = model_params.max_seq_len
      return cls(
          k=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
          v=jnp.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=jnp.bfloat16),
          is_new=True
    )

    def update(self, xk: jax.Array, xv: jax.Array, layer_idx: int, cur_pos: int, n_rep: int):
      ck = jax.lax.dynamic_update_slice(self.k, jnp.bfloat16(xk[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
      cv = jax.lax.dynamic_update_slice(self.v, jnp.bfloat16(xv[None, ...]), (layer_idx, 0, cur_pos, 0, 0))
      if cur_pos == 0:
        keys = jnp.repeat(xk, n_rep, axis=2)
        values = jnp.repeat(xv, n_rep, axis=2)
      else:
        keys = jnp.repeat(ck[layer_idx], n_rep, axis=2)
        values = jnp.repeat(cv[layer_idx], n_rep, axis=2)

      return keys, values, KVCache(k=ck, v=cv, is_new=False)


from typing import Tuple, NamedTuple
import jax.numpy as jnp
import jax
import math
from entropix.config import RopeParams

#@partial(jax.jit, static_argnames=("dtype"))
def apply_rotary_emb(xq: jax.Array, xk: jax.Array, freqs_cis: jax.Array, dtype: jnp.dtype = jnp.float32) -> Tuple[jax.Array, jax.Array]:
  reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
  reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)
  xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
  xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
  xq_out = xq_ * freqs_cis[None, :, None, :]
  xk_out = xk_ * freqs_cis[None, :, None, :]
  xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
  xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
  return xq_out.astype(dtype), xk_out.astype(dtype)

def precompute_freqs_cis(head_dim: int, max_seq_len: int, rope_params: RopeParams, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  freqs = 1.0 / (rope_params.rope_theta ** (jnp.arange(0, head_dim, 2)[: (head_dim // 2)].astype(dtype) / head_dim))
  if rope_params.use_scaled_rope:
    freqs = _apply_scaling(rope_params, freqs)
  t = jnp.arange(max_seq_len, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)

def _apply_scaling(rope_params: RopeParams, freqs: jax.Array):
  scale_factor = rope_params.scale_factor
  low_freq_factor = rope_params.low_freq_factor
  high_freq_factor = rope_params.high_freq_factor
  old_context_len = rope_params.old_context_len  # original llama3 length

  low_freq_wavelen = old_context_len / low_freq_factor
  high_freq_wavelen = old_context_len / high_freq_factor

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq

    def scale_mid(_):
      smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
      return (1 - smooth) * freq / scale_factor + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / scale_factor, scale_mid, None),
      None
    )
  return jax.vmap(scale_freq)(freqs)

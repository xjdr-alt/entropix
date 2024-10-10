
from typing import Tuple
import jax.numpy as jnp
import jax
import math
from entropix.config import ModelParams, ScaledRopeParams

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

def precompute_freqs_cis(model_params: ModelParams, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  dim = model_params.head_dim
  freqs = 1.0 / (model_params.rope_theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if model_params.use_scaled_rope:
    freqs = apply_scaling(model_params.scaled_rope_params, freqs)
  freqs = jnp.outer(jnp.arange(model_params.max_seq_len, dtype=dtype), freqs)
  return jnp.exp(1j * freqs)

def apply_scaling(scaled_rope_params: ScaledRopeParams, freqs: jax.Array):
  scale_factor = scaled_rope_params.scale_factor
  low_freq_factor = scaled_rope_params.low_freq_factor
  high_freq_factor = scaled_rope_params.high_freq_factor
  old_context_len = scaled_rope_params.old_context_len  # original llama3 length

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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from functools import partial

from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.stats import AttnStats
from entropix.weights import XfmrWeights, LayerWeights


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


@partial(jax.jit, static_argnames=("eps"))
def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
  return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))


@partial(jax.jit, static_argnames=("dtype"))
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

@partial(jax.jit, static_argnames=("model_params", "cur_pos", "layer_idx"))
def attention(x: jax.Array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache]:
  bsz, _, _ = x.shape
  n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
  xq = jnp.dot(x, layer_weights.wq.T).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
  xk = jnp.dot(x, layer_weights.wk.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xv = jnp.dot(x, layer_weights.wv.T).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
  keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
  xq = jnp.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
  keys = jnp.transpose(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
  values = jnp.transpose(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
  scores = jnp.matmul(xq, keys)
  pre_scores = scores / jnp.sqrt(model_params.head_dim)
  scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32
  if cur_pos == 0:
    scores = scores + attn_mask
  mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
  padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
  scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
  output = jnp.matmul(scores, values)
  output = jnp.swapaxes(output, 1, 2).reshape(xq.shape[0], xq.shape[2], -1)
  out = jnp.dot(output, layer_weights.wo.T)
  return out, kvcache, pre_scores

@partial(jax.jit)
def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
 return jnp.dot(jax.nn.silu(jnp.dot(x, layer_weights.w1.T)) * jnp.dot(x, layer_weights.w3.T), layer_weights.w2.T)

@partial(jax.jit, static_argnames=("model_params", "cur_pos"))
def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: jax.Array, cur_pos: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array]=None) -> Tuple[jax.Array, KVCache]:
  h = xfmr_weights.tok_embeddings[tokens]
  attn_stats = AttnStats.new(
    bsz=tokens.shape[0],
    n_layers=model_params.n_layers,
    n_heads=model_params.n_local_heads
  )
  for i in range(model_params.n_layers):
    norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
    h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
    attn_stats = attn_stats.update(scores[:,:,-1,:], i)
    h = h + h_attn
    h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
  logits = jnp.dot(rms_norm(h, xfmr_weights.norm), xfmr_weights.output.T)
  return logits, kvcache, scores, attn_stats

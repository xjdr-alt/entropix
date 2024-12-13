from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from functools import partial

from entropix.config import ModelParams
from entropix.kvcache import KVCache
from entropix.weights import XfmrWeights, LayerWeights
from jax.sharding import PartitionSpec as PS
from jax.experimental.pallas.ops.gpu.rms_norm import rms_norm as pl_rms_norm

shard = jax.lax.with_sharding_constraint

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def rms_norm(x: jax.Array, w: jax.Array, eps: float = 1e-6) -> jax.Array:
  x = shard(x, PS())
  return w * (x * jax.lax.rsqrt(jax.lax.pow(x, 2).mean(-1, keepdims=True) + eps))

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

def attention(x: jax.Array, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, KVCache]:
  bsz, _, _ = x.shape
  n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
  xq = jnp.einsum('...e,enh->...nh', x, layer_weights.wq)
  xk = jnp.einsum('...e,enh->...nh', x, layer_weights.wk)
  xv = jnp.einsum('...e,enh->...nh', x, layer_weights.wv)
  xq, xk = jax.vmap(lambda q, k: apply_rotary_emb(q, k, freqs_cis=freqs_cis), in_axes=(0, 0))(xq, xk)
  xq = xq.reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
  xk = xk.reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
  scores = jnp.einsum('...qnh,...knh->...nqk', xq, keys)
  pre_scores = scores / jnp.sqrt(model_params.head_dim)
  scores = pre_scores.astype(jnp.float32)  # Always do attention softmax at float32
  if attn_mask is not None:
    scores = scores.at[..., :attn_mask.shape[-1]].add(attn_mask)
  mask = jnp.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
  padded_logits = jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
  scores = jax.nn.softmax(padded_logits, axis=-1).astype(x.dtype)
  output = jnp.einsum('...nqk,...knh->...qnh', scores, values)
  output = output.reshape((output.shape[0], output.shape[1], -1))
  out = shard(jnp.dot(output, layer_weights.wo), PS())
  return out, kvcache, pre_scores

def feed_forward(x: jax.Array, layer_weights: LayerWeights) -> jax.Array:
 x = shard(x, PS())
 h1 = jax.nn.silu(shard(jnp.dot(x, layer_weights.w1), PS(None, None, 'mp')))
 h =  h1 * shard(jnp.dot(x, layer_weights.w3), PS(None, None, 'mp'))
 return shard(jnp.dot(h, layer_weights.w2), PS())


def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: jax.Array, cur_pos: int, freqs_cis: jax.Array, kvcache: KVCache, attn_mask: Optional[jax.Array]=None) -> Tuple[jax.Array, KVCache]:
  def step_fn(inp, wi):
    h, kvcache = inp
    w, i = wi
    norm_x = rms_norm(h, w.attention_norm)
    h_attn, kvcache, scores = attention(norm_x, w, model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
    h = h + h_attn
    h = h + feed_forward(rms_norm(h, w.ffn_norm), w)
    return (h, kvcache), None

  x = xfmr_weights.tok_embeddings[tokens]
  (x, kvcache), _ = jax.lax.scan(step_fn, (x, kvcache), (jax.tree_util.tree_map(lambda *x: jnp.stack(x, 0), *xfmr_weights.layer_weights), jnp.arange(model_params.n_layers)), model_params.n_layers)
  logits = jnp.dot(rms_norm(x, xfmr_weights.norm), xfmr_weights.output.T)
  return logits, kvcache, None


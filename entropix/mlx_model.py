import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from entropix.config import ModelParams
from entropix.mlx_kvcache import KVCache
from entropix.mlx_stats import AttnStats
from entropix.weights import XfmrWeights, LayerWeights

DEFAULT_MASK_VALUE = -1e9


def rms_norm(x: mx.array, w: mx.array, eps: float = 1e-6) -> mx.array:
  return w * (x * mx.rsqrt(mx.power(x, 2).mean(-1, keepdims=True) + eps))


def apply_rotary_emb(
  xq: mx.array, xk: mx.array, freqs_cis: mx.array, dtype: mx.Dtype = mx.float32
) -> Tuple[mx.array, mx.array]:
  def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)

  # Reshape freqs_cis to match the dimensions of xq and xk
  freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])

  # Expand dimensions for broadcasting
  xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
  xk_r, xk_i = xk[..., ::2], xk[..., 1::2]

  # Compute cos and sin
  freqs_cos = mx.cos(freqs_cis)
  freqs_sin = mx.sin(freqs_cis)

  # Apply rotary embeddings
  xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
  xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
  xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
  xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

  # Combine real and imaginary parts
  xq_out = mx.stack([xq_out_r, xq_out_i], axis=-1).reshape(xq.shape)
  xk_out = mx.stack([xk_out_r, xk_out_i], axis=-1).reshape(xk.shape)

  return xq_out.astype(dtype), xk_out.astype(dtype)


def attention(
  x: mx.array,
  layer_weights: LayerWeights,
  model_params,
  cur_pos: int,
  layer_idx: int,
  freqs_cis: mx.array,
  kvcache: KVCache,
  attn_mask: Optional[mx.array] = None,
) -> Tuple[mx.array, KVCache, mx.array]:
  try:
    bsz, seqlen, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
    xq = (x @ layer_weights.wq.T).reshape(
      bsz, seqlen, model_params.n_local_heads, model_params.head_dim
    )
    xk = (x @ layer_weights.wk.T).reshape(
      bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim
    )
    xv = (x @ layer_weights.wv.T).reshape(
      bsz, seqlen, model_params.n_local_kv_heads, model_params.head_dim
    )
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = mx.transpose(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
    keys = mx.transpose(
      keys, (0, 2, 3, 1)
    )  # (bs, n_heads, head_dim, cache_len + seqlen)
    values = mx.transpose(
      values, (0, 2, 1, 3)
    )  # (bs, n_heads, cache_len + seqlen, head_dim)
    scores = xq @ keys
    pre_scores = scores / mx.sqrt(model_params.head_dim)
    scores = pre_scores.astype(mx.float32)  # Always do attention softmax at float32
    if cur_pos == 0:
      scores = scores + attn_mask
    mask = mx.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = mx.where(
      (mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE
    )
    scores = nn.softmax(padded_logits, axis=-1).astype(x.dtype)
    output = scores @ values
    output = mx.swapaxes(output, 1, 2).reshape(xq.shape[0], xq.shape[2], -1)
    out = output @ layer_weights.wo.T
    return out, kvcache, pre_scores
  except Exception as e:
    print(f"Error in attention function: {str(e)}")
    print(f"Error occurred at layer_idx: {layer_idx}, cur_pos: {cur_pos}")
    raise


def feed_forward(x: mx.array, layer_weights: LayerWeights) -> mx.array:
  return (
    nn.silu(x @ layer_weights.w1.T) * (x @ layer_weights.w3.T)
  ) @ layer_weights.w2.T


def xfmr(
  xfmr_weights: XfmrWeights,
  model_params: ModelParams,
  tokens: mx.array,
  cur_pos: int,
  freqs_cis: mx.array,
  kvcache: KVCache,
  attn_mask: Optional[mx.array] = None,
) -> Tuple[mx.array, KVCache, mx.array, AttnStats]:
  h = xfmr_weights.tok_embeddings[tokens]
  attn_stats = AttnStats.new(
    bsz=tokens.shape[0],
    n_layers=model_params.n_layers,
    n_heads=model_params.n_local_heads,
  )
  for i in range(model_params.n_layers):
    norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
    h_attn, kvcache, scores = attention(
      norm_x,
      xfmr_weights.layer_weights[i],
      model_params,
      cur_pos,
      i,
      freqs_cis,
      kvcache,
      attn_mask=attn_mask,
    )
    attn_stats = attn_stats.update(scores[:, :, -1, :], i)
    h = h + h_attn
    h = h + feed_forward(
      rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i]
    )
  logits = rms_norm(h, xfmr_weights.norm) @ xfmr_weights.output.T
  return logits, kvcache, scores, attn_stats

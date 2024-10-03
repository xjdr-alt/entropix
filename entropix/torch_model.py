from typing import Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import XfmrWeights, LayerWeights

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache]:
  bsz, _, _ = x.shape
  n_rep = model_params.n_local_heads // model_params.n_local_kv_heads
  xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
  xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
  xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
  keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
  xq = torch.permute(xq, (0, 2, 1, 3))  # (bs, n_heads, seqlen, head_dim)
  keys = torch.permute(keys, (0, 2, 3, 1))  # (bs, n_heads, head_dim, cache_len + seqlen)
  values = torch.permute(values, (0, 2, 1, 3))  # (bs, n_heads, cache_len + seqlen, head_dim)
  scores = torch.matmul(xq, keys)
  scores = scores / math.sqrt(model_params.head_dim)
  scores = scores.to(torch.float32)  # Always do attention softmax at float32
  if cur_pos == 0:
    scores = scores + attn_mask
  mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
  padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
  scores = F.softmax(padded_logits, dim=-1).type_as(x)
  output = torch.matmul(scores, values)
  output = output.transpose(1, 2).reshape(xq.shape[0], xq.shape[2], -1)
  out = F.linear(output, layer_weights.wo)
  return out, kvcache

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
 return F.linear(F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3), layer_weights.w2)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache]:
  h = xfmr_weights.tok_embeddings[tokens]
  for i in range(model_params.n_layers):
    norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
    h_attn, kvcache = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
    h = h + h_attn
    h = h + feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
  logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
  return logits, kvcache


# torch_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from entropix.config import ModelParams
from entropix.torch_kvcache import KVCache
from entropix.torch_weights import XfmrWeights, LayerWeights
from entropix.torch_stats import AttnStats

DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

# Device selection: prefer Apple Silicon, then CUDA, fallback is CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#print(f"Using device: {device}")

from typing import Tuple, Optional

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return w * (x * torch.rsqrt(torch.pow(x, 2).mean(-1, keepdim=True) + eps))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, dtype: torch.dtype = torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
    reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_ = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    xq_out = xq_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xk_out = xk_ * freqs_cis.unsqueeze(0).unsqueeze(2)
    xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.to(dtype), xk_out.to(dtype)

def sphere_pack_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    Apply sphere packing to embeddings.

    Pros:
    - Reduces correlation between vectors by maximizing distances.
    - May enhance the attention mechanism's ability to distinguish between different inputs.

    Cons:
    - May interfere with pre-trained representations, potentially degrading performance.
    - Adds computational overhead during inference.
    """
    # Flatten the last dimension for normalization
    original_shape = x.shape
    x_flat = x.view(-1, original_shape[-1])

    # Normalize to lie on the unit sphere
    x_norm = x_flat / torch.norm(x_flat, dim=-1, keepdim=True).clamp(min=1e-8)  # Avoid division by zero

    # Optionally apply scaling to spread out the vectors (commented out)
    # num_vectors = x_flat.size(0)
    # scale_factors = torch.linspace(0.8, 1.2, num_vectors).unsqueeze(1).to(x.device)
    # x_packed = x_norm * scale_factors

    # Reshape back to original dimensions
    x_packed = x_norm.view(original_shape)
    return x_packed

def attention(x: torch.Tensor, layer_weights: LayerWeights, model_params, cur_pos: int, layer_idx: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, KVCache, torch.Tensor]:
    bsz, _, _ = x.shape
    n_rep = model_params.n_local_heads // model_params.n_local_kv_heads

    # Compute queries, keys, and values
    xq = F.linear(x, layer_weights.wq).reshape(bsz, -1, model_params.n_local_heads, model_params.head_dim)
    xk = F.linear(x, layer_weights.wk).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)
    xv = F.linear(x, layer_weights.wv).reshape(bsz, -1, model_params.n_local_kv_heads, model_params.head_dim)

    # Apply rotary embeddings
    xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

    # Apply sphere packing to queries and keys
    # Pros:
    # - Enhances distinctiveness of queries and keys.
    # Cons:
    # - May disrupt learned representations if the model is pre-trained.
    xq = sphere_pack_embeddings(xq)
    xk = sphere_pack_embeddings(xk)

    # Continue with attention mechanism
    keys, values, kvcache = kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)
    xq = torch.permute(xq, (0, 2, 1, 3))  # (bsz, n_heads, seqlen, head_dim)
    keys = torch.permute(keys, (0, 2, 3, 1))  # (bsz, n_heads, head_dim, cache_len + seqlen)
    values = torch.permute(values, (0, 2, 1, 3))  # (bsz, n_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys)
    pre_scores = scores / math.sqrt(model_params.head_dim)
    scores = pre_scores.to(torch.float32)  # Always do attention softmax at float32

    if cur_pos == 0:
        scores = scores + attn_mask
    mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
    padded_logits = torch.where((mask >= DEFAULT_MASK_VALUE * 0.5), scores, DEFAULT_MASK_VALUE)
    scores = F.softmax(padded_logits, dim=-1).to(torch.float32)
    output = torch.matmul(scores, values)
    output = output.transpose(1, 2).reshape(bsz, -1, model_params.n_local_heads * model_params.head_dim)
    out = F.linear(output, layer_weights.wo)
    return out, kvcache, pre_scores

def feed_forward(x: torch.Tensor, layer_weights: LayerWeights) -> torch.Tensor:
    h = F.silu(F.linear(x, layer_weights.w1)) * F.linear(x, layer_weights.w3)
    return F.linear(h, layer_weights.w2)

def xfmr(xfmr_weights: XfmrWeights, model_params: ModelParams, tokens: torch.Tensor, cur_pos: int, freqs_cis: torch.Tensor, kvcache: KVCache, attn_mask: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, KVCache, torch.Tensor, AttnStats]:
    h = xfmr_weights.tok_embeddings[tokens]
    attn_stats = AttnStats.new(
        bsz=tokens.shape[0],
        n_layers=model_params.n_layers,
        n_heads=model_params.n_local_heads
    )
    for i in range(model_params.n_layers):
        norm_x = rms_norm(h, xfmr_weights.layer_weights[i].attention_norm)
        h_attn, kvcache, scores = attention(norm_x, xfmr_weights.layer_weights[i], model_params, cur_pos, i, freqs_cis, kvcache, attn_mask=attn_mask)
        attn_stats = attn_stats.update(scores[:, :, -1, :], i)
        h = h + h_attn
        h_ffn = feed_forward(rms_norm(h, xfmr_weights.layer_weights[i].ffn_norm), xfmr_weights.layer_weights[i])
        h = h + h_ffn
    logits = F.linear(rms_norm(h, xfmr_weights.norm), xfmr_weights.output)
    return logits, kvcache, scores, attn_stats

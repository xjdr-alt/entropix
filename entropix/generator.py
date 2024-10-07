from entropix.model import KVCache
from entropix.rope import precompute_freqs_cis
from entropix.sampler import sample
from entropix.stats import AttnStats
from entropix.sampler import SamplerConfig
from entropix.model import xfmr
import jax.numpy as jnp
import jax

def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


def _initialize(model_params, tokens, max_gen_len):    
    tokens = jnp.array([tokens], dtype=jnp.int32)
    bsz, seqlen = tokens.shape
    max_total_len=seqlen + max_gen_len
    attn_mask = build_attn_mask(seqlen, 0)
    freqs_cis = precompute_freqs_cis(model_params)
    kvcache = KVCache.new(model_params, bsz, max_total_len)
    attn_stats = AttnStats.new(model_params, bsz, max_total_len)
    stop = jnp.array([128001, 128008, 128009], dtype=jnp.int32)
    sampler_cfg = SamplerConfig()
    return {
        'tokens': tokens,
        'kvcache': kvcache,
        'attn_stats': attn_stats,
        'freqs_cis': freqs_cis,
        'attn_mask': attn_mask,
        'stop_tokens': stop,
        'sampler_cfg': sampler_cfg,
    }

def generate(xfmr_weights, model_params, tokenizer, tokens, max_gen_len):
    initial_state = _initialize(model_params, tokens, max_gen_len)

    kvcache = initial_state['kvcache']
    attn_stats = initial_state['attn_stats']
    attn_mask = initial_state['attn_mask']
    freqs_cis = initial_state['freqs_cis']
    stop_tokens = initial_state['stop_tokens']
    sampler_cfg = initial_state['sampler_cfg']
    tokens = initial_state['tokens']
    
    prompt_len = tokens.shape[1]

    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, 0, freqs_cis[:prompt_len], kvcache, attn_stats, attn_mask=attn_mask)
    cur_pos, max_total_len = prompt_len, prompt_len + max_gen_len
    
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    while cur_pos < max_total_len:
      cur_pos += 1
      logits, kvcache, scores, attn_stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache, attn_stats)
      next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, stop_tokens).any():
        break
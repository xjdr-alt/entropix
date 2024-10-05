from entropix.model import KVCache
from entropix.rope import precompute_freqs_cis
from entropix.sampler import sample
from entropix.LMState import LMState
from entropix.model import xfmr
from entropix.sampler import SamplerParams
from entropix.tokenizer import Tokenizer
from entropix.config import ModelParams, RopeParams
import jax.numpy as jnp
import jax

def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask

def generate(xfmr_weights, model_params, sampler_params, tokenizer, tokens):
    tokens = jnp.array([tokens], jnp.int32)
    n_words = tokenizer.n_words
    bsz, seqlen = tokens.shape
    lm_state = LMState(
      prompt=tokens,
      logits=jnp.zeros((bsz, n_words), dtype=jnp.bfloat16),
      freqs_cis=precompute_freqs_cis(head_dim=model_params.head_dim, max_seq_len=model_params.max_seq_len, rope_params=model_params.rope_params),
      kvcache=KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim),
      attn_mask=build_attn_mask(seqlen, 0),
      gen_tokens=jnp.zeros((bsz, 0), dtype=jnp.int32),
      state=jnp.zeros((bsz, 1), dtype=jnp.int32),
      pos=0
    )
    lm_state.logits, lm_state.kvcache, _ = xfmr(xfmr_weights, model_params, lm_state.prompt, lm_state.pos, freqs_cis=lm_state.freqs_cis[:seqlen], kvcache=lm_state.kvcache, attn_mask=lm_state.attn_mask)
    next_token = jnp.argmax(lm_state.logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    lm_state.gen_tokens, lm_state.pos = jnp.concatenate((lm_state.gen_tokens, next_token), axis=1), seqlen
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    #stop = jnp.array(tokenizer.stop_tokens)
    while lm_state.pos < 2048:
      lm_state.pos += 1
      lm_state.logits, lm_state.kvcache, _ = xfmr(xfmr_weights, model_params, next_token, lm_state.pos, lm_state.freqs_cis[lm_state.pos:lm_state.pos+1], lm_state.kvcache)
      next_token = sample(sampler_params, lm_state)
      lm_state.gen_tokens = jnp.concatenate((lm_state.gen_tokens, next_token), axis=1)
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, sampler_params.stop_tokens).any():
        break



from entropix.rope import precompute_freqs_cis
from entropix.sampler import sample
from entropix.lm_state import LMState
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerParams
import jax.numpy as jnp
import jax

def generate(xfmr_weights, model_params, sampler_params, tokenizer, prompts, gen_len):
    tokens = jnp.array([prompts], jnp.int32)
    bsz, prompt_len = tokens.shape
    freqs_cis=precompute_freqs_cis(model_params)
    lm_state = LMState.new(model_params, bsz, gen_len)
    kvcache = KVCache.new(model_params)
    # remove initial attn mask for now bc head entropy of context might be useful! for now we pay more in processing of prompt.
    logits, kvcache, _, attn_stats = xfmr(xfmr_weights, model_params, lm_state.prompt, lm_state.pos, freqs_cis=freqs_cis[:prompt_len], kvcache=lm_state.kvcache) 
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    lm_state.update(next_token, logits, kvcache, attn_stats)
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    #stop = jnp.array(tokenizer.stop_tokens)
    while lm_state.pos < prompt_len + gen_len:
      lm_state.pos += 1
      logits, kvcache, _, attn_stats = xfmr(xfmr_weights, model_params, next_token, lm_state.pos, lm_state.freqs_cis[lm_state.pos:lm_state.pos+1], lm_state.kvcache)
      next_token = sample(sampler_params, lm_state)
      lm_state.update(next_token, logits, kvcache, attn_stats)
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, sampler_params.stop_tokens).any():
        break




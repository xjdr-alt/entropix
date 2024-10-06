import jax
import jax.numpy as jnp
import tyro

from entropix.config import LLAMA_1B_PARAMS
from entropix.lm_state import LMState
from entropix.model import xfmr
from entropix.prompts import prompt4 as prompt
from entropix.sampler import sample, SamplerParams
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.kvcache import KVCache



def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask

def main():
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights()
  tokenizer = Tokenizer("entropix/tokenizer.model")
  raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  # base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')
  sampler_params = SamplerParams(
    steer_tokens=jnp.load('data/STEER_TOKENS.npy'),
    temp=0.66,
    top_k=40,
    top_p=0.9,
    min_p=0.01 # turn down to 0.01 to reduce shoggoth symptoms
  )
  # Create the batch of tokens
  def generate(xfmr_weights, model_params, sampler_params, tokens, gen_len):
    tokens = jnp.array([tokens], jnp.int32)
    bsz, prompt_len = tokens.shape
    attn_mask = build_attn_mask(prompt_len, 0)
    print(attn_mask)
    lm_state = LMState.new(model_params, tokens, gen_len)
    kvcache = KVCache.new(model_params, bsz)
    logits, kvcache, lm_state, _ = xfmr(xfmr_weights, model_params, lm_state, kvcache=kvcache, attn_mask=attn_mask) 
    next_token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
    lm_state = lm_state.update_context(next_token, logits[:,-1,:])
    print(tokenizer.decode(next_token.tolist()), end='', flush=True)
    stop = jnp.array([128001, 128008, 128009])
    #stop = jnp.array(tokenizer.stop_tokens)
    while lm_state.cur_pos < prompt_len + gen_len:
      logits, kvcache, lm_state, scores = xfmr(xfmr_weights, model_params, lm_state, kvcache)
      # next_token = sample(sampler_params, lm_state.context[:,lm_state.cur_pos], logits, scores).reshape(1)
      next_token = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
      lm_state = lm_state.update_context(next_token, logits)
      print(tokenizer.decode(next_token), end='', flush=True)
      if jnp.isin(next_token, stop).any():
        break


  generate(xfmr_weights, model_params, sampler_params, raw_tokens1, 1000)

if __name__ == '__main__':
  tyro.cli(main)
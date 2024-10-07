import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro


from pathlib import Path
from functools import partial
from entropix.stats import AttnStats

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.rope import precompute_freqs_cis

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(weights_path)

  tokenizer = Tokenizer('entropix/tokenizer.model')
  raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')

  # Create the batch of tokens
  def generate(xfmr_weights, model_params, tokens, max_gen_len):
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    max_total_len=seqlen + max_gen_len
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params)
    kvcache = KVCache.new(model_params, bsz, max_total_len)
    attn_stats = AttnStats.new(model_params, bsz, max_total_len)
    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_stats, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    sampler_cfg = SamplerConfig()
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, attn_stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache, attn_stats)
      next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, stop).any():
        break
      
  csv_path = Path('entropix/data/prompts.csv')
  prompts = create_prompts_from_csv(csv_path)
  PROMPT_TEST = False

  if PROMPT_TEST:
    for p in prompts:
      print(p)
      tokens = tokenizer.encode(p,  bos=False, eos=False, allowed_special='all')
      generate(xfmr_weights, model_params, tokens, 100)
  else:
    print(prompt)
    tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokens, 100)

if __name__ == '__main__':
  tyro.cli(main)
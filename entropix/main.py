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
from entropix.generator import generate


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
  csv_path = Path('entropix/data/prompts.csv')
  prompts = create_prompts_from_csv(csv_path)
  PROMPT_TEST = False

  if PROMPT_TEST:
    for p in prompts:
      print(p)
      tokens = tokenizer.encode(p,  bos=False, eos=False, allowed_special='all')
      generate(xfmr_weights, model_params, tokenizer, tokens, 100)
  else:
    print(prompt)
    tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokenizer, tokens, 100)

if __name__ == '__main__':
  tyro.cli(main)
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.extend.backend
import tyro

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

def apply_scaling(freqs: jax.Array):
  SCALE_FACTOR = 8
  LOW_FREQ_FACTOR = 1
  HIGH_FREQ_FACTOR = 4
  OLD_CONTEXT_LEN = 8192  # original llama3 length

  low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
  high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq

    def scale_mid(_):
      smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
      return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
      None
    )

  return jax.vmap(scale_freq)(freqs)

def get_backend_name():
    return jax.extend.backend.get_backend().platform

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array: 
    backend = get_backend_name()
    
    if backend == 'METAL':  # Apple silicon(e) :P
        # Metal implementation (avoid complex numbers)
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(end)
        freqs = jnp.outer(t, freqs)
        return jnp.stack([jnp.cos(freqs), jnp.sin(freqs)], axis=-1)
    
    elif backend in ['gpu', 'tpu', 'cpu']:
        # GPU/TPU implementation (supports complex numbers)
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(end)
        freqs = jnp.outer(t, freqs)
        return jnp.exp(1j * freqs)
    
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct')):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(weights_path.absolute())
  tokenizer = Tokenizer('entropix/tokenizer.model')

  # Create the batch of tokens
  def generate(xfmr_weights, model_params, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    sampler_cfg = SamplerConfig()
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
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
      generate(xfmr_weights, model_params, tokens)
  else:
    print(prompt)
    tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokens)

if __name__ == '__main__':
  tyro.cli(main)

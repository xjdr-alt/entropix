import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Generic

import jax
import jax.numpy as jnp
import tyro

from entropix.config import LLAMA_1B_PARAMS, ModelParams
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.samplers import ST, Cfg_contra, EntropySampler
from entropix.samplers.baseline_sampler import SamplerConfig as BaselineSamplerConfig
from entropix.samplers.baseline_sampler import sample as baseline_sampler
from entropix.tokenizer import Tokenizer
from entropix.weights import XfmrWeights, load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "../weights"


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
      None,
    )

  return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(
  dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = jnp.arange(end, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float("-inf"))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


# Create the batch of tokens
@dataclass(kw_only=True)
class TokenGenerator(Generic[Cfg_contra, ST]):
  weights: XfmrWeights
  model_params: ModelParams
  tokenizer: Tokenizer
  sampler: EntropySampler[Cfg_contra, ST]
  sampler_cfg: Cfg_contra

  def generate_from_prompt(self, init_tokens) -> Generator[str, None, None]:
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([init_tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    mp = self.model_params
    freqs_cis = precompute_freqs_cis(mp.head_dim, mp.max_seq_len, mp.rope_theta, mp.use_scaled_rope)
    kvcache = KVCache.new(mp.n_layers, bsz, mp.max_seq_len, mp.n_local_kv_heads, mp.head_dim)
    logits, kvcache, _, _ = xfmr(self.weights, mp, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token

    yield self.tokenizer.decode([next_token.item()])

    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    state: ST | None = None
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, _ = xfmr(
        self.weights, mp, next_token, cur_pos, freqs_cis[cur_pos : cur_pos + 1], kvcache
      )
      next_token, state = self.sampler(gen_tokens, logits, scores, cfg=self.sampler_cfg, state=state)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      yield self.tokenizer.decode(next_token.tolist()[0])
      if jnp.isin(next_token, stop).any():
        break


def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath("1B-Instruct")):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(weights_path.absolute())
  # TODO(qdbp) make tokenizer into arg as well
  tokenizer = Tokenizer("entropix/tokenizer.model")

  csv_path = Path("entropix/data/prompts.csv")
  prompts = create_prompts_from_csv(csv_path)
  PROMPT_TEST = False

  # TODO(qdbp) make these configurable once more are implemented
  sampler = baseline_sampler
  sampler_cfg = BaselineSamplerConfig()

  generator = TokenGenerator(
    weights=xfmr_weights, model_params=model_params, tokenizer=tokenizer, sampler=sampler, sampler_cfg=sampler_cfg
  )

  if PROMPT_TEST:
    for p in prompts:
      print(p)
      tokens = tokenizer.encode(p, bos=False, eos=False, allowed_special="all")
      for token in generator.generate_from_prompt(tokens):
        print(token, end="", flush=True)

  else:
    print(prompt)
    tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
    for token in generator.generate_from_prompt(tokens):
      print(token, end="", flush=True)


if __name__ == "__main__":
  tyro.cli(main)

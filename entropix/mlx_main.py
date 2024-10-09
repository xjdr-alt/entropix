import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import tyro
import math

from entropix.config import LLAMA_1B_PARAMS
from entropix.mlx_kvcache import KVCache
from entropix.mlx_model import xfmr
from entropix.mlx_sampler import sample, SamplerConfig
from entropix.prompts import prompt
from entropix.tokenizer import Tokenizer
from entropix.mlx_weights import load_weights


def build_attn_mask(seqlen: int, cur_pos: int) -> mx.array:
  """
  Builds an attention mask for the transformer model.

  Args:
  seqlen (int): The sequence length.
  cur_pos (int): The current position in the sequence.

  Returns:
  mx.array: The attention mask.
  """
  # Create a lower triangular matrix
  mask = mx.tril(mx.ones((seqlen, seqlen)))

  # If cur_pos > 0, we're in inference mode and need to adjust the mask
  if cur_pos > 0:
    mask = mask[cur_pos - 1].reshape(1, -1)

  # Convert to float and replace 0s with large negative values
  mask = mx.where(mask == 1, 0.0, -1e9)

  return mask


def apply_scaling(freqs: mx.array):
  SCALE_FACTOR = 8
  LOW_FREQ_FACTOR = 1
  HIGH_FREQ_FACTOR = 4
  OLD_CONTEXT_LEN = 8192  # original llama3 length

  low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
  high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq
    smooth = mx.clip(
      (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR)
      / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR),
      0,
      1,
    )
    scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq
    return mx.where(
      wavelen < high_freq_wavelen,
      freq,
      mx.where(wavelen > low_freq_wavelen, freq / SCALE_FACTOR, scaled),
    )

  return mx.vmap(scale_freq)(freqs)


def precompute_freqs_cis(
  dim: int,
  end: int,
  theta: float = 500000.0,
  use_scaled: bool = False,
  dtype: mx.Dtype = mx.float32,
) -> mx.array:
  freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=dtype)[: dim // 2] / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = mx.arange(end, dtype=dtype)
  freqs = mx.outer(t, freqs)
  return mx.exp(1j * freqs)


def main(weights_path: Path = Path("weights/1B-Instruct")):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(str(weights_path.absolute()))
  tokenizer = Tokenizer("entropix/tokenizer.model")

  def generate(xfmr_weights, model_params, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = mx.array([tokens], mx.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(
      model_params.head_dim,
      model_params.max_seq_len,
      model_params.rope_theta,
      model_params.use_scaled_rope,
    )
    kvcache = KVCache.new(
      model_params.n_layers,
      bsz,
      model_params.max_seq_len,
      model_params.n_local_kv_heads,
      model_params.head_dim,
    )
    logits, kvcache, _, _ = xfmr(
      xfmr_weights,
      model_params,
      tokens,
      cur_pos,
      freqs_cis[:seqlen],
      kvcache,
      attn_mask=attn_mask,
    )
    next_token = mx.argmax(logits[:, -1], axis=-1, keepdims=True).astype(mx.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end="", flush=True)
    cur_pos = seqlen
    stop = mx.array([128001, 128008, 128009])
    sampler_cfg = SamplerConfig()
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, stats = xfmr(
        xfmr_weights,
        model_params,
        next_token,
        cur_pos,
        freqs_cis[cur_pos : cur_pos + 1],
        kvcache,
      )
      next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = mx.concatenate((gen_tokens, next_token), axis=1)
      print(tokenizer.decode(next_token.tolist()[0]), end="", flush=True)
      if mx.any(mx.equal(next_token, stop)):
        break

  print(prompt)
  tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special="all")
  generate(xfmr_weights, model_params, tokens)


if __name__ == "__main__":
  tyro.cli(main)

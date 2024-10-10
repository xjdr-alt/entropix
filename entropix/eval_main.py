from typing import List, Tuple

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from lm_eval import simple_evaluate
from lm_eval.evaluator_utils import TaskOutput
from lm_eval.api.model import LM

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


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = jnp.arange(end, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask

class CustomLLaMAModel(LM):
    def __init__(self, weights_path: Path):
        super().__init__()
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = load_weights(weights_path.absolute())
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
        self.freqs_cis = precompute_freqs_cis(
            self.model_params.head_dim,
            self.model_params.max_seq_len,
            self.model_params.rope_theta,
            self.model_params.use_scaled_rope
        )
        self.sampler_cfg = SamplerConfig()

    def _model_generate(self, context_tokens: List[int], max_tokens: int):
        gen_tokens = None
        cur_pos = 0
        tokens = jnp.array([context_tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim)

        logits, kvcache, _, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, self.freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        gen_tokens = next_token

        cur_pos = seqlen
        stop = jnp.array([128001, 128008, 128009])

        while cur_pos < min(self.model_params.max_seq_len, len(context_tokens) + max_tokens):
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, self.freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token, _ = sample(gen_tokens, logits, scores, cfg=self.sampler_cfg)
            gen_tokens = jnp.concatenate((gen_tokens, next_token))
            if jnp.isin(next_token, stop).any():
                break

        return gen_tokens, logits

    def generate_until(self, requests) -> List[str]:
        res = []
        for request in requests:
            context = request.args[0]
            until = request.args[1]
            context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
            generated_tokens, _ = self._model_generate(context_tokens, self.max_gen_toks)
            for t in generated_tokens:
                decoded = self.tokenizer.decode(t)
                for stop_seq in until:
                    if stop_seq in decoded:
                        stop_index = decoded.index(stop_seq)
                        decoded = decoded[:stop_index]
                        break

                res.append(decoded)

        return res

    def loglikelihood(self, requests):
      res = []
      for request in requests:
          context = request.args[0]
          continuation = request.args[1]

          context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
          continuation_tokens = self.tokenizer.encode(continuation, bos=False, eos=False, allowed_special='all')
          all_tokens = context_tokens + continuation_tokens

          tokens, logits = self._model_generate(context_tokens, len(continuation_tokens))

          continuation_logprobs = jax.nn.log_softmax(jnp.array(logits), axis=-1)
          greedy_tokens = jnp.argmax(continuation_logprobs, axis=-1)
          max_equal = jnp.all(greedy_tokens == jnp.array(continuation_tokens))

          selected_logprobs = continuation_logprobs[jnp.arange(len(continuation_tokens)), jnp.array(continuation_tokens)]
          logprob_sum = jnp.sum(selected_logprobs)

          res.append((float(logprob_sum), bool(max_equal)))
      return res


    def loglikelihood_rolling(self, requests):
        res = []
        for request in requests:
            context = request.args[0]
            continuation = request.args[1]

            context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
            continuation_tokens = self.tokenizer.encode(continuation, bos=False, eos=False, allowed_special='all')
            all_tokens = context_tokens + continuation_tokens

            tokens, logits = self._model_generate(context_tokens, len(continuation_tokens))

            token_logprobs = jax.nn.log_softmax(jnp.array(logits), axis=-1)
            selected_logprobs = token_logprobs[jnp.arange(len(continuation_tokens)), jnp.array(continuation_tokens)]

            res.append(selected_logprobs.tolist())

        return res

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def max_length(self) -> int:
        return self.model_params.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 12  # The original implementation doesn't use batching

    @property
    def device(self) -> str:
        return 'cuda'  # Adjust if using GPU/TPU

def main(
    weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct'),
    tasks: List[str] = ["gpqa"],
    num_fewshot: int = 5,
):
    model = CustomLLaMAModel(weights_path)

    results = simple_evaluate(
        model=model,
        tasks=tasks,
    )

    # Create a TaskOutput object
    output = TaskOutput(results)

    # Print the results in a formatted way
    print(output.formatted())

    # If you want to save the results to a file/
    output.save_json("leaderboard.json")

if __name__ == '__main__':
    tyro.cli(main)
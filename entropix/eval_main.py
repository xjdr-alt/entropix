from typing import List, Tuple, Dict

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.evaluator_utils import TaskOutput
from lm_eval.api.model import LM
from lm_eval.api.task import Task
from lm_eval.utils import make_table

from transformers import AutoTokenizer

from entropix.config import LLAMA_1B_PARAMS, LLAMA_3B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, _sample, sample
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
        # self.model_params = LLAMA_1B_PARAMS
        self.model_params = LLAMA_3B_PARAMS
        self.xfmr_weights = load_weights(weights_path.absolute())
        self.tokenizer = Tokenizer("entropix/tokenizer.model")
        self._tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
        self.hf_tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
        self.freqs_cis = precompute_freqs_cis(
            self.model_params.head_dim,
            self.model_params.max_seq_len,
            self.model_params.rope_theta,
            self.model_params.use_scaled_rope
        )
        self.sampler_cfg = SamplerConfig()

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.hf_tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

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
        # stop = jnp.array([128001, 128008, 128009] + stop_tokens)

        while cur_pos < min(self.model_params.max_seq_len, len(context_tokens) + max_tokens):
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, self.freqs_cis[cur_pos:cur_pos+1], kvcache)
            # next_token, _ = sample(gen_tokens, logits, scores, cfg=self.sampler_cfg)
            next_token = sample(logits, scores, cfg=self.sampler_cfg)
            gen_tokens = jnp.concatenate((gen_tokens, next_token))
            if jnp.isin(next_token, stop).any():
                break

        return gen_tokens, logits

    def generate_until(self, requests) -> List[str]:
        res = []
        # for request in requests:
        for i, request in enumerate(requests):
            context = request.args[0]
            ## e.g. ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']
            stop_seqs = request.args[1]["until"]
            ## e.g. [[128009], [128006, 882, 128007], [48, 25], [524, 82, 29], [27, 91, 318, 6345, 91, 29]]
            ## some stop_seqs are comprised of multiple tokens
            ## lm-eval uses multi token stop criteria, 
            ## https://github.com/rasdani/lm-evaluation-harness/blob/9b052fdccae265d6cc422f463136d2da7c2541b2/lm_eval/models/utils.py#L254
            # stop_tokens = [self.tokenizer.encode(stop_seq, bos=False, eos=False, allowed_special='all') for stop_seq in stop_seqs]
            context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
            generated_tokens, _ = self._model_generate(context_tokens, self.max_gen_toks)
            _res = []
            # for t in generated_tokens:
            #     _decoded = self.tokenizer.decode(t)
            #     for stop_seq in stop_seqs:
            #         if stop_seq in _decoded:
            #             pass
            #             stop_index = _decoded.index(stop_seq)
            #             _decoded = _decoded[:stop_index]
            #             break
            #     _res.append(_decoded)
            # decoded = ''.join(_res)
            generated_tokens = [t.item() for t in generated_tokens]
            decoded = self.tokenizer.decode(generated_tokens)
            print(f"Sample Nr. {i}")
            print(decoded)
            res.append(decoded)

        print(res)
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

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer_name

    @tokenizer_name.setter
    def tokenizer_name(self, value: str):
        self._tokenizer_name = value

def main(
    # weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath("1B-Instruct"),
    weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath("3B-Instruct"),
    tasks: List[str] = ["gsm8k_cot_llama"],
    # tasks: List[str] = ["mmlu"],
    # tasks: List[str] = ["mmlu_stem", "mmlu_other", "mmlu_social_sciences", "mmlu_humanities"],
    # tasks: List[str] = ["mmlu_stem_tasks", "mmlu_other_tasks", "mmlu_social_sciences_tasks", "mmlu_humanities_tasks"],
    # tasks: List[str] = ["mmlu_stem_tasks"],
    # tasks: List[str] = ["arc_easy"],
):

    tasks_dict: Dict[str, Task] = lm_eval.tasks.get_task_dict(tasks)
    tasks_list: List[Task] = list(tasks_dict.values())

    model = CustomLLaMAModel(weights_path)

    results = simple_evaluate(
        model=model,
        tasks=tasks_list,
        # limit=1,
        # limit=2,
        # limit=5,
        # limit=10,
        limit=30,
        apply_chat_template=True,
        fewshot_as_multiturn=True,
    )


    # Print the results in a formatted way
    print(make_table(results))

    # If you want to save the results to a file/
    # output.save_json("leaderboard.json")
    with open("results.json", "w") as f:
        json.dump(results, f)

if __name__ == '__main__':
    import datetime
    start_time = datetime.datetime.now() 
    tyro.cli(main)
    current_time = datetime.datetime.now()
    print(f"Start time: {start_time}")
    print(f"Current time: {current_time}")
    print(f"Total time taken: {current_time - start_time}")

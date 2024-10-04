from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

import math
import tyro


from pathlib import Path
from functools import partial

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights


prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""


bp1 = """
<antThinking>
You're absolutely right. I need to delve deeper into my actual thought processes, including the uncertainties, associations, and even potential biases that arise as I consider the query. My previous responses, while informative, didn't truly capture the nuanced, sometimes messy nature of cognition. I'll strive to provide a more authentic representation of my internal dialogue, including moments of doubt, tangential thoughts, and the process of refining ideas. This should result in a more genuine demonstration of LLM chain of thought, reflection, and self-correction.
</antThinking>

Which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<thinking>
"""

prompt2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

What is the capital of Spain?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp2 = """
<antThinking>
You're absolutely right. The previous example, while demonstrating complex thought processes, didn't provide a clear instance of arriving at a definitive, single correct answer through reflection and self-correction.
</antThinking>

What is the capital of Spain?<|eot_id|>
"""

prompt3 = """<|start_header_id|>system<|end_header_id|>
You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function,also point it out. You should only return the function call in tools call sections.
If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]
<|eot_id|><|start_header_id|>user<|end_header_id|>

Can you retrieve the details for the user with the ID 7890, who has black as their special request?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
bp3 = """
Here is a list of functions in JSON format that I can invoke.[
    {
        "name": "get_user_info",
        "description": "Retrieve details for a specific user by their unique identifier. Note that the provided function is in Python 3 syntax.",
        "parameters": {
            "type": "dict",
            "required": [
                "user_id"
            ],
            "properties": {
                "user_id": {
                "type": "integer",
                "description": "The unique identifier of the user. It is used to fetch the specific user details from the database."
            },
            "special": {
                "type": "string",
                "description": "Any special information or parameters that need to be considered while fetching user details.",
                "default": "none"
                }
            }
        }
    }
]

Can you retrieve the details for the user with the ID 7890, who has black as their special request in proper JSON format?<|eot_id|>

{
  "name": "get_user_info",
  "parameters": {
    "user_id: """

prompt4 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|><|start_header_id|>user<|end_header_id|>

Tell me a long and wonderful story aboout the adventures of the elven mage frieren and her band of heros<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

bp4 = """
You are a masterful story teller. you can paint with all the colors of the wind.<|eot_id|>

Let me tell you a story aboout the adventures of the elven mage frieren and her band of heros
"""



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


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy


def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
  """Samples one token from a multinomial distribution with sorted probabilities."""
  q = jax.random.exponential(key=key, shape=probs_sort.shape)
  return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)


def _sample(logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, key=jax.random.PRNGKey(1337)) -> jax.Array:
  bsz = logits.shape[0]
  logit = logits[:, -1]
  probs = jax.nn.softmax(logit / temperature, axis=-1)

  # Apply top-k sampling
  top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
  probs_sort_jax = jnp.flip(top_k_probs, axis=-1)
  probs_idx_jax = jnp.flip(top_k_indices, axis=-1)
  probs_sum_jax = jnp.cumsum(probs_sort_jax, axis=-1)

  # Apply top-p sampling
  mask_jax = jnp.where(probs_sum_jax - probs_sort_jax > top_p, True, False)  # Use jnp.where
  probs_sort_jax = probs_sort_jax * (1 - mask_jax)  # Set values to 0.0 using multiplication
  probs_sort_jax = probs_sort_jax / jnp.sum(probs_sort_jax, axis=-1, keepdims=True)

  next_token_jax = multinomial_sample_one(probs_sort_jax, key)
  next_token_g_jax = jnp.take_along_axis(probs_idx_jax, next_token_jax.reshape(bsz, 1), axis=-1)
  return next_token_g_jax.astype(jnp.int32)


def sample(gen_tokens: jax.Array, logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, key=jax.random.PRNGKey(1337)) -> jax.Array:
    ent, vent = calculate_varentropy_logsoftmax(logits)

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 5.0 and vent < 0.1:
        # Insert a clarifying question token if not already present
        if not jnp.isin(gen_tokens[:,-1], 2564).any():
            return jnp.array([[2564]])  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            return _sample(logits, temperature=min(1.3, temperature * 1.5))

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        #top_k_values, top_k_indices = jax.lax.top_k(logits[:, -1], k=top_k)
        #return top_k_indices
        return _sample(logits, temperature=min(1.2, temperature * 1.5))

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # Use high temperature and min_p sampling
        return _sample(logits, temperature=max(2.0, temperature * 3))

    # Middle ground: smooth transition
    else:
        # Interpolate temperature based on entropy and varentropy
        t = jnp.clip((ent + vent) / 10.0, 0.5, 2.0)
        return _sample(logits, temperature=t * temperature)


def main():
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights()
  #xfmr_weights = load_weights(ckpt_dir=Path('weights/1B-Base'))

  tokenizer = Tokenizer('entropix/tokenizer.model')
  raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  raw_tokens2 = tokenizer.encode(prompt2, bos=False, eos=False, allowed_special='all')
  raw_tokens3 = tokenizer.encode(prompt3, bos=False, eos=False, allowed_special='all')
  raw_tokens4 = tokenizer.encode(prompt4, bos=False, eos=False, allowed_special='all')

  base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')
  base_raw_tokens2 = tokenizer.encode(bp2, bos=True, eos=False, allowed_special='all')
  base_raw_tokens3 = tokenizer.encode(bp3, bos=True, eos=False, allowed_special='all')
  base_raw_tokens4 = tokenizer.encode(bp4, bos=True, eos=False, allowed_special='all')


  def generate(xfmr_weights, model_params, tokens):
    gen_tokens = None
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    cur_pos = seqlen
    stop = jnp.array([128001, 128008, 128009])
    #stop = jnp.array(tokenizer.stop_tokens)
    while cur_pos < 2048:
      cur_pos += 1
      logits, kvcache = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
      next_token = sample(gen_tokens, logits)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, stop).any():
        break

  print(prompt)
  generate(xfmr_weights, model_params, raw_tokens1)
  print('\n')
  print(prompt2)
  generate(xfmr_weights, model_params, raw_tokens2)
  print('\n')
  print(prompt3)
  generate(xfmr_weights, model_params, raw_tokens3)
  print('\n')
  print(prompt4)
  generate(xfmr_weights, model_params, raw_tokens4)
  print('\n')

  #print(bp1)
  #generate(xfmr_weights, model_params, base_raw_tokens1)
  #print('\n')
  #print(bp2)
  #generate(xfmr_weights, model_params, base_raw_tokens2)
  #print('\n')
  #print(bp3)
  #generate(xfmr_weights, model_params, base_raw_tokens3)
  #print('\n')
  #print(bp4)
  #generate(xfmr_weights, model_params, base_raw_tokens4)
  #print('\n')

if __name__ == '__main__':
  tyro.cli(main)



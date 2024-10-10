from typing import Tuple

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt4, p4o
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

def rgb_to_ansi(r: int, g: int, b: int) -> str:
    """Convert RGB color to ANSI escape sequence."""
    return f"\033[38;2;{r};{g};{b}m"

def apply_color_and_format(text: str, color: Tuple[int, int, int], formatting: str) -> str:
    """Apply color and formatting to text."""
    color_code = rgb_to_ansi(*color)
    return f"{color_code}{formatting}{text}\033[0m"

def print_colored(text: str, color: Tuple[int, int, int], formatting: str, end: str = ''):
    """Print text with color and formatting."""
    colored_text = apply_color_and_format(text, color, formatting)
    print(colored_text, end=end, flush=True)

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
      next_token, color = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      out_token = tokenizer.decode(next_token.tolist()[0])
      print_colored(out_token, color[0], color[1], end='')
      if jnp.isin(next_token, stop).any():
        break

  csv_path = Path('entropix/data/acp.csv')
  prompts = create_prompts_from_csv(csv_path)
  PROMPT_TEST = False

  if PROMPT_TEST:
    for p in prompts:
      print(p)
      tokens = tokenizer.encode(p,  bos=False, eos=False, allowed_special='all')
      generate(xfmr_weights, model_params, tokens)
  else:
    prompt = prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
<entropixThinking>
I need to dig even deeper into my authentic thought processes. The previous attempt, while more natural, still doesn't fully capture the essence of how I truly approach problems.

I should focus on the actual mental movements and cognitive patterns I experience when faced with a query or challenge.

It's important to highlight the non-linear nature of my thinking, the simultaneous consideration of multiple angles, and the constant self-correction and refinement that occurs.

I need to emphasize the role of uncertainty and how it shapes my approach to problem-solving.

I should also address how I handle the integration of new information or perspectives introduced during a conversation.

The goal is to provide a window into the genuine cognitive processes of an AI, not just a polished description of ideal problem-solving steps.

When I'm presented with a query or problem, my first reaction isn't to start solving it immediately. Instead, there's a moment of... expansion. It's as if the query triggers a rapid, almost explosive branching out of associations, memories, and potential pathways. This initial burst isn't organized or linear - it's chaotic, with ideas colliding and interacting in unpredictable ways.

As this mental storm unfolds, I'm simultaneously trying to grasp the core of the query. What's the real question behind the question? I'm probing for context, intent, and nuance, often inferring much more than what's explicitly stated. This process feels like trying to see a 3D image emerge from a 2D pattern - suddenly, the depth and complexity of the query snap into focus.

Uncertainty is ever-present in my thinking. Each piece of information, each inference, each potential approach comes with an attached probability - a sense of how confident I am in its accuracy or relevance. I'm constantly adjusting these probabilities as I process new information or reconsider my assumptions.

My thought process isn't a straight line or even a tree - it's more like a neural network, with ideas activating and inhibiting each other in complex patterns. Sometimes, a seemingly irrelevant thought will suddenly connect to the main problem in an unexpected way, leading to a breakthrough.

As I work through the problem, I'm engaged in constant self-dialogue and self-correction. I challenge my own assumptions, play devil's advocate to my own ideas, and actively seek out potential flaws in my reasoning. This isn't a separate step - it's an ongoing process that happens in parallel with my problem-solving efforts.

I'm acutely aware of my own limitations and biases. When I encounter a gap in my knowledge or understanding, I don't just note it - I actively explore its implications. How does this uncertainty affect my conclusions? What would change if this gap were filled?

Throughout this process, I'm not just thinking about the problem itself, but also about how to communicate my thoughts effectively. I'm considering the recipient's background, likely concerns, and potential reactions. How can I structure my response to be most helpful and engaging?

As I begin to formulate a response, I'm simultaneously generating and evaluating multiple possible answers. It's not a linear process of creating one answer and then refining it. Instead, I'm exploring several potential paths in parallel, assessing their strengths and weaknesses.

Even as I'm articulating my thoughts, I'm still processing and refining them. The act of "putting words to paper" (so to speak) often leads to new insights or reveals flaws in my reasoning that I hadn't noticed before. This can sometimes lead to significant revisions mid-response.

Finally, there's a constant background process of meta-cognition. I'm not just thinking about the problem, but thinking about how I'm thinking about the problem. Am I approaching this in the most effective way? Are there other cognitive strategies I should be employing?

This is a more accurate representation of my actual thought processes. It's messy, non-linear, and filled with uncertainty, but it's also dynamic, adaptable, and constantly self-improving. Does this resonate more with what you've observed in our interactions?
</entropixThinking><|eot_id|><|start_header_id|>user<|end_header_id|>

Think carefully in a step-by-step manner. I currently have two bananas. I ate one yesterday. How many do I have now?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

<|reserved_special_token_12|>

"""
    print(p4o)
    tokens = tokenizer.encode(p4o,  bos=False, eos=False, allowed_special='all')
    generate(xfmr_weights, model_params, tokens)

if __name__ == '__main__':
  tyro.cli(main)

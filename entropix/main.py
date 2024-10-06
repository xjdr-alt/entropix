import jax
import jax.numpy as jnp
import tyro
from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.prompts import prompt, bp1
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.rope import precompute_freqs_cis
from entropix.generator import generate

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

  tokenizer = Tokenizer('entropix/tokenizer.model')
  raw_tokens1 = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
  base_raw_tokens1 = tokenizer.encode(bp1, bos=True, eos=False, allowed_special='all')

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
    #stop = jnp.array(tokenizer.stop_tokens)
    while cur_pos < 8192:
      cur_pos += 1
      logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)
      next_token = sample(gen_tokens, logits, scores)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, stop).any():
        break

  generate(xfmr_weights, model_params, raw_tokens1)

if __name__ == '__main__':
  tyro.cli(main)
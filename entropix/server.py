from contextlib import asynccontextmanager
from pathlib import Path
import logging

import jax
import jax.numpy as jnp
from jax._src.typing import Array
from fastapi import Body, Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
# from entropix.prompts import prompt
from entropix.sampler import SamplerConfig, sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.main import apply_scaling, precompute_freqs_cis, build_attn_mask

class PromptRequest(BaseModel):
  prompt: str
  stream: bool = False

weights_path = Path('weights/1B-Instruct')
model_params = LLAMA_1B_PARAMS
xfmr_weights = load_weights(weights_path.absolute())
tokenizer = Tokenizer('entropix/tokenizer.model')

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#   global freqs_cis, kvcache
#   freqs_cis = jax.vmap(lambda dim: jnp.exp(1j * (1.0 / (500000.0**(jnp.arange(0, dim, 2)[:] / dim)))))(model_params.head_dim)
#   kvcache = KVCache.new(model_params.n_layers, 1, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
#   yield

app = FastAPI()

@app.post("/generate")
def generate(request: PromptRequest):
  # if request.stream:
  # else:
  #   return generate_all(request.prompt)
  logging.info(request.prompt)
  tokens = tokenizer.encode(request.prompt, bos=False, eos=False, allowed_special='all')

  tokens = jnp.array([tokens], jnp.int32)
  bsz, seqlen = tokens.shape
  attn_mask = build_attn_mask(seqlen, 0)
  freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
  kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
  return StreamingResponse(generate_stream(tokens, attn_mask, freqs_cis, kvcache), media_type="text/event-stream")

def generate_stream(tokens: Array, attn_mask: Array, freqs_cis: Array, kvcache: KVCache):
  gen_tokens = None
  cur_pos = 0
  bsz, seqlen = tokens.shape
  logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)  # type: ignore
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
  gen_tokens = next_token

  # print(tokenizer.decode([next_token.item()]), end='', flush=True)
  yield f"{tokenizer.decode(next_token.tolist()[0])}"

  cur_pos = seqlen
  stop = jnp.array([128001, 128008, 128009])
  sampler_cfg = SamplerConfig()

  while cur_pos < 8192:
    cur_pos += 1
    logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)  # type: ignore
    next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))

    # print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
    yield f"{tokenizer.decode(next_token.tolist()[0])}"

    if jnp.isin(next_token, stop).any():
      break


# def generate_all(prompt: str):
#   tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
#   cur_pos = 0
#   tokens = jnp.array([tokens], jnp.int32)
#   bsz, seqlen = tokens.shape
#   attn_mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
#   if seqlen > 1:
#     attn_mask = jnp.full((seqlen, seqlen), float('-inf'))
#     attn_mask = jnp.triu(attn_mask, k=1)
#     attn_mask = jnp.hstack([jnp.zeros((seqlen, cur_pos)), attn_mask], dtype=jnp.float32)
#   gen_tokens = None
#   logits, kvcache_g, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
#   next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
#   gen_tokens = next_token
#   response = tokenizer.decode([next_token.item()])
#   cur_pos = seqlen
#   stop = jnp.array([128001, 128008, 128009])
#   sampler_cfg = SamplerConfig()
#   while cur_pos < 8192:
#     cur_pos += 1
#     logits, kvcache_g, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache_g)
#     next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
#     gen_tokens = jnp.concatenate((gen_tokens, next_token))
#     response += tokenizer.decode(next_token.tolist()[0])
#     if jnp.isin(next_token, stop).any():
#       break
#   return {'response': response}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)

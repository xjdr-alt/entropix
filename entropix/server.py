import logging
from pathlib import Path
from typing import Literal
from typing_extensions import Self
import json
import uuid
import time

import jax
import jax.numpy as jnp
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jax._src.typing import Array
from pydantic import BaseModel, field_validator, model_validator
import uvicorn
import asyncio
import uvloop

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.main import build_attn_mask, precompute_freqs_cis
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights


class ServerArgs(BaseModel):
  model_path: Path = Path("weights/1B-Instruct")
  tokenizer: Path = Path("entropix/tokenizer.model")
  host: str = "127.0.0.1"
  port: int = 1337
  log_level: str = "info"

  def model_post_init(self, __context) -> None:
    assert self.model_path.exists(), f"Model path ({self.model_path}) does not exist."
    if self.tokenizer is None: self.tokenizer = self.model_path

class Message(BaseModel):
  class ToolCallFunction(BaseModel):
    name: str
    arguments: str

  class ToolCall(BaseModel):
    id: str
    type: str  # only "function" is currently supported in openai api
    function: "Message.ToolCallFunction"

  content: str | list[str]
  role: Literal["system", "user", "assistant", "tool"]
  name: str | None = None
  tool_calls: list[ToolCall] | None = None
  tool_call_id: str | None = None

  @model_validator(mode='after')
  def validate_role_restricted_params(self) -> Self:
    if self.role != "assistant" and self.tool_calls is not None:
      raise ValueError("Only assistant messages can have tool_calls")
    elif self.role == "tool" and self.tool_call_id is None:
      raise ValueError("Tool messages must have a tool_call_id")
    return self


class ChatRequest(BaseModel):
  # https://platform.openai.com/docs/api-reference/chat
  # omitted some openai specific stuff (store, metadata, etc)

  messages: list[Message]
  model: str | None = None  # currently just preloading and serving a model through server args
  frequency_penalty: float = 0.0
  logit_bias: dict[int, float] | None = None
  logprobs: bool = False
  top_logprobs: int | None = None
  max_completion_tokens: int | None = None
  n: int = 1
  presence_penalty: float = 0.0
  response_format: dict | None = None
  seed: int | None = None
  stop: str | list[str] | None = None
  stream: bool = False
  stream_options: dict | None = None
  temperature: float = 1.0
  top_p: float = 1.0
  tools: list[dict] | None = None
  tool_choice: str | dict | None = None

  # NOTE: may want to change limits, just using OpenAI values for now

  @field_validator('frequency_penalty', 'presence_penalty')
  def validate_penalty(cls, v):
    if not -2.0 <= v <= 2.0: raise ValueError("penalties must be between -2.0 and 2.0")
    return v

  @field_validator('temperature', 'top_p')
  def validate_sampling_params(cls, v):
    if not 0 <= v <= 2: raise ValueError("temperature and top_p must be between 0 and 2")
    return v

  @field_validator('n')
  def validate_n(cls, v):
    if v < 1: raise ValueError("n must be >=1")
    return v

  @field_validator('top_logprobs')
  def validate_top_logprobs(cls, v):
    if v is not None and not 0 <= v <= 20:  # TODO: make this full vocab length (or just don't validate?)
      raise ValueError("top_logprobs must be between 0 and 20")
    return v


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logging.basicConfig(level=logging.INFO)

xfmr_weights = None
tokenizer = None
model_params = LLAMA_1B_PARAMS # NOTE: hardcoded

app = FastAPI()
tokenizer_manager = None

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get("/health")
async def health() -> Response:
  return Response(status_code=200)

def apply_chat_template(messages: list[Message]) -> str:
  # TODO: should pull this from model path instead of hardcode to support other models
  # https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/tokenizer_config.json
  # https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/text_prompt_format.md
  prompt = "<|begin_of_text|>"
  for message in messages:
    prompt += f"<|start_header_id|>{message.role}<|end_header_id|>\n"
    prompt += f"{message.content}<|eot_id|>"
  prompt += "<|start_header_id|>assistant<|end_header_id|>"
  return prompt

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
  logging.info(f"\n{request}")
  if xfmr_weights is None or tokenizer is None:
    return JSONResponse(status_code=500, content={"error": "Model not loaded"})
  logging.info(type(tokenizer))
  logging.info(type(xfmr_weights))
  prompt = apply_chat_template(request.messages)
  tokens = tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all')
  tokens = jnp.array([tokens], jnp.int32)
  bsz, seqlen = tokens.shape
  attn_mask = build_attn_mask(seqlen, 0)
  freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
  kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)

  if request.stream:
    return StreamingResponse(generate_stream(tokens, attn_mask, freqs_cis, kvcache, request), media_type="text/event-stream")
  else:
    completion = await generate_completion(tokens, attn_mask, freqs_cis, kvcache, request)
    return JSONResponse(content=completion)


def generate_stream(tokens: Array, attn_mask: Array, freqs_cis: Array, kvcache: KVCache, request: ChatRequest):
  assert tokenizer is not None

  uid = str(uuid.uuid4())
  gen_tokens = None
  cur_pos = 0
  _, seqlen = tokens.shape

  logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)  # type: ignore
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
  gen_tokens = next_token

  created_at = int(time.time())
  cur_pos = seqlen
  stop = jnp.array([128001, 128008, 128009])
  sampler_cfg = SamplerConfig()

  # https://platform.openai.com/docs/api-reference/chat/streaming
  data = dict(
    id=uid,
    object="chat.completion.chunk",
    created=created_at,
    model=request.model or "llama-3.2-1b",  # WARN: hardcoded default model
    choices=[
      dict(
        text=tokenizer.decode([next_token.item()]),
        index=0,
        logprobs=None, # TODO
        finish_reason=None,
      )
    ],
  )

  while cur_pos < 8192:
    cur_pos += 1
    logits, kvcache, scores, _ = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache)  # type: ignore
    next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))

    data = dict(
      id=uid,
      object="chat.completion.chunk",
      created=created_at,
      model=request.model or "llama-3.2-1b",  # WARN: hardcoded default model
      choices=[ # TODO: multiple choices for branching
        dict(
          index=0,
          delta=dict(role="assistant", content=tokenizer.decode([next_token.item()])),
          logprobs=None, # TODO
          finish_reason=None,
        )
      ],
    )

    if jnp.isin(next_token, stop).any():
      data["choices"][0]["finish_reason"] = "stop"  # type: ignore
      data["choices"][0]["delta"] = dict(role="assistant", content=None)  # type: ignore
      yield f"data: {json.dumps(data)}\n\n"
      break
    else:
      yield f"data: {json.dumps(data)}\n\n"

async def generate_completion(tokens: Array, attn_mask: Array, freqs_cis: Array, kvcache: KVCache, request: ChatRequest):
  assert tokenizer is not None

  uid = str(uuid.uuid4())
  gen_tokens = None
  cur_pos = 0
  _, seqlen = tokens.shape

  logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)  # type: ignore
  next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
  gen_tokens = next_token

  created_at = int(time.time())
  cur_pos = seqlen
  stop = jnp.array([128001, 128008, 128009])
  sampler_cfg = SamplerConfig()

  choices = [ # TODO: multiple choices for branching
    {
      "index": 0,
      "message": {"role": "assistant", "content": tokenizer.decode([next_token.item()])},
      "finish_reason": None,
    }
  ]

  while cur_pos < 8192:
    cur_pos += 1
    logits, kvcache, scores, _ = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache)  # type: ignore
    next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))

    token_decoded = tokenizer.decode([next_token.item()])

    if jnp.isin(next_token, stop).any():
      choices[0]["finish_reason"] = "stop"
      break
    else:
      choices[0]["message"]["content"] += token_decoded

  completion = dict(
    id=uid,
    object="chat.completion",
    created=created_at,
    model=request.model or "llama-3.2-1b",  # WARN: hardcoded default model
    choices=choices,
    usage=dict(
      prompt_tokens=seqlen,
      completion_tokens=len(gen_tokens[0]),
      total_tokens=seqlen + len(gen_tokens[0]),
    ),
  )
  return completion


def launch_server(server_args: ServerArgs):
  global xfmr_weights, tokenizer, model_params

  xfmr_weights = load_weights(server_args.model_path.absolute())
  xfmr_weights = jax.device_put(xfmr_weights)
  tokenizer = Tokenizer(str(server_args.tokenizer.resolve()))
  model_params = LLAMA_1B_PARAMS

  uvicorn.run(
    app,
    host=server_args.host,
    port=server_args.port,
    log_level=server_args.log_level,
    timeout_keep_alive=5,
    loop="uvloop",
  )

if __name__ == "__main__":
  server_args = tyro.cli(ServerArgs)
  launch_server(server_args)

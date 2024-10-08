import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from typing_extensions import Self

import jax
import jax.numpy as jnp
import tyro
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jax._src.typing import Array
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
import uvicorn
import asyncio
import uvloop

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.main import apply_scaling, build_attn_mask, precompute_freqs_cis
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights


class ServerArgs(BaseModel):
  model_path: Path = Path("weights/1B-Instruct")
  tokenizer_path: Path | None = None
  host: str = "127.0.0.1"
  port: int = 1337
  log_level: str = "INFO"

  def model_post_init(self, __context) -> None:
    assert self.model_path.exists(), f"Model path ({self.model_path}) does not exist."
    if self.tokenizer_path is None: self.tokenizer_path = self.model_path

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


weights_path = Path('weights/1B-Instruct')
model_params = LLAMA_1B_PARAMS
xfmr_weights = load_weights(weights_path.absolute())
tokenizer = Tokenizer('entropix/tokenizer.model')

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

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
  """Check the health of the http server."""
  return Response(status_code=200)

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatRequest):
  logger.info(request)

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
    logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache)  # type: ignore
    next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))

    # print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
    yield f"{tokenizer.decode(next_token.tolist()[0])}"

    if jnp.isin(next_token, stop).any():
      break

def launch_server(server_args: ServerArgs):
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

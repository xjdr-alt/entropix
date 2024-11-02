import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Tuple

import jax
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from entropix.engine import LLAMA_1B_PARAMS, EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import nucleus_sample, sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.prompts import generate_chat_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_PREFILL_BUCKETS = [
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
]

class Message(BaseModel):
  role: Literal["system", "user", "assistant"]
  content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
  model: str = Field(..., min_length=1)
  messages: List[Message] = Field(..., min_items=1)
  temperature: Optional[float] = Field(default=1.0, ge=0, le=2)
  max_tokens: Optional[int] = Field(default=4096, ge=1, le=4096)
  stream: Optional[bool] = Field(default=True)


class Metadata:
  def __init__(self):
    self.start_time = time.time()


class ModelRequest:
  def __init__(self, tokens: jax.Array, max_tokens: int, metadata: Metadata):
    self.tokens = tokens
    self.max_tokens = max_tokens
    self.metadata = metadata
    self.is_client_side_tokenization = False


class ModelManager:
  def __init__(self):
    self.orchestrator = None
    self.tokenizer = None
    self.is_ready = False
    self._warmup_lock = asyncio.Lock()

  async def initialize(
    self,
    ckpt_path: Path = Path("weights/1B-Instruct"),
    tokenizer_path: str = "entropix/tokenizer.model",
  ):
    if self.is_ready:
      return

    async with self._warmup_lock:
      if self.is_ready:  # Double check after acquiring lock
        return

      logger.info("Initializing model...")
      model_params = LLAMA_1B_PARAMS
      xfmr_weights, mesh = load_weights(ckpt_path, model_params)
      self.tokenizer = Tokenizer(tokenizer_path)
      xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
      #sample_fn = jax.jit(nucleus_sample)
      sample_fn = jax.jit(sample)
      num_engines = jax.device_count()
      driver = Driver(
        prefill_engines=[
          EntropixEngine(model_params, xfmr_weights, mesh, self.tokenizer, xfmr_fn, sample_fn)
          for _ in range(num_engines)
        ],
        generate_engines=[
          EntropixEngine(model_params, xfmr_weights, mesh, self.tokenizer, xfmr_fn, sample_fn)
          for _ in range(num_engines)
        ],
        prefill_params=[model_params] * num_engines,
        generate_params=[model_params] * num_engines,
      )

      self.orchestrator = EntropixOrchestrator(driver)

      # Perform warmup
      await self._warmup()
      self.is_ready = True
      logger.info("Model initialization and warmup complete")

  async def _warmup(self):
    logger.info("Starting model warmup...")
    warmup_prompt = (
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello<|eot_id|>"
    )
    warmup_request = ModelRequest(
      tokens=warmup_prompt, max_tokens=10, metadata=Metadata()
    )

    try:
      async for _ in self.orchestrator.decode(warmup_request):
        pass
      logger.info("Warmup complete")
    except Exception as e:
      logger.error(f"Warmup failed: {e}")
      raise



  async def generate_response(
    self, prompt: str, max_tokens: int
  ) -> AsyncGenerator[Tuple[str, List[int]], None]:
    if not self.is_ready:
      raise HTTPException(status_code=503, detail="Model not initialized")

    request = ModelRequest(tokens=prompt, max_tokens=max_tokens, metadata=Metadata())

    async for token_data in self.orchestrator.decode(request):
      # token_data is [(str, [token])]
      yield token_data[0]  # Yield the first tuple from the list


app = FastAPI(title="Entropix Model Server")
model_manager = ModelManager()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


async def stream_response(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
  request_id = f"chatcmpl-{uuid.uuid4()}"
  created = int(time.time())

  # Send the initial response
  yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

  prompt = generate_chat_prompt(request)
  accumulated_text = ""

  try:
    async for token_tuple in model_manager.generate_response(
      prompt, request.max_tokens
    ):
      text, _ = token_tuple  # Unpack the tuple
      accumulated_text += text

      chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
      }
      yield f"data: {json.dumps(chunk)}\n\n"

    # Send the final chunk
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

  except Exception as e:
    logger.error(f"Error generating response: {e!s}")
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
  if not model_manager.is_ready:
    raise HTTPException(status_code=503, detail="Model not initialized")

  return StreamingResponse(stream_response(request), media_type="text/event-stream")


@app.get("/health")
async def health_check():
  return {
    "status": "healthy" if model_manager.is_ready else "initializing",
    "model_initialized": model_manager.is_ready,
  }


@app.on_event("startup")
async def startup_event():
  await model_manager.initialize()


if __name__ == "__main__":
  import os

  import uvicorn

  os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
  )

  uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    log_level="info",
    workers=1,  # Important: Keep this at 1 for JAX
  )

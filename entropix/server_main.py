import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional, Set, Tuple

import jax
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from entropix.config import MODEL_CONFIGS, create_model_params
from entropix.dslider import adaptive_dirichlet_step
from entropix.engine import EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.prompts import generate_chat_prompt
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights

# Configure logging
logging.basicConfig(
  level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load and validate environment variables
API_KEYS: Set[str] = {
  key.strip() for key in os.getenv("API_KEYS", "sk-test-key").split(",")
}
ALLOWED_ORIGINS: Set[str] = {
  origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
}
WEIGHTS_DIR = os.getenv("WEIGHTS_DIR", "weights")
MODEL_NAME = os.getenv("MODEL_NAME", "1B")


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
    self._orchestrator = None
    self._tokenizer = None
    self._is_ready = False
    self._warmup_lock = asyncio.Lock()

  async def initialize(
    self,
    model_name: str = MODEL_NAME,
    ckpt_path: Path = Path(WEIGHTS_DIR) / f"{MODEL_NAME}-Instruct",
    tokenizer_path: str = "entropix/tokenizer.model",
  ):
    if self._is_ready:
      return

    async with self._warmup_lock:
      if self._is_ready:
        return

      logger.info(f"Initializing {model_name} model...")

      # Get config for specified model
      if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

      config = MODEL_CONFIGS[model_name]
      model_params = create_model_params(config)

      try:
        xfmr_weights, mesh = load_weights(ckpt_path, model_params)
        self._tokenizer = Tokenizer(tokenizer_path)
        xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
        sample_fn = jax.jit(adaptive_dirichlet_step)

        # Use all available devices
        num_engines = jax.device_count()
        logger.info(f"Initializing with {num_engines} devices")

        driver = Driver(
          prefill_engines=[
            EntropixEngine(
              model_params, xfmr_weights, mesh, self._tokenizer, xfmr_fn, sample_fn
            )
            for _ in range(num_engines)
          ],
          generate_engines=[
            EntropixEngine(
              model_params, xfmr_weights, mesh, self._tokenizer, xfmr_fn, sample_fn
            )
            for _ in range(num_engines)
          ],
          prefill_params=[model_params] * num_engines,
          generate_params=[model_params] * num_engines,
        )

        self._orchestrator = EntropixOrchestrator(driver)

        # Perform warmup
        await self._warmup()
        self._is_ready = True
        logger.info("Model initialization and warmup complete")

      except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

  async def _warmup(self):
    logger.info("Starting model warmup...")
    warmup_prompt = (
      "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Hello<|eot_id|>"
    )
    warmup_request = ModelRequest(
      tokens=warmup_prompt, max_tokens=10, metadata=Metadata()
    )

    try:
      async for _ in self._orchestrator.decode(warmup_request):
        pass
      logger.info("Warmup complete")
    except Exception as e:
      logger.error(f"Warmup failed: {str(e)}")
      raise

  async def generate_response(
    self, prompt: str, max_tokens: int
  ) -> AsyncGenerator[Tuple[str, List[int]], None]:
    if not self._is_ready:
      raise HTTPException(status_code=503, detail="Model not initialized")

    request = ModelRequest(tokens=prompt, max_tokens=max_tokens, metadata=Metadata())

    async for token_data in self._orchestrator.decode(request):
      yield token_data


# Initialize FastAPI app
app = FastAPI(title="Entropix Model Server")
model_manager = ModelManager()

# Configure CORS
app.add_middleware(
  CORSMiddleware,
  allow_origins=[origin for origin in ALLOWED_ORIGINS],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


# API key validation middleware
@app.middleware("http")
async def validate_api_key(request: Request, call_next):
  if request.url.path == "/health":
    return await call_next(request)

  api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
  if not api_key or api_key not in API_KEYS:
    print(f"Invalid API key: {api_key}")
    print(f"{request.headers}")
    raise HTTPException(status_code=401, detail="Invalid API key")
  return await call_next(request)


async def stream_response(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
  request_id = f"chatcmpl-{uuid.uuid4()}"
  created = int(time.time())

  yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

  prompt = generate_chat_prompt(request)

  try:
    async for token_batch in model_manager.generate_response(
      prompt, request.max_tokens
    ):
      chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": request.model,
        "choices": [
          {"index": idx, "delta": {"content": text}, "finish_reason": None}
          for idx, (text, _) in enumerate(token_batch)
        ],
      }
      yield f"data: {json.dumps(chunk)}\n\n"

    final_chunk = {
      "id": request_id,
      "object": "chat.completion.chunk",
      "created": created,
      "model": request.model,
      "choices": [
        {"index": idx, "delta": {}, "finish_reason": "stop"}
        for idx in range(len(token_batch))
      ],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

  except Exception as e:
    logger.error(f"Error generating response: {e!s}")
    raise HTTPException(status_code=500, detail=str(e))


async def non_streaming_response(request: ChatCompletionRequest):
  request_id = f"chatcmpl-{uuid.uuid4()}"
  created = int(time.time())
  prompt = generate_chat_prompt(request)
  choices = {}

  async for token_batch in model_manager.generate_response(prompt, request.max_tokens):
    for idx, (text, _) in enumerate(token_batch):
      if idx not in choices:
        choices[idx] = {
          "index": idx,
          "message": {"role": "assistant", "content": ""},
          "finish_reason": None,
        }
      choices[idx]["message"]["content"] += text

  for choice in choices.values():
    choice["finish_reason"] = "stop"

  return {
    "id": request_id,
    "object": "chat.completion",
    "created": created,
    "model": request.model,
    "choices": list(choices.values()),
    "usage": {
      "prompt_tokens": 0,  # TODO: Implement token counting
      "completion_tokens": 0,
      "total_tokens": 0,
    },
  }


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
  if not model_manager._is_ready:
    raise HTTPException(status_code=503, detail="Model not initialized")

  if request.stream:
    return StreamingResponse(stream_response(request), media_type="text/event-stream")
  else:
    return await non_streaming_response(request)


@app.get("/health")
async def health_check():
  return {
    "status": "healthy" if model_manager._is_ready else "initializing",
    "model_initialized": model_manager._is_ready,
    "num_gpus": jax.device_count(),
    "model_name": MODEL_NAME,
  }


@app.on_event("startup")
async def startup_event():
  await model_manager.initialize()


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(
    "server_main:app",
    host="0.0.0.0",
    port=8000,
    log_level="info",
    workers=1,  # Important: Keep this at 1 for JAX
  )

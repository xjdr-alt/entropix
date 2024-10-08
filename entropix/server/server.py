import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jax._src.typing import Array
from pydantic import BaseModel
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

class PromptRequest(BaseModel):
  prompt: str
  stream: bool = False

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

@app.get("/health_generate")
async def health_generate(request: Request) -> Response:
  """Check the health of the inference server by generating one token."""
  gri = GenerateReqInput(text="s", sampling_params={"max_new_tokens": 1, "temperature": 0.7})
  try:
    async for _ in tokenizer_manager.generate_request(gri, request):
      break
    return Response(status_code=200)
  except Exception as e:
    logger.exception(e)
    return Response(status_code=503)

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
    logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos + 1], kvcache)  # type: ignore
    next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
    gen_tokens = jnp.concatenate((gen_tokens, next_token))

    # print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
    yield f"{tokenizer.decode(next_token.tolist()[0])}"

    if jnp.isin(next_token, stop).any():
      break

def launch_engine(server_args: ServerArgs):
  """
    Launch the Tokenizer Manager in the main process, the Scheduler in a subprocess, and the Detokenizer Manager in another subprocess.
    """

  global tokenizer_manager

  # Configure global environment
  # configure_logger(server_args)
  format = "[%(asctime)] %(message)s"
  logging.basicConfig(
      level=getattr(logging, server_args.log_level.upper()),
      format=format,
      datefmt="%H:%M:%S",
      force=True,
  )

  # server_args.check_server_args()
  # _set_envs_and_config(server_args)

  # Allocate ports for inter-process communications
  # port_args = PortArgs.init_new(server_args)
  logger.info(f"{server_args=}")

  # If using model from www.modelscope.cn, first download the model.
  server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(server_args.model_path, server_args.tokenizer_path)

  # Launch tensor parallel scheduler processes
  scheduler_procs = []
  scheduler_pipe_readers = []
  tp_size_per_node = server_args.tp_size // server_args.nnodes
  tp_rank_range = range(
      tp_size_per_node * server_args.node_rank,
      tp_size_per_node * (server_args.node_rank + 1),
  )
  for tp_rank in tp_rank_range:
    reader, writer = mp.Pipe(duplex=False)
    gpu_id = tp_rank % tp_size_per_node
    proc = mp.Process(
        target=run_scheduler_process,
        args=(server_args, port_args, gpu_id, tp_rank, writer),
    )
    proc.start()
    scheduler_procs.append(proc)
    scheduler_pipe_readers.append(reader)

  if server_args.node_rank >= 1:
    # For other nodes, they do not need to run tokenizer or detokenizer,
    # so they can just wait here.
    while True:
      pass

  # Launch detokenizer process
  detoken_proc = mp.Process(
      target=run_detokenizer_process,
      args=(
          server_args,
          port_args,
      ),
  )
  detoken_proc.start()

  # Launch tokenizer process
  tokenizer_manager = TokenizerManager(server_args, port_args)
  if server_args.chat_template:
    load_chat_template_for_openai_api(tokenizer_manager, server_args.chat_template)

  # Wait for model to finish loading
  for i in range(len(scheduler_pipe_readers)):
    scheduler_pipe_readers[i].recv()

def launch_server(server_args: ServerArgs):
  """Launch server (based on SGLang Runtime Server)"""

  launch_engine(server_args=server_args)

  # Send a warmup request
  t = threading.Thread(target=_wait_and_warmup, args=(server_args, pipe_finish_writer, os.getpid()))
  t.start()

  try:
    # Listen for HTTP requests
    uvicorn.run(
      app,
      host=server_args.host,
      port=server_args.port,
      log_level=server_args.log_level,
      timeout_keep_alive=5,
      loop="uvloop",
    )
  finally:
    t.join()

if __name__ == "__main__":
  # import uvicorn
  # uvicorn.run(app, host="0.0.0.0", port=8000)
  server_args = tyro.cli(ServerArgs)
  launch_server(server_args)

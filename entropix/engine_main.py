from typing import NamedTuple

import asyncio
from pathlib import Path

import tyro

import jax
import jax.numpy as jnp
from entropix.engine import EntropixEngine, LLAMA_1B_PARAMS
from entropix.orchestrator import EntropixOrchestrator, Driver
from entropix.weights import load_weights
from entropix.tokenizer import Tokenizer
from entropix.model import xfmr
from entropix.sampler import sample

class Metadata:
  def __init__(self):
    self.start_time = None


class Request:
  def __init__(self, tokens: jax.Array, max_tokens: int, metadata: Metadata, is_client_side_tokenization: bool = False):
    self.tokens: jax.Array = tokens
    self.max_tokens: int = max_tokens
    self.metadata: Metadata = metadata
    self.is_client_side_tokenization: bool = is_client_side_tokenization


async def run(ckpt_path: Path = Path('weights/1B-Instruct'), tokenizer_path: str = 'entropix/tokenizer.model'):
  model_params = LLAMA_1B_PARAMS
  xfmr_weights = load_weights(ckpt_path, n_layers=model_params.n_layers)
  tokenizer = Tokenizer(tokenizer_path)
  xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
  sample_fn = jax.jit(sample)
  driver = Driver(
    prefill_engines=[EntropixEngine(model_params, xfmr_weights, tokenizer, xfmr_fn, sample_fn)],
    generate_engines=[EntropixEngine(model_params, xfmr_weights, tokenizer, xfmr_fn, sample_fn)],
    prefill_params=[model_params], # These should be engine specific params
    generate_params=[model_params], # These should be engine specific params
  )

  orchestrator = EntropixOrchestrator(driver)
  prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
  #Think carefully in a step-by-step manner. Can you write a python agent that generates passwords with modern best practices?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  #Think carefully in a step-by-step manner. Oliver picks 44 kiwis on Friday. Then he picks 58 kiwis on Saturday. On Sunday, he picks double the number of kiwis he did on Friday, but five of them were a bit smaller than average. How many kiwis does Oliver have?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  #Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
  metadata = Metadata()
  request = Request(tokens=prompt, max_tokens=4096, metadata=metadata)
  g1 = orchestrator.decode(request)
  g2 = orchestrator.decode(request)
  g3 = orchestrator.decode(request)
  g4 = orchestrator.decode(request)
  async for decoded in g1:
    print(f'LOOP 1: {decoded}')
  async for decoded in g2:
    print(f'LOOP 2: {decoded}')
  async for decoded in g3:
    print(f'LOOP 3: {decoded}')
  async for decoded in g4:
    print(f'LOOP 4: {decoded}')


def main():
  asyncio.run(run())

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_collectives=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
)

if __name__ == '__main__':
  tyro.cli(main)

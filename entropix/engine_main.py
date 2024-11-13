import asyncio
from pathlib import Path
import jax
import tyro

from entropix.config import MODEL_CONFIGS, create_model_params
from entropix.engine import EntropixEngine
from entropix.model import xfmr
from entropix.orchestrator import Driver, EntropixOrchestrator
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights


class Metadata:
    def __init__(self):
        self.start_time = None


class ModelRequest:
    def __init__(
        self,
        tokens: jax.Array,
        max_tokens: int,
        metadata: Metadata,
        is_client_side_tokenization: bool = False,
    ):
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.metadata = metadata
        self.is_client_side_tokenization = is_client_side_tokenization


async def run(
    model_name: str = "1B",
    ckpt_path: Path = Path("weights/1B-Instruct"),
    tokenizer_path: str = "entropix/tokenizer.model",
):
    # Get config for specified model
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model_params = create_model_params(config)

    # Load model components
    xfmr_weights, mesh = load_weights(ckpt_path, model_params)
    tokenizer = Tokenizer(tokenizer_path)
    xfmr_fn = jax.jit(xfmr, static_argnames=("model_params",))
    sample_fn = jax.jit(sample)

    # Use all available devices
    num_engines = jax.device_count()
    print(f"Initializing with {num_engines} devices")

    # Create driver with multiple engines
    driver = Driver(
        prefill_engines=[
            EntropixEngine(
                model_params, xfmr_weights, mesh, tokenizer, xfmr_fn, sample_fn
            )
            for _ in range(num_engines)
        ],
        generate_engines=[
            EntropixEngine(
                model_params, xfmr_weights, mesh, tokenizer, xfmr_fn, sample_fn
            )
            for _ in range(num_engines)
        ],
        prefill_params=[model_params] * num_engines,
        generate_params=[model_params] * num_engines,
    )

    orchestrator = EntropixOrchestrator(driver)

    # Test prompt
    prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

Think carefully in a step-by-step manner. which number is larger, 9.9 or 9.11?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Create requests for testing
    requests = [
        ModelRequest(tokens=prompt, max_tokens=4096, metadata=Metadata())
        for _ in range(1)
    ]

    # Process requests concurrently
    async def process_request(request, idx):
        print(f"\nProcessing request {idx + 1}:")
        async for token_data in orchestrator.decode(request):
            print(f"Request {idx + 1} output:", token_data)

    await asyncio.gather(
        *[process_request(request, i) for i, request in enumerate(requests)]
    )


def main():
    asyncio.run(run())


import os
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

if __name__ == "__main__":
    tyro.cli(main)
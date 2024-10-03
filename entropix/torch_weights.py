from typing import List, NamedTuple


import torch
import jax
import jax.numpy as jnp
import numpy as np

import ml_dtypes

from pathlib import Path

class LayerWeights(NamedTuple):
  wq: torch.Tensor
  wk: torch.Tensor
  wv: torch.Tensor
  wo: torch.Tensor
  w1: torch.Tensor
  w2: torch.Tensor
  w3: torch.Tensor
  ffn_norm: torch.Tensor
  attention_norm: torch.Tensor

class XfmrWeights(NamedTuple):
  tok_embeddings: torch.Tensor
  norm: torch.Tensor
  output: torch.Tensor
  layer_weights: List[LayerWeights]

def compare_outputs(torch_output: torch.Tensor, jax_output: jax.Array, atol: float = 1e-5, rtol: float = 1e-8) -> None:
  jax_output_np = np.array(jax_output)
  torch_output_np = torch_output.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)

  try:
    np.testing.assert_allclose(torch_output_np, jax_output_np, atol=atol, rtol=rtol)
  except AssertionError as e:
    print(f'JAX output (first 30): {jax_output_np.flatten()[:30]}')
    print(f'PyTorch output (first 30): {torch_output_np.flatten()[:30]}')
    raise e

def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 16):
  w = {}
  layer_weights = []
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  with torch.inference_mode():
    for file in ckpt_dir.glob("*.npy"):
      name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
      jax_weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
      #print(f'JAX output (first 30): {jax_weight.flatten()[:30]}')
      np_weight = np.array(jax_weight).astype(np.float32)
      weight = torch.from_numpy(np_weight).to(torch.bfloat16)
      compare_outputs(torch_output=weight, jax_output=jax_weight)
      w[name] = weight.to(device)
    for i in range(n_layers):
      layer_weights.append(LayerWeights(
        wq=w[f'layers.{i}.attention.wq.weight'],
        wk=w[f'layers.{i}.attention.wk.weight'],
        wv=w[f'layers.{i}.attention.wv.weight'],
        wo=w[f'layers.{i}.attention.wo.weight'],
        w1=w[f'layers.{i}.feed_forward.w1.weight'],
        w2=w[f'layers.{i}.feed_forward.w2.weight'],
        w3=w[f'layers.{i}.feed_forward.w3.weight'],
        ffn_norm=w[f'layers.{i}.ffn_norm.weight'],
        attention_norm=w[f'layers.{i}.attention_norm.weight'],
      ))

    xfmr_weights = XfmrWeights(
      tok_embeddings=w['tok_embeddings.weight'],
      norm=w['norm.weight'],
      output=w['output.weight'],
      layer_weights=layer_weights
    )

    return xfmr_weights
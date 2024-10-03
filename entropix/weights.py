from typing import List, NamedTuple

import jax
import jax.numpy as jnp

from pathlib import Path


class LayerWeights(NamedTuple):
  wq: jax.Array
  wk: jax.Array
  wv: jax.Array
  wo: jax.Array
  w1: jax.Array
  w2: jax.Array
  w3: jax.Array
  ffn_norm: jax.Array
  attention_norm: jax.Array


class XfmrWeights(NamedTuple):
  tok_embeddings: jax.Array
  norm: jax.Array
  output: jax.Array
  layer_weights: List[LayerWeights]


def load_weights(ckpt_dir: Path = Path('weights/1B-Instruct'), n_layers: int = 16):
  w = {}
  layer_weights = []
  device = jax.devices("gpu")[0]
  for file in ckpt_dir.glob("*.npy"):
    name = '.'.join(str(file).split('/')[-1].split('.')[:-1])
    weight = jnp.load(file=file, mmap_mode='r', allow_pickle=True)
    #w[name] = weight
    w[name] = jax.device_put(weight, device)
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

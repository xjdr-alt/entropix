from typing import List, NamedTuple
import mlx.core as mx

from pathlib import Path


class LayerWeights(NamedTuple):
  wq: mx.array
  wk: mx.array
  wv: mx.array
  wo: mx.array
  w1: mx.array
  w2: mx.array
  w3: mx.array
  ffn_norm: mx.array
  attention_norm: mx.array


class XfmrWeights(NamedTuple):
  tok_embeddings: mx.array
  norm: mx.array
  output: mx.array
  layer_weights: List[LayerWeights]


def load_weights(ckpt_dir: str, n_layers: int = 16):
  w = {}
  layer_weights = []
  for file in Path(ckpt_dir).glob("*.npy"):
    name = ".".join(str(file).split("/")[-1].split(".")[:-1])
    weight = mx.load(str(file))
    w[name] = weight
  for i in range(n_layers):
    layer_weights.append(
      LayerWeights(
        wq=w[f"layers.{i}.attention.wq.weight"],
        wk=w[f"layers.{i}.attention.wk.weight"],
        wv=w[f"layers.{i}.attention.wv.weight"],
        wo=w[f"layers.{i}.attention.wo.weight"],
        w1=w[f"layers.{i}.feed_forward.w1.weight"],
        w2=w[f"layers.{i}.feed_forward.w2.weight"],
        w3=w[f"layers.{i}.feed_forward.w3.weight"],
        ffn_norm=w[f"layers.{i}.ffn_norm.weight"],
        attention_norm=w[f"layers.{i}.attention_norm.weight"],
      )
    )

  xfmr_weights = XfmrWeights(
    tok_embeddings=w["tok_embeddings.weight"],
    norm=w["norm.weight"],
    output=w["output.weight"],
    layer_weights=layer_weights,
  )

  return xfmr_weights

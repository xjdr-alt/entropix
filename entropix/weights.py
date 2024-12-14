from typing import List, NamedTuple, Optional, Tuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as PS
from jax.experimental import mesh_utils

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


@dataclass
class WeightConfig:
  """Configuration for weight loading and sharding."""

  dp_dim: str = "dp"
  mp_dim: str = "mp"
  fsdp_dim: str = "fsdp"


def create_mesh(device_count: int) -> jax.sharding.Mesh:
  """Creates device mesh for distributed execution."""
  devices = jax.devices()
  mesh_shape = (device_count, 1)
  device_mesh = jax.experimental.mesh_utils.create_device_mesh(mesh_shape)
  return jax.sharding.Mesh(device_mesh, ("mp", "fsdp"))


def create_partition_spec(key):
  dp = "dp"
  mp = "mp"
  fsdp = "fsdp"
  if "norm" in key:
    return PS()
  if "rope.freqs" in key:
    return PS()
  elif "tok_embeddings" in key:
    return PS(fsdp, mp)
  elif "output" in key:
    return PS(fsdp, mp)
  elif "w2" in key or "wo" in key:
    return PS(mp, fsdp)
  else:
    return PS(fsdp, mp)


def load_weights(
  ckpt_dir: Path, model_params, weight_config: Optional[WeightConfig] = None
) -> Tuple[XfmrWeights, jax.sharding.Mesh]:
  """Load and shard model weights across devices."""
  weight_config = weight_config or WeightConfig()
  mesh = create_mesh(jax.device_count())

  w = {}
  layer_weights = []

  for file in ckpt_dir.glob("*.npy"):
    name = ".".join(str(file).split("/")[-1].split(".")[:-1])
    weight = jnp.load(file=file, mmap_mode="r", allow_pickle=True)
    partition_spec = create_partition_spec(name)
    sharding = NamedSharding(mesh, partition_spec)
    if any(lyr in name for lyr in ["wq", "wk", "wv", "wo", "w1", "w2", "w3"]):
      weight = weight.T
      if "wq" in name or "wk" in name or "wv" in name:
        weight = weight.reshape(
          -1,
          model_params.n_local_heads if "wq" in name else model_params.n_local_kv_heads,
          model_params.head_dim,
        )
    # print(name, weight.shape, sharding._to_xla_hlo_sharding(weight.ndim))
    if weight.ndim == 0:
        weight = jnp.stack([weight] * jax.device_count())
    w[name] = jax.device_put(weight, sharding)


  for i in range(model_params.n_layers):
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

  return xfmr_weights, mesh

from typing import NamedTuple
import jax

class RopeParams(NamedTuple):
  use_scaled_rope: bool
  theta: float
  dim: int
  scale_factor: int # = 8
  low_freq_factor: int # = 1
  high_freq_factor: int # = 4
  old_context_len: int # = 8192 (original llama3 length)

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_params: RopeParams

class SamplerParams(NamedTuple):
    steer_tokens: jax.Array
    stop_tokens: jax.Array
    base_temp: float
    base_top_p: float
    base_top_k: int

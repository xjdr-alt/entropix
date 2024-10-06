from typing import NamedTuple
import jax

class RopeParams(NamedTuple):
  rope_theta: float
  use_scaled_rope: bool
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
  d_model: int

class SamplerParams(NamedTuple):
    steer_tokens: jax.Array
    stop_tokens: jax.Array
    base_temp: float
    base_top_p: float
    base_top_k: int

params = {
  "dim": 2048,
  "n_layers": 16,
  "n_heads": 32,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.5,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}
LLAMA_1B_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)
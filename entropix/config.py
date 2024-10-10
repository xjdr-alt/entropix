from typing import NamedTuple

class ScaledRopeParams(NamedTuple):
  scale_factor: int # = 8
  low_freq_factor: int # = 1
  high_freq_factor: int # = 4
  old_context_len: int # = 8192 (original llama3 length)

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  vocab_size: int
  max_seq_len: int
  scaled_rope_params: ScaledRopeParams
  use_scaled_rope: bool
  rope_theta: float

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
  vocab_size=params["vocab_size"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"],
  scaled_rope_params=ScaledRopeParams(
     scale_factor=8,
      low_freq_factor=1,
      high_freq_factor=4,
      old_context_len=8192
  )
)
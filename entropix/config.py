from typing import NamedTuple


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

# params = {
#   "dim": 8192,
#   "n_layers": 80,
#   "n_heads": 64,
#   "n_kv_heads": 8,
#   "vocab_size": 128256,
#   "ffn_dim_multiplier": 1.5,
#   "multiple_of": 256,
#   "norm_eps": 1e-05,
#   "rope_theta": 500000.0,
#   "use_scaled_rope": True,
#   "max_seq_len": 4096
# }

# LLama 3B
params_3b = {
  "dim": 3072,
  "n_layers": 28,
  "n_heads": 24,
  "n_kv_heads": 8,
  "vocab_size": 128256,
  "ffn_dim_multiplier": 1.0,
  "multiple_of": 256,
  "norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "use_scaled_rope": True,
  "max_seq_len": 4096
}

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool


LLAMA_1B_PARAMS = ModelParams(
  n_layers=params["n_layers"],
  n_local_heads=params["n_heads"],
  n_local_kv_heads=params["n_kv_heads"],
  head_dim=params["dim"] // params["n_heads"],
  max_seq_len=params["max_seq_len"],
  rope_theta=params["rope_theta"],
  use_scaled_rope=params["use_scaled_rope"]
)

LLAMA_3B_PARAMS = ModelParams(
  n_layers=params_3b["n_layers"],
  n_local_heads=params_3b["n_heads"],
  n_local_kv_heads=params_3b["n_kv_heads"],
  head_dim=params_3b["dim"] // params_3b["n_heads"],
  max_seq_len=params_3b["max_seq_len"],
  rope_theta=params_3b["rope_theta"],
  use_scaled_rope=params_3b["use_scaled_rope"]
)

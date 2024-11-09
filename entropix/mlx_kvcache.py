import mlx.core as mx


class KVCache:
  def __init__(
    self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int
  ):
    self.k = mx.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=mx.bfloat16)
    self.v = mx.zeros((layers, bsz, max_seq_len, kv_heads, head_dim), dtype=mx.bfloat16)
    self.max_seq_len = max_seq_len

  @classmethod
  def new(
    cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int
  ) -> "KVCache":
    return cls(layers, bsz, max_seq_len, kv_heads, head_dim)

  def update(
    self, xk: mx.array, xv: mx.array, layer_idx: int, cur_pos: int, n_rep: int
  ):
    seq_len = xk.shape[1]

    # Create new k and v arrays
    k_new = mx.zeros_like(self.k)
    v_new = mx.zeros_like(self.v)

    # Copy the existing data
    k_new = mx.concatenate(
      [k_new[:layer_idx], self.k[layer_idx : layer_idx + 1], k_new[layer_idx + 1 :]],
      axis=0,
    )
    v_new = mx.concatenate(
      [v_new[:layer_idx], self.v[layer_idx : layer_idx + 1], v_new[layer_idx + 1 :]],
      axis=0,
    )

    # Update the specific layer
    k_layer = mx.concatenate(
      [k_new[layer_idx, :, :cur_pos], xk, k_new[layer_idx, :, cur_pos + seq_len :]],
      axis=1,
    )
    v_layer = mx.concatenate(
      [v_new[layer_idx, :, :cur_pos], xv, v_new[layer_idx, :, cur_pos + seq_len :]],
      axis=1,
    )

    # Insert the updated layer back
    k_new = mx.concatenate(
      [k_new[:layer_idx], k_layer[None], k_new[layer_idx + 1 :]], axis=0
    )
    v_new = mx.concatenate(
      [v_new[:layer_idx], v_layer[None], v_new[layer_idx + 1 :]], axis=0
    )

    self.k = k_new
    self.v = v_new

    if cur_pos == 0:
      keys = mx.repeat(xk, n_rep, axis=2)
      values = mx.repeat(xv, n_rep, axis=2)
    else:
      keys = mx.repeat(self.k[layer_idx, :, : cur_pos + seq_len], n_rep, axis=2)
      values = mx.repeat(self.v[layer_idx, :, : cur_pos + seq_len], n_rep, axis=2)

    return keys, values, self

  def clear(self):
    self.k = mx.zeros_like(self.k)
    self.v = mx.zeros_like(self.v)

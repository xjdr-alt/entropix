import mlx.core as mx
import mlx.nn as nn


class AttnStats:
  def __init__(self, bsz: int, n_layers: int, n_heads: int):
    self.entropy = mx.zeros((bsz, n_layers, n_heads), dtype=mx.float32)
    self.varentropy = mx.zeros((bsz, n_layers, n_heads), dtype=mx.float32)
    self.n_layers = n_layers
    self.n_heads = n_heads

  @classmethod
  def new(cls, bsz: int, n_layers: int, n_heads: int) -> "AttnStats":
    return cls(bsz, n_layers, n_heads)

  @property
  def avg_entropy(self):
    return mx.mean(self.entropy, axis=-1)  # Average across heads

  @property
  def std_error(self):
    return mx.sqrt(mx.mean(self.varentropy)) / (self.n_heads * self.n_layers)

  def update(self, scores: mx.array, layer_idx: int):
    probs = nn.softmax(scores, axis=-1)
    new_entropy = -mx.sum(mx.where(probs > 0, probs * mx.log(probs), 0), axis=-1)
    new_varentropy = mx.sum(
      probs * (mx.log(probs) + new_entropy[..., None]) ** 2, axis=-1
    )

    self.entropy = mx.concatenate(
      [
        self.entropy[:, :layer_idx],
        new_entropy[None, ...],
        self.entropy[:, layer_idx + 1 :],
      ],
      axis=1,
    )
    self.varentropy = mx.concatenate(
      [
        self.varentropy[:, :layer_idx],
        new_varentropy[None, ...],
        self.varentropy[:, layer_idx + 1 :],
      ],
      axis=1,
    )

    return self

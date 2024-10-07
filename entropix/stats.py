from typing import NamedTuple
import jax
import jax.numpy as jnp

class AttnStats(NamedTuple):
  attn_stats: jax.Array  # (bsz, seqlen, n_layers, num_heads, seqlen+cache_len)
  entropy: jax.Array  # (bsz, seqlen, n_layers, num_heads)
  varentropy: jax.Array  # (bsz, seqlen, n_layers, num_heads)

  @classmethod
  def new(cls, bsz: int, seqlen: int, n_layers: int, n_heads: int) -> 'AttnStats':
    return cls(
        entropy=jnp.zeros((bsz, seqlen, n_layers, n_heads), dtype=jnp.float32),
        varentropy=jnp.zeros((bsz, seqlen, n_layers, n_heads), dtype=jnp.float32)
    )

  @property
  def avg_entropy(self):
    return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads

  @property
  def std_error(self):
    return jnp.sqrt(jnp.mean(self.varentropy)) / (self.n_heads * self.n_layers)

  def update(self, scores: jax.Array, layer_idx: int):
    # scores shape: (bsz, n_heads, seqlen, seqlen)
    probs = jax.nn.softmax(scores, axis=-1)
    new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
    new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)    
    return self._replace(
        scores=scores,
        entropy=self.entropy.at[:, :, layer_idx, :].set(new_entropy.transpose(0,2,1)),
        varentropy=self.varentropy.at[:, :, layer_idx, :].set(new_varentropy.transpose(0,2,1))
    )
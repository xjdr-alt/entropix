from typing import NamedTuple
import jax
import jax.numpy as jnp
from entropix.config import ModelParams

class AttnStats(NamedTuple):
  scores: jax.Array  # (bsz, seqlen, n_layers, num_heads, seqlen)
  entropy: jax.Array  # (bsz, seqlen, n_layers, num_heads)
  varentropy: jax.Array  # (bsz, seqlen, n_layers, num_heads)


  @classmethod
  def new(cls, model_params: ModelParams, bsz: int, max_total_len: int) -> 'AttnStats':
    n_heads, n_layers = model_params.n_local_heads, model_params.n_layers
    return cls(
        entropy=jnp.zeros((bsz, max_total_len, n_layers, n_heads), dtype=jnp.float32),
        varentropy=jnp.zeros((bsz, max_total_len, n_layers, n_heads), dtype=jnp.float32),
        scores=jnp.zeros((bsz, max_total_len, n_layers, n_heads, max_total_len), dtype=jnp.float32)
    )

  @property
  def avg_entropy(self):
    return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads

  @property
  def std_error(self):
    return jnp.sqrt(jnp.mean(self.varentropy)) / (self.n_heads * self.n_layers)

  def update(self, scores: jax.Array, cur_pos: int, layer_idx: int):
    # scores shape: (bsz, n_heads, seqlen, seqlen)
    seqlen = scores.shape[-1]
    probs = jax.nn.softmax(scores, axis=-1)
    new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
    new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)  
    if cur_pos==0:
      return self._replace(
        scores=self.scores.at[:, :seqlen, layer_idx, :, :seqlen].set(scores.transpose(0,2,1,3)),  # (bsz, seqlen, n_layers, n_heads, seqlen) <-- (bsz, seqlen, n_heads, seqlen)
        entropy=self.entropy.at[:, :seqlen, layer_idx, :].set(new_entropy.transpose(0,2,1)),
        varentropy=self.varentropy.at[:, :seqlen, layer_idx, :].set(new_varentropy.transpose(0,2,1))
    )
    else:
      return self._replace(
          scores=self.scores.at[:, cur_pos, layer_idx, :, :].set(scores[:,:,-1,:]),  # (bsz, seqlen, n_layers, n_heads, seqlen) <-- (bsz, seqlen, n_heads, seqlen)
          entropy=self.entropy.at[:, cur_pos, layer_idx, :].set(new_entropy[...,-1]),
          varentropy=self.varentropy.at[:, cur_pos, layer_idx, :].set(new_varentropy[...,-1])
      )
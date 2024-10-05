from typing import NamedTuple
import jax
import jax.numpy as jnp

class AttnStats(NamedTuple):
    entropy: jax.Array  # (bsz, seq_len, n_layers, num_heads)
    varentropy: jax.Array  # (bsz, seq_len, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, seq_len: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=jnp.zeros((bsz, seq_len, n_layers, n_heads), dtype=jnp.float32),
            varentropy=jnp.zeros((bsz, seq_len, n_layers, n_heads), dtype=jnp.float32),
            n_layers=n_layers,
            n_heads=n_heads
        )

    def update(self, scores: jax.Array, layer_idx: int):
        scores = scores.transpose(0, 2, 1, 3)
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = jax.nn.softmax(scores, axis=-1)
        new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
        new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)
        updated_stats = self._replace(
            entropy=self.entropy.at[:, :, layer_idx, :].set(new_entropy),
            varentropy=self.varentropy.at[:, :, layer_idx, :].set(new_varentropy)
        )
        print(updated_stats.entropy.shape)
        return updated_stats
from typing import NamedTuple
import jax
import jax.numpy as jnp

class AttnStats(NamedTuple):
    entropy: jax.Array  # (bsz, n_layers, num_heads)
    varentropy: jax.Array  # (bsz, n_layers, num_heads)
    n_layers: int
    n_heads: int

    @classmethod
    def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
        return cls(
            entropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
            varentropy=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
            n_layers=n_layers,
            n_heads=n_heads
        )

    @property
    def avg_entropy(self):
        # print(f"Entropy shape: {self.entropy.shape}")
        # print(f"Entropy: {self.entropy}")
        return self.entropy.sum(axis=-1, keepdims=False)  # Average across heads
  
    @property
    def std_error(self):
        return jnp.sqrt(jnp.mean(self.varentropy)) / (self.n_heads * self.n_layers)

    def update(self, scores: jax.Array, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, vocab_size)
        probs = jax.nn.softmax(scores, axis=-1)
        new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
        new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)
        
        # print(f"Layer {layer_idx} - Scores shape: {scores.shape}, Probs shape: {probs.shape}")
        # print(f"Layer {layer_idx} - New entropy shape: {new_entropy.shape}, Min: {jnp.min(new_entropy)}, Max: {jnp.max(new_entropy)}")
        
        updated_stats = self._replace(
            entropy=self.entropy.at[:, layer_idx, :].set(new_entropy),
            varentropy=self.varentropy.at[:, layer_idx, :].set(new_varentropy)
        )
        
        # print(f"Layer {layer_idx} - Updated entropy shape: {updated_stats.entropy.shape}")
        # print(f"Layer {layer_idx} - Updated entropy for this layer: {updated_stats.entropy[:, layer_idx, :]}")
        
        return updated_stats
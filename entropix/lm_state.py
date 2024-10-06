from dataclasses import dataclass
from entropix.model import KVCache
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple


class AttnStats(NamedTuple):
  head_ent: jax.Array  # (bsz, n_layers, num_heads)
  head_vent: jax.Array  # (bsz, n_layers, num_heads)
  n_layers: int
  n_heads: int

  @classmethod
  def new(cls, bsz: int, n_layers: int, n_heads: int) -> 'AttnStats':
    return cls(
        head_ent=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        head_vent=jnp.zeros((bsz, n_layers, n_heads), dtype=jnp.float32),
        n_layers=n_layers,
        n_heads=n_heads
    )

  def update(self, scores: jax.Array, layer_idx: int):
    # scores shape: (bsz, n_heads, seqlen, n_words)
    probs = jax.nn.softmax(scores, axis=-1)
    new_ent= -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
    new_vent = jnp.sum(probs * (jnp.log(probs) + new_ent[..., None])**2, axis=-1)
    updated_stats = self._replace(
        head_ent=self.entropy.at[:, layer_idx, :].set(new_ent),
        head_vent=self.varentropy.at[:, layer_idx, :].set(new_vent)
    )
    return updated_stats

@dataclass
class LMState:
    prompt: jax.Array # (bsz, prompt_len)
    gen_tokens: jax.Array # (bsz, seq_len)
    logits: jax.Array # (bsz, n_words)
    kvcache: KVCache
    freqs_cis: jax.Array
    attn_mask: jax.Array
    head_entropy: jax.Array # (bsz, seq_len, n_layers, n_heads) 
    pos: jax.Array # (bsz, max_seq_len)
    state: jax.Array # (bsz, n_states) which state are we? (flow, turn, fork, explore...)
    
    @property
    def context(self) -> jnp.ndarray:
        return jnp.concatenate((self.prompt, self.gen_tokens), axis=1)

    def update(self, scores: jax.Array, layer_idx: int):
        # scores shape: (bsz, n_heads, seqlen, n_words)
        probs = jax.nn.softmax(scores, axis=-1)
        new_entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0), axis=-1)
        new_varentropy = jnp.sum(probs * (jnp.log(probs) + new_entropy[..., None])**2, axis=-1)
        updated_stats = self._replace(
            entropy=self.entropy.at[:, layer_idx, :].set(new_entropy),
            varentropy=self.varentropy.at[:, layer_idx, :].set(new_varentropy)
        )
        return updated_stats
    


from typing import NamedTuple
from dataclasses import dataclass
from entropix.model import KVCache
import math
import jax
import jax.numpy as jnp
from entropix.model import ModelParams

@dataclass
class LMState:
    prompt: jax.Array # (bsz, prompt_len)
    gen_tokens: jax.Array # (bsz, seq_len)
    logits: jax.Array # (bsz, n_words)
    kvcache: KVCache
    freqs_cis: jax.Array 
    attn_mask: jax.Array
    pos: jax.Array # (bsz, max_seq_len)
    state: jax.Array # (bsz, n_states) which state are we? (flow, turn, fork, explore...)
    
    @property
    def context(self) -> jnp.ndarray:
        return jnp.concatenate((self.prompt, self.gen_tokens), axis=1)

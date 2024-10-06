import jax
import jax.numpy as jnp
from typing import NamedTuple
from entropix.config import ModelParams


class LMState(NamedTuple):
    context: jax.Array # (bsz, prompt_len + max_gen_len)
    start_pos: int # should be array in batch case
    logits: jax.Array # (bsz, n_words)
    head_ent: jax.Array # (bsz, buffer_size, n_layers, n_heads) 
    head_vent: jax.Array # (bsz, buffer_size, n_layers, n_heads) 
    cur_pos: int # should be array in the batch case
    
    @classmethod
    def new(cls, model_params: ModelParams, prompt: jax.Array, max_gen_len: int) -> 'LMState':
      """
      Initialize a new LMState object.
      """
      bsz, prompt_len = prompt.shape
      seqlen = prompt_len + max_gen_len
      return cls(
          context=jnp.concatenate([prompt, jnp.zeros((bsz, max_gen_len), dtype=jnp.int32)], axis=1),
          logits=jnp.zeros((bsz, model_params.vocab_size), dtype=jnp.float32),
          head_ent=jnp.zeros((bsz, seqlen, model_params.n_layers, model_params.n_local_heads), dtype=jnp.float32),
          head_vent=jnp.zeros((bsz, seqlen, model_params.n_layers, model_params.n_local_heads), dtype=jnp.float32),
          cur_pos=prompt_len,
          start_pos=prompt_len
      )
        
    def update_context(self, tokens: jax.Array, logits: jax.Array):
      return self._replace(
         logits=logits,
         context=self.context.at[:, self.cur_pos].set(tokens[0]), # fix funny shape issue!
         cur_pos=self.cur_pos + 1,
      )


    def update_attn_stats(self, scores: jax.Array, layer_idx: int):
      probs = jax.nn.softmax(scores[:self.cur_pos+1], axis=-1)
      logprobs = jax.nn.log_softmax(scores[: self.cur_pos+1], axis=-1)
      entropy = -jnp.sum(probs * logprobs, axis=-1)
      varentropy = jnp.sum(jnp.where(probs > 0, probs * (entropy[...,None] + logprobs)**2 , 0), axis=-1)
      entropy, varentropy = entropy.transpose(0,2,1), varentropy.transpose(0,2,1)
      if self.cur_pos == self.start_pos:
         return self._replace(
            head_ent=self.head_ent.at[:, :self.start_pos, layer_idx, :].set(entropy),
            head_vent=self.head_vent.at[:, :self.start_pos, layer_idx, :].set(varentropy)
         )
      else:
        return self._replace(
          head_ent=self.head_ent.at[:, self.cur_pos, layer_idx, :].set(entropy[:,self.cur_pos]),
          head_vent=self.head_vent.at[:,  self.cur_pos, layer_idx, :].set(varentropy[:,self.cur_pos])
        )
    
    @property
    def prompt(self) -> jnp.ndarray:
        return self.context[:, :self.start_pos]


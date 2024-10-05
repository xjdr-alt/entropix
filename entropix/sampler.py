import jax.numpy as jnp
from dataclasses import dataclass
from entropix.tokenizer import Tokenizer
from entropix.config import SamplerParams
from entropix.model import KVCache
from entropix.model import xfmr
from typing import NamedTuple, Tuple
from entropix.LMState import LMState
import math
import jax 

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

def sample(sampler_params: SamplerParams, lm_state: LMState, key=jax.random.PRNGKey(1337)) -> jax.Array:
    temp, top_p, top_k = sampler_params.base_temp, sampler_params.base_top_p, sampler_params.base_top_k
    logits = lm_state.logits
    ent, vent = _calculate_varentropy_logsoftmax(logits)

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 5.0 and vent < 0.1:
        preceding_token, control_tokens = lm_state.gen_tokens[:, -1], sampler_params.control_tokens
        # Insert a control token if not already present
        if not jnp.isin(preceding_token, control_tokens).any():
            # Insert control token
            return select_control_token(logits, control_tokens)
        else:
            # If we've just asked a question, sample with slightly higher temperature
            return _default_sample(logits, temperature=min(1.3, temp * 1.5), top_p=top_p, top_k=top_k)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        # TODO(xjdr): Implement proper branching logic
        # Return top-k tokens to allow for branching
        #top_k_values, top_k_indices = jax.lax.top_k(logits[:, -1], k=top_k)
        #return top_k_indices
        return _default_sample(logits, temperature=min(1.2, temp * 1.5), top_p=top_p, top_k=top_k)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # Use high temperature and min_p sampling
        return _default_sample(logits, temperature=max(2.0, temp * 3), top_p=top_p, top_k=top_k)

    # Middle ground: smooth transition
    else:
        # Interpolate temperature based on entropy and varentropy
        t = jnp.clip((ent + vent) / 10.0, 0.5, 2.0)
        return _default_sample(logits, temperature=t * temp, top_p=top_p, top_k=top_k)

def select_control_token(logits: jnp.ndarray, control_tokens: jax.Array) -> jnp.ndarray:
    n_words = logits.shape[-1]
    control_logits = jnp.take(logits, control_tokens, axis=-1)
    control_argmax = jnp.argmax(control_logits, axis=-1)
    selected_control_tokens = control_tokens[control_argmax]
    return selected_control_tokens.astype(jnp.int32)

@jax.jit
def _calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy

def _default_sample(logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, key=jax.random.PRNGKey(1337)) -> jax.Array:
  bsz = logits.shape[0]
  logit = logits[:, -1]
  probs = jax.nn.softmax(logit / temperature, axis=-1)

  # Apply top-k sampling
  top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
  probs_sort_jax = jnp.flip(top_k_probs, axis=-1)
  probs_idx_jax = jnp.flip(top_k_indices, axis=-1)
  probs_sum_jax = jnp.cumsum(probs_sort_jax, axis=-1)

  # Apply top-p sampling
  mask_jax = jnp.where(probs_sum_jax - probs_sort_jax > top_p, True, False)  # Use jnp.where
  probs_sort_jax = probs_sort_jax * (1 - mask_jax)  # Set values to 0.0 using multiplication
  probs_sort_jax = probs_sort_jax / jnp.sum(probs_sort_jax, axis=-1, keepdims=True)

  next_token_jax = multinomial_sample_one(probs_sort_jax, key)
  next_token_g_jax = jnp.take_along_axis(probs_idx_jax, next_token_jax.reshape(bsz, 1), axis=-1)
  return next_token_g_jax.astype(jnp.int32)

def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
  """Samples one token from a multinomial distribution with sorted probabilities."""
  q = jax.random.exponential(key=key, shape=probs_sort.shape)
  return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)



from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import jax
import jax.numpy as jnp

from entropix.dslider import DSState, adaptive_dirichlet_step, initialize_state
from entropix.dslider_config import DSConfig, DEFAULT_DS_CONFIG


MAX_K = 256
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E


@dataclass
class SamplerConfig:
  # Naked (logits) entropy thresholds
  low_naked_entropy_threshold = 0.3  # Captures most observed LELV cases
  medium_naked_entropy_threshold = 1.2  # Separates medium from high entropy cases
  high_naked_entropy_threshold = 2.5  # Above this we see clear high entropy cases

  # Naked (logits) varentropy thresholds
  low_naked_varentropy_threshold = 1.2  # Most LELV cases are below this
  high_naked_varentropy_threshold = 2.5  # Clear separation for high variance cases

  # Scaffold (attention) metrics thresholds
  # These don't appear in logs, keeping unchanged
  low_scaffold_entropy_threshold = 1.0
  high_scaffold_entropy_threshold = 2.0
  low_scaffold_varentropy_threshold = 0.3
  high_scaffold_varentropy_threshold = 0.8


@partial(jax.jit, static_argnames=("config",))
def sample(
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  clarifying_question_token: int = 2564,
  key=jax.random.PRNGKey(1337),
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
  cfg = SamplerConfig()
  bsz = logits.shape[0]
  (
    new_state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    naked_token_logprob,
    scaffold_token_logprob,
  ) = adaptive_dirichlet_step(key, state, logits, config)
  new_token = new_token.reshape((bsz, 1))

  def _and(*args):
    res = True
    for a in args:
      res = jax.lax.bitwise_and(res, a)
    return res

  def sample_one(
    idx,
    logit,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    loops=0,
  ):
    LELV = _and(
      naked_ent < cfg.low_naked_entropy_threshold,
      naked_varent < cfg.low_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    HELV = _and(
      naked_ent > cfg.high_naked_entropy_threshold,
      naked_varent < cfg.low_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent < cfg.low_scaffold_varentropy_threshold,
    ).astype(float)

    LEHV = _and(
      naked_ent < cfg.high_naked_entropy_threshold,
      naked_varent > cfg.high_naked_varentropy_threshold,
      # scaffold_ent < cfg.low_scaffold_entropy_threshold,
      # scaffold_varent > cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    HEHV = _and(
      naked_ent > cfg.medium_naked_entropy_threshold,
      naked_varent > cfg.high_naked_varentropy_threshold,
      # scaffold_ent > cfg.high_scaffold_entropy_threshold,
      # scaffold_varent > cfg.high_scaffold_varentropy_threshold,
    ).astype(float)

    case = jnp.argmax(jnp.hstack([LELV, HELV, LEHV, HEHV]))

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    def lelv():
      # jax.debug.print("LELV Naked Ent: {}", naked_ent)
      # jax.debug.print("LELV Naked Varent: {}", naked_varent)
      # jax.debug.print("LELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LELV Scaffold Varent: {}\n", scaffold_varent)
      return new_token, state

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    def helv():
      # jax.debug.print("HELV Naked Ent: {}", naked_ent)
      # jax.debug.print("HELV Naked Varent: {}", naked_varent)
      # jax.debug.print("HELV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HELV Scaffold Varent: {}\n", scaffold_varent)
      return jnp.array([2564]), state

    # Low Entropy, High Varentropy: "exploring forks in the path"
    def lehv():
      # jax.debug.print("LEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("LEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("LEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("LEHV Scaffold Varent: {}\n", scaffold_varent)
      # TODO(xjdr): We need to do a differnt version of tree search here with constant return dimensions
      return new_token, state

    # High Entropy, High Varentropy: "resampling in the mist"
    def hehv():
      # jax.debug.print("HEHV Naked Ent: {}", naked_ent)
      # jax.debug.print("HEHV Naked Varent: {}", naked_varent)
      # jax.debug.print("HEHV Scaffold Ent: {}\n", scaffold_ent)
      # jax.debug.print("HEHV Scaffold Varent: {}\n", scaffold_varent)
      plogit = logit.at[new_token].set(float("-inf"))

      # Run ADS with single batch
      (
        new_state,
        resampled_token,
        *_,  # Other metrics
      ) = adaptive_dirichlet_step(
        key,
        jax.tree_map(lambda x: x[None, ...], state),
        plogit[None, ...],  # Shape (1, vocab)
        DEFAULT_DS_CONFIG,
      )
      return resampled_token, jax.tree_map(lambda x: jnp.bfloat16(x[-1]), new_state)

    def default():
      # jax.debug.print("Default Naked Ent: {}", naked_ent)
      # jax.debug.print("Default Naked Varent: {}", naked_varent)
      return new_token, state

    return jax.lax.switch(case, (lelv, helv, lehv, hehv, default))

  result, new_state = jax.vmap(sample_one)(
    jnp.arange(bsz),
    logits,
    state,
    new_token,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
  )
  return result.reshape((bsz, 1)), new_state

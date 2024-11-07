from functools import partial
from typing import NamedTuple, Optional, Tuple

from entropix.dslider_tuning import OnlineTuner
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from entropix.dslider_config import EPS, MAX_TEMP, MIN_TEMP, DSConfig
from entropix.dslider_utils import *


@jax.jit
def kl_divergence(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
  """Compute KL divergence between two log probability distributions."""
  p = jnp.exp(logp)
  return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)


@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Compute entropy and varentropy from log probabilities."""
  p = jnp.exp(logp)
  ent = -jnp.sum(p * logp, axis=-1)
  diff = logp + ent[..., None]
  varent = jnp.sum(p * diff**2, axis=-1)
  return ent, varent


@jax.jit
def normalize_logits(logits: jnp.ndarray, noise_floor: float) -> jnp.ndarray:
  """Normalize logits to log probabilities with noise floor truncation."""
  shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
  normalized = shifted - jax.nn.logsumexp(shifted + EPS, axis=-1, keepdims=True)
  # noise floor calculated for bfloat16
  return jnp.where(normalized < noise_floor, jnp.log(EPS), normalized)


class DSState(NamedTuple):
  emwa_dir: jnp.ndarray
  emwa_logp_on_supp: jnp.ndarray
  emwa_temp: jnp.ndarray
  emwa_ent_scaffold: jnp.ndarray
  emwa_ent_naked: jnp.ndarray
  emwa_varent_scaffold: jnp.ndarray
  emwa_varent_naked: jnp.ndarray
  token_cross_ent_scaffold: jnp.ndarray
  token_cross_ent_naked: jnp.ndarray
  token_cross_var_scaffold: jnp.ndarray
  token_cross_var_naked: jnp.ndarray
  emwa_dir_ent: jnp.ndarray
  emwa_topk_ent_naked: jnp.ndarray


@partial(jax.jit, static_argnames=("bsz", "config", "dtype"))
def initialize_state(
  logits: jax.Array, bsz: int, config: DSConfig, dtype=jnp.bfloat16
) -> DSState:
  _, seqlen, _ = logits.shape
  logprobs = normalize_logits(logits, config.noise_floor)
  ent, varent = ent_varent(logprobs)
  avg_ent, avg_varent = ent.mean(axis=-1), varent.mean(axis=-1)

  topk_logits, topk_indices = jax.lax.top_k(logprobs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  topk_ent, _ = ent_varent(topk_logprobs)
  avg_topk_ent = topk_ent.mean(axis=-1)

  logprobs_on_supp = normalize_logits(
    logits[..., config.dirichlet_support], config.noise_floor
  )
  avg_logprobs_on_supp = jnp.mean(logprobs_on_supp, axis=1)

  initial_dir, _, _ = fit_dirichlet(avg_logprobs_on_supp)
  avg_dir_ent = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, initial_dir[:, None, :]
  ).mean(axis=-1)

  topk_token_logprobs = jnp.take_along_axis(logprobs, topk_indices, axis=-1)
  initial_cross_ent_naked = -topk_token_logprobs.mean(axis=(1, 2))
  initial_cross_var_naked = topk_token_logprobs.var(axis=(1, 2))

  state = DSState(
    emwa_dir=initial_dir.repeat(bsz, axis=0),
    emwa_logp_on_supp=avg_logprobs_on_supp.repeat(bsz, axis=0),
    emwa_temp=jnp.ones((bsz,), dtype=dtype),
    emwa_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    emwa_ent_naked=avg_ent.repeat(bsz, axis=0),
    emwa_varent_scaffold=jnp.zeros((bsz,), dtype=dtype),
    emwa_varent_naked=avg_varent.repeat(bsz, axis=0),
    token_cross_ent_scaffold=avg_ent.repeat(bsz, axis=0),
    token_cross_ent_naked=initial_cross_ent_naked.repeat(bsz, axis=0),
    token_cross_var_scaffold=jnp.zeros((bsz,), dtype=dtype),
    token_cross_var_naked=initial_cross_var_naked.repeat(bsz, axis=0),
    emwa_dir_ent=avg_dir_ent.repeat(bsz, axis=0),
    emwa_topk_ent_naked=avg_topk_ent.repeat(bsz, axis=0),
  )
  return state


@partial(jax.jit, static_argnames=("config", "tuner", "wild"))
def adaptive_dirichlet_step(
  key: jax.random.PRNGKey,
  state: DSState,
  logits: jnp.ndarray,
  config: DSConfig,
  tuner: Optional[OnlineTuner] = None,
  wild: bool = True,
) -> Tuple[DSState, jnp.ndarray]:
  dtype = logits.dtype
  bsz, vsz = logits.shape
  output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
  EPS = jnp.array(1e-8, dtype=dtype)
  naked_log_probs = normalize_logits(logits, config.noise_floor)
  # update naked entropy rate
  naked_ent, naked_varent = ent_varent(naked_log_probs)
  # fix shape issue!
  new_emwa_ent_naked = update_emwa(
    naked_ent, state.emwa_ent_naked, config.emwa_ent_naked_coeff
  )
  new_emwa_varent_naked = update_emwa(
    naked_varent, state.emwa_varent_naked, config.emwa_varent_naked_coeff
  )
  # entropy and varentropy vectors - shape (bsz, 4)
  state_ent = jnp.array(
    [
      state.token_cross_ent_scaffold,
      state.token_cross_ent_naked,
      state.emwa_ent_scaffold,
      state.emwa_ent_naked,
    ]
  ).T  # TODO(doomslide): add dirichlet expected entropy...
  state_std = jnp.sqrt(
    jnp.array(
      [
        state.token_cross_var_scaffold,
        state.token_cross_var_naked,
        state.emwa_varent_scaffold,
        state.emwa_varent_naked,
      ]
    )
  ).T  # TODO(doomslide): add dirichlet expected std...
  outlier_threshold = compute_outlier_threshold(
    state_ent, state_std, naked_ent, naked_varent, config
  )
  outlier_mask = outlier_threshold > 0
  # update emwa topk entropy
  topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
  topk_logprobs = normalize_logits(topk_logits, config.noise_floor)
  naked_topk_ent, _ = ent_varent(topk_logprobs)
  new_emwa_topk_ent_naked = update_emwa(
    naked_topk_ent, state.emwa_topk_ent_naked, config.emwa_topk_ent_naked_coeff
  )
  """
  argmax policy for concentrated inliers
  """
  argmax_threshold = (
    config.argmax_threshold.weight * state.emwa_topk_ent_naked
    + config.argmax_threshold.bias
  )
  argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold)
  argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
  argmax_tokens = jnp.take_along_axis(
    topk_indices, argmax_indices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
  """
  topk temperature tuning policy for dispersed inliers
  """
  inlier_sampling_indices = ~outlier_mask & ~argmax_mask
  inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
  sampling_inlier_choices = jax.random.categorical(
    key, topk_logprobs / inlier_sampling_temp[:, None]
  )
  sampling_inlier_tokens = jnp.take_along_axis(
    topk_indices, sampling_inlier_choices[:, None], axis=-1
  ).squeeze(1)
  output_tokens = jnp.where(
    inlier_sampling_indices, sampling_inlier_tokens, output_tokens
  )
  """
  tune temperature of outliers to match target entropy
  """
  target_entropy = (
    jnp.dot(state_ent, config.target_entropy.linear)
    + jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1)
    + config.target_entropy.bias
  )
  temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy)
  new_emwa_temp = update_emwa(temp, state.emwa_temp, config.emwa_temp_coeff)
  tuned_logprobs = normalize_logits(
    naked_log_probs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP), config.noise_floor
  )
  """
  update emwa logp (on dirichlet support)
  """
  logprobs_on_supp = normalize_logits(
    tuned_logprobs[:, config.dirichlet_support], config.noise_floor
  )
  kl = jnp.sum(
    jnp.exp(logprobs_on_supp) * (logprobs_on_supp - state.emwa_logp_on_supp), axis=-1
  )
  emwa_logp_coeff = config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  new_emwa_logp_on_supp = update_emwa(
    logprobs_on_supp, state.emwa_logp_on_supp, emwa_logp_coeff[..., None]
  )
  new_emwa_dir, _, _ = fit_dirichlet(new_emwa_logp_on_supp)
  """
  update dirichlet and compute threshold
  """
  dir_log_likelihood = dirichlet_log_likelihood_from_logprob(
    logprobs_on_supp, state.emwa_dir
  )
  new_emwa_dir_ent = update_emwa(
    -dir_log_likelihood, state.emwa_dir_ent, config.emwa_dir_ent_coeff
  )
  dirichlet_threshold = (
    config.dirichlet_threshold.weight * state.emwa_dir_ent
    + config.dirichlet_threshold.bias
  )
  use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
  if wild:  # if wild, sample from dirichlet, else use expectation
    dir_probs = sample_dirichlet(key, new_emwa_dir)
  else:
    dir_probs = dirichlet_expectation(new_emwa_dir)
  """
  below dirichlet threshold, interpolate and sample (can improve this in the future)
  """
  kl = jnp.sum(dir_probs * (jnp.log(dir_probs + EPS) - logprobs_on_supp), axis=-1)
  perturb_coeff = 1 - jnp.pow(
    config.perturb_base_coeff, -config.perturb_exp_coeff * (1 / (kl + EPS))
  )
  interpolated_probs = perturb_coeff[:, None] * dir_probs + (
    1 - perturb_coeff[:, None]
  ) * jnp.exp(logprobs_on_supp)
  # in use_dirichlet case take argmax of the slided probs
  dicihlet_choices = jnp.argmax(interpolated_probs, axis=-1)
  dirichlet_tokens = jnp.take(config.dirichlet_support, dicihlet_choices)
  output_tokens = jnp.where(use_dirichlet, dirichlet_tokens, output_tokens)
  """
  above dirichlet threshold youre ngmi
  """
  ood_choices = jax.random.categorical(key, jnp.log(dir_probs + EPS))
  ood_tokens = jnp.take(config.dirichlet_support, ood_choices)
  output_tokens = jnp.where(outlier_mask & ~use_dirichlet, ood_tokens, output_tokens)
  # update scaffold entropy rate
  scaffold_ent, scaffold_varent = ent_varent(jnp.log(interpolated_probs + EPS))
  new_emwa_ent_scaffold = update_emwa(
    scaffold_ent, state.emwa_ent_scaffold, config.emwa_ent_scaffold_coeff
  )
  new_emwa_varent_scaffold = update_emwa(
    scaffold_varent, state.emwa_varent_scaffold, config.emwa_varent_scaffold_coeff
  )
  # update token cross entropies
  batch_indices = jnp.arange(bsz)
  scaffold_token_logprob = jnp.log(
    interpolated_probs[batch_indices, output_tokens] + EPS
  )
  naked_token_logprob = jnp.log(naked_log_probs[batch_indices, output_tokens] + EPS)
  (
    new_token_cross_ent_scaffold,
    new_token_cross_ent_naked,
    new_token_cross_var_scaffold,
    new_token_cross_var_naked,
  ) = update_token_cross_entropies(
    state, scaffold_token_logprob, naked_token_logprob, config
  )
  if tuner:
    config = tuner.update(
        jnp.log(interpolated_probs + EPS),
        naked_log_probs,
        new_token_cross_ent_naked,
        new_token_cross_ent_scaffold
    )
  # assemble new state
  new_state = DSState(
    emwa_dir=new_emwa_dir,
    emwa_logp_on_supp=new_emwa_logp_on_supp,
    emwa_temp=new_emwa_temp,
    emwa_ent_scaffold=new_emwa_ent_scaffold,
    emwa_ent_naked=new_emwa_ent_naked,
    emwa_varent_scaffold=new_emwa_varent_scaffold,
    emwa_varent_naked=new_emwa_varent_naked,
    token_cross_ent_scaffold=new_token_cross_ent_scaffold,
    token_cross_ent_naked=new_token_cross_ent_naked,
    token_cross_var_scaffold=new_token_cross_var_scaffold,
    token_cross_var_naked=new_token_cross_var_naked,
    emwa_dir_ent=new_emwa_dir_ent,
    emwa_topk_ent_naked=new_emwa_topk_ent_naked,
  )

  if tuner:
    tuner.idx += 1
    if tuner.idx < tuner.max_idx:
      return adaptive_dirichlet_step(key, new_state, logits, config, tuner, wild)

  return (
    new_state,
    output_tokens,
    naked_ent,
    naked_varent,
    scaffold_ent,
    scaffold_varent,
    naked_token_logprob,
    scaffold_token_logprob,
  )


@jax.jit
def update_emwa(new: jax.Array, old: jax.Array, coeff: float | jax.Array) -> jax.Array:
  return coeff * new + (1 - coeff) * old


@partial(jax.jit, static_argnames=("config",))
def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
  return (
    jnp.einsum("bi,ij,bj->b", state_ent, config.outlier_threshold.bilinear, state_std)
    + jnp.einsum("bi,i->b", state_ent, config.outlier_threshold.linear_state_ent)
    + jnp.einsum("bi,i->b", state_std, config.outlier_threshold.linear_state_std)
    + naked_ent * config.outlier_threshold.linear_naked_ent
    + naked_varent * config.outlier_threshold.linear_naked_varent
    + config.outlier_threshold.bias
  )


@partial(jax.jit, static_argnames=("config",))
def update_dirichlet_params(tuned_logprobs_on_supp, state, config):
  kl = kl_divergence(tuned_logprobs_on_supp, state.emwa_logp_on_supp)
  emwa_logp_coeff = (
    config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS))
  )[:, None]
  new_emwa_logp_dir_sup = (
    emwa_logp_coeff * tuned_logprobs_on_supp
    + (1 - emwa_logp_coeff) * state.emwa_logp_on_supp
  )
  new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
  return new_dir_params, new_emwa_logp_dir_sup


@jax.jit
def update_token_cross_entropies(
  state: DSState,
  scaffold_token_logprob: jnp.ndarray,
  naked_token_logprob: jnp.ndarray,
  config: DSConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Update token cross entropy statistics."""
  token_cross_ent_naked = (
    config.token_cross_ent_naked_coeff * (-naked_token_logprob)
    + (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
  )
  token_cross_ent_scaffold = (
    config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob)
    + (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
  )
  token_cross_var_naked = (
    config.token_cross_var_naked_coeff
    * (token_cross_ent_naked - naked_token_logprob) ** 2
    + (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
  )
  token_cross_var_scaffold = (
    config.token_cross_var_scaffold_coeff
    * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2
    + (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
  )
  return (
    token_cross_ent_scaffold,
    token_cross_ent_naked,
    token_cross_var_scaffold,
    token_cross_var_naked,
  )

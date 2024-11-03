from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from functools import partial

from entropix.dslider_utils import temp_tune, fit_dirichlet
from entropix.dslider_config import DSConfig, EPS, MIN_TEMP, MAX_TEMP


@jax.jit
def kl_divergence(logp: jnp.ndarray, logq: jnp.ndarray) -> jnp.ndarray:
    """Compute KL divergence between two log probability distributions."""
    p = jnp.exp(logp)
    return jnp.sum(jnp.where(p > 0, p * (logp - logq), 0.0), axis=-1)

@jax.jit
def ent_varent(logp: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute entropy and variance from log probabilities."""
    p = jnp.exp(logp)
    ent = -jnp.sum(p * logp, axis=-1)
    diff = logp + ent[:, None]  # broadcasting
    varent = jnp.sum(p * diff**2, axis=-1)
    return ent, varent

@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
    """Sample from a Dirichlet distribution."""
    gamma_samples = jax.random.gamma(
        key,
        alpha,
        shape=alpha.shape
    )
    return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)

class DSState(NamedTuple):
    """State maintained by the Adaptive Dirichlet Sampler"""
    emwa_dir: jnp.ndarray
    emwa_logp_dir_supp: jnp.ndarray
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

@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expectation E[X|X~Dir(alpha)]"""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
    return alpha / alpha_sum

@jax.jit
def dirichlet_expected_entropy(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected entropy of a Dirichlet distribution."""
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # alpha_0
    K = alpha.shape[-1]

    # ln B(alpha) term
    log_beta = jnp.sum(jsp.special.gammaln(alpha), axis=-1) - jsp.special.gammaln(alpha_sum.squeeze())

    # (alpha_0 - K)ψ(alpha_0) term
    digamma_sum = jsp.special.digamma(alpha_sum)
    second_term = (alpha_sum.squeeze() - K) * digamma_sum.squeeze()

    # -sum((alpha_j - 1)ψ(alpha_j)) term
    digamma_alpha = jsp.special.digamma(alpha)
    third_term = -jnp.sum((alpha - 1) * digamma_alpha, axis=-1)

    return log_beta + second_term + third_term

@jax.jit
def dirichlet_log_likelihood_from_logprob(logprobs: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute log probability of probs under Dirichlet(alpha)."""
    return jnp.sum((alpha - 1.0) * logprobs, axis=-1) - jsp.special.gammaln(jnp.sum(alpha, axis=-1)) + jnp.sum(jsp.special.gammaln(alpha), axis=-1)

@jax.jit
def dirichlet_expected_varentropy(alpha: jnp.ndarray) -> jnp.ndarray:
    """Compute the expected varentropy E[∑ᵢ ln(Xᵢ)² * Xᵢ] of a Dirichlet distribution.

    Args:
        alpha: Dirichlet parameters of shape (..., K)

    Returns:
        Expected varentropy of shape (...)
    """
    alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # α₀

    # E[Xᵢ] = αᵢ/α₀
    expected_x = alpha / alpha_sum

    # ψ(αᵢ)² + ψ₁(αᵢ) term
    digamma_alpha = jsp.special.digamma(alpha)
    trigamma_alpha = jsp.special.polygamma(1, alpha)
    squared_plus_deriv = digamma_alpha**2 + trigamma_alpha

    # Sum over dimensions: ∑ᵢ (αᵢ/α₀) * (ψ₁(αᵢ) + ψ(αᵢ)²)
    return jnp.sum(expected_x * squared_plus_deriv, axis=-1)


@partial(jax.jit, static_argnames=('bsz', 'vsz', 'config', 'dtype'))
def initialize_state(bsz: int, vsz: int, config: DSConfig, dtype=jnp.bfloat16) -> DSState:
    """Initialize the DSState with specified dtype."""
    state = DSState(
        emwa_dir=jnp.ones((bsz, config.dirichlet_support.size), dtype=dtype),
        emwa_logp_dir_supp=jnp.zeros((bsz, config.dirichlet_support.size), dtype=dtype),
        emwa_temp=jnp.ones((bsz,), dtype=dtype),

        emwa_ent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        emwa_ent_naked=jnp.zeros((bsz,), dtype=dtype),
        emwa_varent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        emwa_varent_naked=jnp.zeros((bsz,), dtype=dtype),

        token_cross_ent_scaffold=jnp.zeros((bsz,), dtype=dtype),
        token_cross_ent_naked=jnp.zeros((bsz,), dtype=dtype),
        token_cross_var_scaffold=jnp.zeros((bsz,), dtype=dtype),
        token_cross_var_naked=jnp.zeros((bsz,), dtype=dtype),

        emwa_dir_ent=jnp.zeros((bsz,), dtype=dtype),
        emwa_topk_ent_naked=jnp.zeros((bsz,), dtype=dtype)
    )
    return state

@partial(jax.jit, static_argnames=('config',))
def adaptive_dirichlet_step(
    key: jax.random.PRNGKey,
    state: DSState,
    logits: jnp.ndarray,
    config: DSConfig,
    wild: bool = True
) -> Tuple[DSState, jnp.ndarray]:
    """Single step of the Adaptive Dirichlet Sampler."""
    dtype = logits.dtype
    bsz, _ = logits.shape
    output_tokens = jnp.zeros(bsz, dtype=jnp.int32)
    # Constants cast to dtype
    EPS = jnp.array(1e-8, dtype=dtype)
    # normalize logits
    naked_log_probs = normalize_logits(logits)
    # update naked entropy rate
    naked_ent, naked_varent = ent_varent(naked_log_probs)
    new_emwa_ent_naked = (
        config.emwa_ent_naked_coeff * naked_ent  +
        (1 - config.emwa_ent_naked_coeff) * state.emwa_ent_naked
    )
    new_emwa_varent_naked = (
        config.emwa_varent_naked_coeff * (naked_varent) +
        (1 - config.emwa_varent_naked_coeff) * state.emwa_varent_naked
    )
    # entropy and varentropy vectors - shape (bsz, 4)
    state_ent = jnp.array([
        state.token_cross_ent_scaffold,
        state.token_cross_ent_naked,
        state.emwa_ent_scaffold,
        state.emwa_ent_naked
    ]).T # TODO: add dirichlet expected entropy...
    state_std = jnp.sqrt(jnp.array([
        state.token_cross_var_scaffold,
        state.token_cross_var_naked,
        state.emwa_varent_scaffold,
        state.emwa_varent_naked
    ])).T # TODO: add dirichlet expected std...
    outlier_threshold = compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config)
    outlier_mask = outlier_threshold > 0
    # extract topk
    topk_logits, topk_indices = jax.lax.top_k(naked_log_probs, config.outlier_topk)
    # update emwa topk entropy
    topk_logprobs = normalize_logits(topk_logits)
    naked_topk_ent, _ = ent_varent(topk_logprobs)
    new_emwa_topk_ent_naked = config.emwa_topk_ent_naked_coeff * naked_topk_ent + (1 - config.emwa_topk_ent_naked_coeff) * state.emwa_topk_ent_naked
    """
    argmax policy for concentrated inliers
    """
    argmax_threshold = config.argmax_threshold.weight * state.emwa_topk_ent_naked + config.argmax_threshold.bias
    argmax_mask = ~outlier_mask & (naked_topk_ent < argmax_threshold)
    # Get indices of maximum probabilities within top-k
    argmax_indices = jnp.argmax(topk_logprobs, axis=-1)
    # Map these indices back to the original token space using topk_indices
    argmax_tokens = jnp.take_along_axis(topk_indices, argmax_indices[:, None], axis=1).squeeze(1)
    # Only use these tokens where argmax_mask is True
    output_tokens = jnp.where(argmax_mask, argmax_tokens, output_tokens)
    """
    topk temperature tuning policy for dispersed inliers
    """
    inlier_sampling_indices = ~outlier_mask & ~argmax_mask
    # Handle less confident inliers by sampling with entropy-tuned temperature
    inlier_sampling_temp, _, _ = temp_tune(topk_logprobs, state.emwa_topk_ent_naked)
    sampling_inlier_choices = jax.random.categorical(key, topk_logprobs / inlier_sampling_temp[:, None])
    sampling_inlier_tokens = jnp.take_along_axis(topk_indices, sampling_inlier_choices[:, None], axis=1).squeeze(1)
    output_tokens = jnp.where(inlier_sampling_indices, sampling_inlier_tokens, output_tokens)
    """
    target entropy = affine function of state_ent and inverse emwa temperature
    """
    # outlier target entropy is affine function of state_ent and inverse emwa temperature
    target_entropy = (
        jnp.dot(state_ent, config.target_entropy.linear) +
        jnp.sum(config.target_entropy.linear_inv_temp / state.emwa_temp, axis=-1) +
        config.target_entropy.bias
    )
    temp, _, _ = temp_tune(naked_log_probs.astype(jnp.float32), target_entropy)
    # update emwa temperature
    new_emwa_temp = config.emwa_temp_coeff * temp + (1 - config.emwa_temp_coeff) * state.emwa_temp
    """
    tune temperature and update emwa logp on dirichlet support
    """
    # scale log probabilities
    tuned_logprobs = normalize_logits(naked_log_probs / jnp.clip(temp[:, None], MIN_TEMP, MAX_TEMP))
    """
    update emwa logp and dirichlet parameters
    """
    dir_support_logp = normalize_logits(tuned_logprobs[:, config.dirichlet_support])
    new_emwa_dir, new_emwa_logp_dir_sup = update_dirichlet_params(dir_support_logp, state, config)
    """
    update Dirichlet entropy
    """
    dir_log_likelihood = dirichlet_log_likelihood_from_logprob(dir_support_logp, state.emwa_dir)
    new_emwa_dir_ent = (
        config.emwa_dir_ent_coeff * (-dir_log_likelihood) +
        (1 - config.emwa_dir_ent_coeff) * state.emwa_dir_ent
    )
    dirichlet_threshold = config.dirichlet_threshold.weight * state.emwa_dir_ent + config.dirichlet_threshold.bias
    use_dirichlet = outlier_mask & (-dir_log_likelihood < dirichlet_threshold)
    """
    below dirichlet threshold, interpolate and sample (can improve this in the future)
    """
    # compute perturbation coefficient
    dir_expectation = dirichlet_expectation(state.emwa_dir)
    kl_div = dirichlet_expected_entropy(state.emwa_dir) - jnp.sum(dir_expectation * dir_support_logp, axis=-1)
    perturb_coeff = 1 - jnp.pow(config.perturb_base_coeff, - config.perturb_exp_coeff * (1 / (kl_div + EPS)))
    # Calculate interpolated probabilities for the support tokens
    interpolated_probs = (
        perturb_coeff[:, None] * dir_expectation +
        (1 - perturb_coeff[:, None]) * jnp.exp(dir_support_logp)
    )
    # For use_dirichlet case: sample from support space then map back
    interpolated_choices = jnp.argmax(interpolated_probs, axis=-1)
    dirichlet_tokens = jnp.take(config.dirichlet_support, interpolated_choices)
    output_tokens = jnp.where(use_dirichlet, dirichlet_tokens, output_tokens)
    """
    above dirichlet threshold youre ngmi
    """
    if wild:
        # sample from random dirichlet distributed
        sampled_probs = sample_dirichlet(key, new_emwa_dir)
        ood_choices = jax.random.categorical(key, jnp.log(sampled_probs + EPS))
        ood_tokens = jnp.take(config.dirichlet_support, ood_choices)
    else:
        # sample from the pure tuned logprobs
        support_choices = jax.random.categorical(key, tuned_logprobs)
        ood_tokens = jnp.take(config.dirichlet_support, support_choices)
    # Update output tokens where appropriate
    output_tokens = jnp.where(
        outlier_mask & ~use_dirichlet,
        ood_tokens,
        output_tokens
    )
    # update scaffold entropy rate
    scaffold_ent, scaffold_varent =  ent_varent(jnp.log(interpolated_probs + EPS))
    new_emwa_ent_scaffold = (
        config.emwa_ent_scaffold_coeff * scaffold_ent +
        (1 - config.emwa_ent_scaffold_coeff) * state.emwa_ent_scaffold
    )
    new_emwa_varent_scaffold = (
        config.emwa_varent_scaffold_coeff * scaffold_varent +
        (1 - config.emwa_varent_scaffold_coeff) * state.emwa_varent_scaffold
    )
    # update token cross entropies
    batch_indices = jnp.arange(bsz)
    scaffold_token_logprob = jnp.log(interpolated_probs[batch_indices, output_tokens] + EPS)
    naked_token_logprob = jnp.log(naked_log_probs[batch_indices, output_tokens] + EPS)
    new_token_cross_ent_scaffold, new_token_cross_ent_naked, new_token_cross_var_scaffold, new_token_cross_var_naked = update_token_cross_entropies(
        state,
        scaffold_token_logprob,
        naked_token_logprob,
        config
    )
    # assemble new state
    new_state = DSState(
        emwa_dir=new_emwa_dir,
        emwa_logp_dir_supp=new_emwa_logp_dir_sup,
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
        emwa_topk_ent_naked=new_emwa_topk_ent_naked
    )

    return new_state, output_tokens, naked_ent, naked_varent, scaffold_ent, scaffold_varent, naked_token_logprob, scaffold_token_logprob

@jax.jit
def normalize_logits(logits: jnp.ndarray) -> jnp.ndarray:
    """Normalize logits to log probabilities with numerical stability."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    return shifted - jax.nn.logsumexp(shifted, axis=-1, keepdims=True)

@jax.jit
def update_token_cross_entropies(
    state: DSState,
    scaffold_token_logprob: jnp.ndarray,
    naked_token_logprob: jnp.ndarray,
    config: DSConfig
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Update token cross entropy statistics."""
    token_cross_ent_scaffold = (
        config.token_cross_ent_scaffold_coeff * (-scaffold_token_logprob) +
        (1 - config.token_cross_ent_scaffold_coeff) * state.token_cross_ent_scaffold
    )
    token_cross_ent_naked = (
        config.token_cross_ent_naked_coeff * (-naked_token_logprob) +
        (1 - config.token_cross_ent_naked_coeff) * state.token_cross_ent_naked
    )
    token_cross_var_scaffold = (
        config.token_cross_var_naked_coeff * (token_cross_ent_naked - naked_token_logprob) ** 2 +
        (1 - config.token_cross_var_naked_coeff) * state.token_cross_var_naked
    )
    token_cross_var_naked = (
        config.token_cross_var_scaffold_coeff * (token_cross_ent_scaffold - scaffold_token_logprob) ** 2 +
        (1 - config.token_cross_var_scaffold_coeff) * state.token_cross_var_scaffold
    )
    return (
        token_cross_ent_scaffold,
        token_cross_ent_naked,
        token_cross_var_scaffold,
        token_cross_var_naked
    )

@partial(jax.jit, static_argnames=('config',))
def compute_outlier_threshold(state_ent, state_std, naked_ent, naked_varent, config):
    return (
        jnp.einsum('bi,ij,bj->b', state_ent, config.outlier_threshold.bilinear, state_std) +
        jnp.einsum('bi,i->b', state_ent, config.outlier_threshold.linear_state_ent) +
        jnp.einsum('bi,i->b', state_std, config.outlier_threshold.linear_state_std) +
        naked_ent * config.outlier_threshold.linear_naked_ent +
        naked_varent * config.outlier_threshold.linear_naked_varent +
        config.outlier_threshold.bias
    )

@partial(jax.jit, static_argnames=('config',))
def update_dirichlet_params(dir_support_logp, state, config):
    kl = kl_divergence(dir_support_logp, state.emwa_logp_dir_supp)
    emwa_logp_coeff = (config.emwa_logp_base ** (-config.emwa_logp_exp_factor / (kl + EPS)))[:, None]
    new_emwa_logp_dir_sup = emwa_logp_coeff * dir_support_logp + (1 - emwa_logp_coeff) * state.emwa_logp_dir_supp
    new_dir_params, _, _ = fit_dirichlet(new_emwa_logp_dir_sup)
    return new_dir_params, new_emwa_logp_dir_sup
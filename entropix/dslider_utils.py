from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


@jax.jit
def sample_dirichlet(key: jax.random.PRNGKey, alpha: jnp.ndarray) -> jnp.ndarray:
  """Sample from a Dirichlet distribution."""
  gamma_samples = jax.random.gamma(key, alpha, shape=alpha.shape)
  return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)


@jax.jit
def dirichlet_log_likelihood_from_logprob(
  logprobs: jnp.ndarray, alpha: jnp.ndarray
) -> jnp.ndarray:
  """
  Computes Dirichlet log likelihood:

  log Dir(p|α) = ln Γ(α₀) - ∑ᵢln Γ(αᵢ) + ∑ᵢ(αᵢ-1)ln(pᵢ)

  where:
  - α₀ = ∑ᵢαᵢ is the sum of all parameters
  - Γ(x) is the gamma function
  - pᵢ are probabilities (passed as logprobs)
  """
  return (
    jnp.sum((alpha - 1.0) * logprobs, axis=-1)
    - jsp.gammaln(jnp.sum(alpha, axis=-1))
    + jnp.sum(jsp.gammaln(alpha), axis=-1)
  )


@jax.jit
def dirichlet_expectation(alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the expectation of p ~ Dir(α):

  E[p] = αᵢ/∑ⱼαⱼ

  where:
  - αᵢ is the i-th parameter
  - ∑ⱼαⱼ is the sum of all parameters
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)
  return alpha / alpha_sum


@jax.jit
def dirichlet_expected_entropy(alpha: jnp.ndarray) -> jnp.ndarray:
  """
  Computes the expected entropy of p ~ Dir(α):

  E[H(p)] = ln B(α) + (α₀ - K)ψ(α₀) - ∑ⱼ(αⱼ - 1)ψ(αⱼ)

  where:
  - B(α) is the multivariate beta function
  - K is the dimension
  - ψ(x) is the digamma function
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # alpha_0
  K = alpha.shape[-1]
  # ln B(α) term
  log_beta = jnp.sum(jsp.gammaln(alpha), axis=-1) - jsp.gammaln(alpha_sum.squeeze())

  # (α₀ - K)ψ(α₀) term
  digamma_sum = jsp.digamma(alpha_sum)
  second_term = (alpha_sum.squeeze() - K) * digamma_sum.squeeze()

  # -sum((αⱼ - 1)ψ(αⱼ)) term
  digamma_alpha = jsp.digamma(alpha)
  third_term = -jnp.sum((alpha - 1) * digamma_alpha, axis=-1)

  return log_beta + second_term + third_term


@jax.jit
def dirichlet_expected_varentropy(alpha: jnp.ndarray) -> jnp.ndarray:
  """Compute the expected varentropy of p ~ Dir(α):

  E[∑ᵢ ln(pᵢ)² * pᵢ] = ∑ᵢ (αᵢ/α₀) * (ψ(αᵢ)² + ψ₁(αᵢ))

  where:
  - α₀ = ∑ᵢαᵢ is the sum of all parameters
  - ψ(x) is the digamma function
  - ψ₁(x) is the trigamma function (first derivative of digamma)
  """
  alpha_sum = jnp.sum(alpha, axis=-1, keepdims=True)  # α₀
  # E[Xᵢ] = αᵢ/α₀
  expected_x = alpha / alpha_sum
  # ψ(αᵢ)² + ψ₁(αᵢ) term
  digamma_alpha = jsp.digamma(alpha)
  trigamma_alpha = jsp.polygamma(1, alpha)
  squared_plus_deriv = digamma_alpha**2 + trigamma_alpha
  # ∑ᵢ (αᵢ/α₀) * (ψ₁(αᵢ) + ψ(αᵢ)²)
  return jnp.sum(expected_x * squared_plus_deriv, axis=-1)


@jax.jit
def halley_update(alpha, target_values):
  """
  Compute the Halley's method update direction for the function
  """
  p1 = jsp.polygamma(1, alpha)
  p2 = jsp.polygamma(2, alpha)
  S = jnp.sum(alpha, axis=-1, keepdims=True)
  s1 = jsp.polygamma(1, S)
  s2 = jsp.polygamma(2, S)
  p1_inv = 1.0 / p1
  sum_p1_inv = jnp.sum(p1_inv, axis=-1, keepdims=True)
  denom = 1.0 - s1 * sum_p1_inv
  denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
  coeff = s1 / denom
  error = jsp.digamma(alpha) - jsp.digamma(S) - target_values
  temp = p1_inv * error
  sum_temp = jnp.sum(temp, axis=-1, keepdims=True)
  J_inv_error = temp + coeff * sum_temp * p1_inv
  sum_J_inv_error = jnp.sum(J_inv_error, axis=-1, keepdims=True)
  H_J_inv_error = p2 * J_inv_error - s2 * sum_J_inv_error
  temp2 = p1_inv * H_J_inv_error
  sum_temp2 = jnp.sum(temp2, axis=-1, keepdims=True)
  J_inv_H_J_inv_error = temp2 + coeff * sum_temp2 * p1_inv
  return -J_inv_error + 0.5 * J_inv_H_J_inv_error


@partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9))
def fit_dirichlet(
  target_values,
  init_alpha=None,
  initial_lr=1.2,
  decay_alpha=0.1,
  decay_beta=2.0,
  decay_gamma=0.25,
  decay_nu=0.75,
  max_iters=140,
  tol=1e-4,
  dtype: jnp.dtype = jnp.bfloat16,
):
  """
  Estimates Dirichlet parameters (alpha) from target logprobs.
  """
  batch_shape = target_values.shape[:-1]
  n = target_values.shape[-1]
  min_lr = 1e-8
  target_values = target_values.astype(
    jnp.float32
  )  # for large vocab size needs float64
  if init_alpha is None:
    init_alpha = jnp.ones((*batch_shape, n), dtype=jnp.float32)

  def scan_body(carry, _):
    alpha, converged, error_norm, step = carry
    S = jnp.sum(alpha, axis=-1, keepdims=True)
    digamma_alpha = jsp.digamma(alpha)
    psi_S = jsp.digamma(S)
    error = digamma_alpha - psi_S - target_values
    error_norm = jnp.linalg.norm(error, axis=-1)
    new_converged = converged | (error_norm < tol)
    exp_factor = jnp.exp(-decay_alpha * (step**decay_nu))
    cos_factor = jnp.abs(jnp.cos(decay_beta / (step**decay_gamma)))
    lr = initial_lr * exp_factor * cos_factor
    lr = jnp.maximum(lr, min_lr)
    delta_alpha = halley_update(alpha, target_values)
    scaled_delta_alpha = lr[..., None] * delta_alpha
    max_delta = 0.5 * alpha
    scaled_delta_alpha = jnp.clip(scaled_delta_alpha, -max_delta, max_delta)
    new_alpha = jnp.where(
      new_converged[..., None],
      alpha,
      jnp.maximum(alpha + scaled_delta_alpha, alpha / 2),
    )
    return (new_alpha, new_converged, error_norm, step + 1), None

  init_state = (
    init_alpha,
    jnp.zeros(batch_shape, dtype=jnp.bool_),
    jnp.full(batch_shape, jnp.inf),
    jnp.ones(batch_shape, dtype=jnp.int32),
  )
  (final_alpha, final_converged, _, final_step), _ = jax.lax.scan(
    scan_body, init_state, None, length=max_iters
  )

  return final_alpha.astype(dtype), final_step - 1, final_converged


@jax.jit
def ent_grad_hess(
  logits: jnp.ndarray, T: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  p = jax.nn.softmax(logits / T[..., None], axis=-1)
  log_p = jax.nn.log_softmax(logits / T[..., None], axis=-1)
  mu1 = jnp.sum(p * log_p, axis=-1)
  diff = log_p - mu1[..., None]
  mu2 = jnp.sum(p * diff**2, axis=-1)
  mu3 = jnp.sum(p * diff**3, axis=-1)
  return -mu1, mu2 / T, -(2 * mu3 + 3 * mu2) / (T * T)


@partial(jax.jit, static_argnums=(3, 4, 5, 6))
def temp_tune(
  logits: jnp.ndarray,
  target_ent: jnp.ndarray,
  T_init: float = 1.0,
  lr: float = 0.1,
  max_iters: int = 10,
  tol: float = 1e-6,
  dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  batch_size = logits.shape[0]
  logits = logits.astype(jnp.float32)

  def scan_body(carry, _):
    T, iters, converged = carry
    ent, grad, hess = ent_grad_hess(logits, T)
    error = ent - target_ent
    new_converged = converged | (jnp.abs(error) < tol)
    denominator = 2 * grad * grad - error * hess
    halley_step = jnp.where(
      jnp.abs(denominator) > 1e-8,
      2 * error * grad / denominator,
      jnp.full_like(T, jnp.inf),
    )
    newton_step = jnp.where(
      jnp.abs(grad) > 1e-8, error / grad, jnp.full_like(T, jnp.inf)
    )
    grad_step = jnp.where(error > 0, lr * T, -lr * T)

    delta_T = jnp.where(
      jnp.abs(grad) < 1e-8,
      grad_step,
      jnp.where(jnp.abs(denominator) < 1e-8, newton_step, halley_step),
    )
    delta_T = jnp.clip(delta_T, -0.5 * T, 0.5 * T)
    new_T = jnp.where(new_converged, T, jnp.maximum(T - delta_T, T / 2))
    return (new_T, iters + 1, new_converged), None

  init_state = (
    jnp.full((batch_size,), T_init, dtype=jnp.float32),
    jnp.zeros(batch_size, dtype=jnp.int32),
    jnp.zeros(batch_size, dtype=jnp.bool_),
  )
  (final_T, final_iters, final_converged), _ = jax.lax.scan(
    scan_body, init_state, None, length=max_iters
  )
  return final_T.astype(dtype), final_iters, final_converged

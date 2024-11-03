from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp


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

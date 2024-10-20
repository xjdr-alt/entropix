from typing import Dict, Tuple

import jax
import jax.numpy as jnp


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy

@jax.jit
def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = jax.random.exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)

@jax.jit
def adaptive_sample(logits: jax.Array, *, temperature: float | jax.Array, key=jax.random.PRNGKey(1337), epsilon: float = 0.01) -> jax.Array:
    """
    Perform adaptive sampling by dynamically adjusting the candidate set size based on entropy and varentropy.
    """
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    # Sort tokens by probability
    sorted_probs, sorted_indices = jax.lax.top_k(probs, k=probs.shape[-1])

    # Initialize candidate set size
    candidate_mask = jnp.zeros_like(sorted_probs, dtype=bool)  # To track which tokens are in the candidate set
    cumulative_entropy = jnp.zeros((bsz,))
    cumulative_varentropy = jnp.zeros((bsz,))
    previous_entropy = -jnp.sum(sorted_probs[0] * jnp.log2(jnp.clip(sorted_probs[0], 1e-10, 1.0)))

    def cond_fn(state):
        # Continue if entropy reduction is greater than epsilon
        cumulative_entropy, cumulative_varentropy, i, mask = state
        entropy_reduction = cumulative_entropy - previous_entropy
        return (entropy_reduction >= epsilon) & (i < sorted_probs.shape[-1])

    def body_fn(state):
        cumulative_entropy, cumulative_varentropy, i, mask = state
        current_prob = sorted_probs[:, i]

        # Update entropy and varentropy with current token
        current_entropy = -jnp.sum(current_prob * jnp.log2(jnp.clip(current_prob, 1e-10, 1.0)))
        current_varentropy = jnp.sum(current_prob * (jnp.log2(jnp.clip(current_prob, 1e-10, 1.0)) + cumulative_entropy[:, None])**2)

        entropy_reduction = cumulative_entropy - current_entropy
        varentropy_reduction = cumulative_varentropy - current_varentropy

        # Update cumulative entropy, varentropy, and mask if reductions are sufficient
        mask = jnp.where(entropy_reduction >= epsilon, mask.at[:, i].set(True), mask)

        cumulative_entropy = cumulative_entropy.at[:, i].set(current_entropy)
        cumulative_varentropy = cumulative_varentropy.at[:, i].set(current_varentropy)

        return cumulative_entropy, cumulative_varentropy, i + 1, mask

    initial_state = (cumulative_entropy, cumulative_varentropy, 0, candidate_mask)
    final_state = jax.lax.while_loop(cond_fn, body_fn, initial_state)

    # Mask out tokens not in the candidate set
    final_mask = final_state[-1]
    candidate_probs = sorted_probs * final_mask
    candidate_probs = candidate_probs / jnp.sum(candidate_probs, axis=-1, keepdims=True)

    # Sample from the final candidate set
    next_token = multinomial_sample_one(candidate_probs, key)
    next_token_g = jnp.take_along_axis(sorted_indices, next_token.reshape(bsz, 1), axis=-1)

    return next_token_g.astype(jnp.int32)

@jax.jit
def calculate_metrics(logits: jnp.ndarray, attention_scores: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = jax.nn.softmax(attention_scores, axis=-1)
    attn_entropy = -jnp.sum(attention_probs * jnp.log2(jnp.clip(attention_probs, 1e-10, 1.0)), axis=-1)
    attn_varentropy = jnp.var(attn_entropy, axis=1)

    mean_attention = jnp.mean(attention_probs, axis=1)
    agreement = jnp.mean(jnp.abs(attention_probs - mean_attention[:, None, :]), axis=(1, 2))

    interaction_strength = jnp.mean(jnp.abs(attention_scores), axis=(1, 2, 3))

    return {
        "logits_entropy": jnp.mean(entropy),
        "logits_varentropy": jnp.mean(varentropy),
        "attn_entropy": jnp.mean(attn_entropy),
        "attn_varentropy": jnp.mean(attn_varentropy),
        "agreement": jnp.mean(agreement),
        "interaction_strength": interaction_strength
    }

@jax.jit
def new_sample(logits: jax.Array, attention_scores: jax.Array, cur_pos: int, key=jax.random.PRNGKey(1337)) -> jax.Array:

    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    return adaptive_sample(
            logits,
            temperature=0.666,
            key=key,
            epsilon=0.1  # Confidence threshold for adaptive decoding
        )

def multinomial_sample_one(probs_sort: jax.Array, key=jax.random.PRNGKey(1337)) -> jax.Array:
  """Samples one token from a multinomial distribution with sorted probabilities."""
  q = jax.random.exponential(key=key, shape=probs_sort.shape)
  return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)


# TODO: change this to support batch settings for temperature, top_p, and top_k
# def sample(tokens: jax.Array, logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, key = jax.random.PRNGKey(1337)) -> jax.Array:
def sample(logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, key=jax.random.PRNGKey(1337)) -> jax.Array:
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

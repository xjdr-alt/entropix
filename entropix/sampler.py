from typing import Dict, Tuple

import jax
import jax.numpy as jnp


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """
    Sample from a multinomial distribution using the Gumbel-max trick.
    This method provides better numerical stability than naive multinomial sampling.
    
    Args:
        probs_sort: Sorted probability distribution array
        key: JAX PRNG key for random number generation
        
    Returns:
        Array of sampled indices with shape [..., 1]
    """
    q = jax.random.exponential(key=key, shape=probs_sort.shape).astype(jnp.bfloat16)
    result = jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)
    return result

@jax.jit
def get_window_probs(sorted_probs: jax.Array, i: jax.Array) -> jax.Array:
    """
    Extract probabilities up to index i from sorted probability distribution.
    Uses efficient masking for TPU optimization.
    
    Args:
        sorted_probs: Sorted probability array of shape [batch_size, vocab_size]
        i: Index up to which probabilities should be included
        
    Returns:
        Masked probability array where values beyond index i are set to zero
    """
    vocab_size = sorted_probs.shape[1]
    indices = jnp.arange(vocab_size, dtype=jnp.int32)
    mask = indices <= i
    mask = jnp.broadcast_to(mask, sorted_probs.shape)
    return jnp.where(mask, sorted_probs, jnp.zeros_like(sorted_probs))

@jax.jit
def adaptive_sample(logits: jax.Array, *, temperature: float | jax.Array = 0.666, key=jax.random.PRNGKey(1337), epsilon: float = 0.01) -> jax.Array:
    """
    Perform entropy-based adaptive sampling from a probability distribution.
    
    This implementation dynamically determines the set of candidate tokens by measuring
    how each additional token affects the entropy of the sampling distribution. It stops
    adding tokens when their contribution to the entropy falls below a threshold,
    providing an adaptive alternative to fixed top-k or top-p sampling.
    
    Args:
        logits: Raw model logits of shape [batch_size, vocab_size]
        temperature: Softmax temperature to control distribution sharpness (default: 0.666)
        key: JAX PRNG key for sampling (default: fixed seed 1337)
        epsilon: Minimum required entropy gain to include additional tokens (default: 0.01)
        
    Returns:
        Selected token indices of shape [batch_size, 1]
    
    Algorithm:
    1. Convert logits to probabilities using temperature-scaled softmax
    2. Sort tokens by probability in descending order
    3. Iteratively build candidate set:
       - Start with highest probability token
       - Add next token if it increases distribution entropy by >= epsilon
       - Continue until entropy gain falls below epsilon or all tokens processed
    4. Sample from the final candidate distribution
    
    Example:
    For a probability distribution [0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.01]:
    - First token contribution: -0.5 * log2(0.5) ≈ 0.5 bits
    - Second token adds: -0.3 * log2(0.3) ≈ 0.38 bits
    - Later tokens add progressively less entropy
    - Stop when next token would add < epsilon bits
    
    This approach naturally adapts the sampling pool size based on the shape of
    the probability distribution, avoiding the need for hand-tuned cutoffs.
    """
    bsz = logits.shape[0]
    
    # Cast temperature to bfloat16 and ensure it's an array
    temperature = jnp.array(temperature, dtype=jnp.bfloat16)
    
    # Compute softmax with improved numerical stability
    logits = logits.astype(jnp.bfloat16)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits / temperature)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)
    
    # Use top_k for sorting - very efficient on TPU
    sorted_probs, sorted_indices = jax.lax.top_k(probs, k=probs.shape[-1])
    
    def cond_fn(state):
        current_entropy, previous_entropy, i, mask = state
        entropy_gain = current_entropy - previous_entropy
        return (jnp.any(entropy_gain >= epsilon)) & (i < sorted_probs.shape[-1])
    
    def body_fn(state):
        current_entropy, previous_entropy, i, mask = state
        
        # Get probabilities up to current index
        current_probs = get_window_probs(sorted_probs, i)
        
        # Normalize probabilities
        normalizing_factor = jnp.sum(current_probs, axis=-1, keepdims=True)
        normalized_probs = jnp.where(
            normalizing_factor > 0,
            current_probs / (normalizing_factor + jnp.bfloat16(1e-6)),
            jnp.zeros_like(current_probs)
        )
        
        # Calculate entropy
        log_probs = jnp.log2(jnp.maximum(normalized_probs, jnp.bfloat16(1e-6)))
        new_entropy = -jnp.sum(
            jnp.where(normalized_probs > 0, 
                     normalized_probs * log_probs,
                     jnp.zeros_like(normalized_probs)),
            axis=-1
        )
        
        # Update mask
        entropy_gain = new_entropy - current_entropy
        new_mask = mask.at[:, i].set(entropy_gain >= epsilon)
        
        return (new_entropy, current_entropy, i + 1, new_mask)
    
    # Initialize state
    initial_entropy = jnp.zeros((bsz,), dtype=jnp.bfloat16)
    initial_mask = jnp.zeros((bsz, sorted_probs.shape[-1]), dtype=bool)
    initial_state = (
        initial_entropy,
        initial_entropy,
        jnp.array(0, dtype=jnp.int32),
        initial_mask
    )
    
    # Run the while loop
    final_entropy, _, _, final_mask = jax.lax.while_loop(
        cond_fn, body_fn, initial_state
    )
    
    # Create final candidate distribution
    candidate_probs = jnp.where(final_mask, sorted_probs, jnp.zeros_like(sorted_probs))
    normalizing_factor = jnp.sum(candidate_probs, axis=-1, keepdims=True)
    candidate_probs = candidate_probs / (normalizing_factor + jnp.bfloat16(1e-6))
    
    # Sample and map back to original indices
    next_token = multinomial_sample_one(candidate_probs, key)
    next_token_global = jnp.take_along_axis(
        sorted_indices, 
        next_token.reshape(bsz, 1), 
        axis=-1
    )
    
    return next_token_global.astype(jnp.int32)

    """
    # First attempt - we got so far but it still wasn't enough...
    #

    # Get probabilities for the last position in the sequence
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)
    
    # Sort tokens by probability (descending order)
    sorted_probs, sorted_indices = jax.lax.top_k(probs, k=probs.shape[-1])
    
    # Initialize state for the while loop
    candidate_mask = jnp.zeros_like(sorted_probs, dtype=bool)
    
    # Initialize with empty distribution (zero entropy)
    initial_entropy = jnp.zeros((bsz,))
    
    def cond_fn(state):
        # Continue if we haven't reached vocab size and last entropy gain was sufficient
        current_entropy, previous_entropy, i, mask = state
        entropy_gain = current_entropy - previous_entropy
        return (jnp.any(entropy_gain >= epsilon)) & (i < sorted_probs.shape[-1])
    
    def body_fn(state):
        current_entropy, previous_entropy, i, mask = state
        
        # Convert i to int32 in a way that JAX can handle during tracing
        i_int32 = jax.lax.convert_element_type(i, jnp.int32)
        
        # Get probabilities up to and including position i+1
        current_probs = jax.lax.dynamic_slice_in_dim(
            sorted_probs,
            start_index=0,
            slice_size=i_int32 + 1,
            axis=1
        )
        
        # Rest of the function remains the same...
        normalizing_factor = jnp.sum(current_probs, axis=-1, keepdims=True)
        normalized_probs = current_probs / normalizing_factor
        
        new_entropy = -jnp.sum(
            normalized_probs * jnp.log2(jnp.clip(normalized_probs, 1e-10, 1.0)),
            axis=-1
        )
        
        entropy_gain = new_entropy - current_entropy
        mask = mask.at[:, i_int32].set(entropy_gain >= epsilon)
        
        return new_entropy, current_entropy, i + 1, mask
    
    # Run the while loop to build candidate set
    initial_state = (initial_entropy, initial_entropy, 0, candidate_mask)
    final_entropy, _, _, final_mask = jax.lax.while_loop(
        cond_fn, body_fn, initial_state
    )
    
    # Create the final candidate distribution
    # Zero out probabilities of tokens not in candidate set
    candidate_probs = sorted_probs * final_mask
    
    # Renormalize the candidate probabilities
    candidate_probs = candidate_probs / jnp.sum(
        candidate_probs, axis=-1, keepdims=True
    )
    
    # Sample from the final candidate set
    next_token = multinomial_sample_one(candidate_probs, key)
    
    # Map back to original token indices
    next_token_global = jnp.take_along_axis(
        sorted_indices, 
        next_token.reshape(bsz, 1), 
        axis=-1
    )
    
    return next_token_global.astype(jnp.int32)
    """
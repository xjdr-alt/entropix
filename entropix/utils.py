import jax
import jax.numpy as jnp
import math
import jax.random as random

def apply_scaling(freqs: jax.Array):
    SCALE_FACTOR = 8
    LOW_FREQ_FACTOR = 1
    HIGH_FREQ_FACTOR = 4
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq):
        wavelen = 2 * math.pi / freq

        def scale_mid(_):
            smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
            return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        return jax.lax.cond(
            wavelen < high_freq_wavelen,
            lambda _: freq,
            lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
            None
        )

    return jax.vmap(scale_freq)(freqs)

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)
    t = jnp.arange(end, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask

def sample(logits: jax.Array, temperature=0.8, top_p=0.95, top_k=40, key=None) -> jax.Array:
    if key is None:
        key = random.PRNGKey(0)
    
    bsz = logits.shape[0]
    logit = logits[:, -1]
    
    # Apply temperature
    logit = logit / temperature
    
    # Apply top-k sampling
    top_k_logits, top_k_indices = jax.lax.top_k(logit, k=min(top_k, logit.shape[-1]))
    
    # Apply top-p sampling
    sorted_logits = jnp.sort(top_k_logits, axis=-1)[:, ::-1]
    cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
    sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
    indices_to_remove = jnp.take_along_axis(sorted_indices_to_remove, jnp.argsort(-top_k_logits, axis=-1), axis=-1)
    top_k_logits = jnp.where(indices_to_remove, -jnp.inf, top_k_logits)
    
    # Sample from the filtered distribution
    probs = jax.nn.softmax(top_k_logits, axis=-1)
    next_token = random.categorical(key, probs, axis=-1)
    next_token = jnp.take_along_axis(top_k_indices, next_token[:, None], axis=-1)
    
    return next_token.astype(jnp.int32)

def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = jax.random.exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)

# Add any other common functions here
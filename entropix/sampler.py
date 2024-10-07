import jax
import jax.numpy as jnp
from typing import Tuple, Dict

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

@jax.jit
def calculate_varentropy_logsoftmax(logits: jnp.ndarray, axis: int = -1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = jax.nn.log_softmax(logits, axis=axis)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=axis) / LN_2  # Convert to base-2
    varentropy = jnp.sum(probs * (log_probs / LN_2 + entropy[..., None])**2, axis=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: jax.Array, key) -> jax.Array:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = jax.random.exponential(key=key, shape=probs_sort.shape)
    return jnp.argmax(probs_sort / q, axis=-1, keepdims=True).astype(jnp.int32)

def _sample(logits: jax.Array, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, key=jax.random.PRNGKey(1337)) -> jax.Array:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = jax.nn.softmax(logit / temperature, axis=-1)

    # Apply min_p sampling
    if min_p > 0.0:
      p_max = jnp.max(probs, axis=-1, keepdims=True)
      indices_to_remove = probs < (min_p * p_max)
      logit = jnp.where(indices_to_remove, jnp.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = jax.lax.top_k(probs, k=top_k)
    probs_sort = jnp.flip(top_k_probs, axis=-1)
    probs_idx = jnp.flip(top_k_indices, axis=-1)
    probs_sum = jnp.cumsum(probs_sort, axis=-1)
    # Apply top-p sampling
    mask = jnp.where(probs_sum - probs_sort > top_p, 1.0, 0.0)
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / jnp.sum(probs_sort, axis=-1, keepdims=True)
    next_token = multinomial_sample_one(probs_sort, key)
    next_token_g = jnp.take_along_axis(probs_idx, next_token.reshape(bsz, 1), axis=-1)
    return next_token_g.astype(jnp.int32)

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

def adaptive_sample(logits: jax.Array, metrics: Dict[str, jnp.ndarray],
                    gen_tokens: jax.Array, n_samples: int,
                    base_temp: float = 0.666, base_top_p: float = 0.90, base_top_k: int = 40, base_min_p: float = 0.03, # Turn this down to 0.01 to reduce the shoggoth
                    key: jax.random.PRNGKey = jax.random.PRNGKey(1337)) -> jax.Array:
    logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
    attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

    temperature = base_temp * (1 + 0.3 * logits_uncertainty + 0.2 * attn_uncertainty - 0.2 * metrics["agreement"])
    top_p = jnp.clip(base_top_p * (1 + 0.1 * metrics["attn_varentropy"]), 0.1, 1.0)
    top_k = int(jnp.clip(
        jnp.round(base_top_k * (1 + 0.3 * metrics["interaction_strength"].item() - 0.2 * metrics["agreement"].item())),
        a_min=1,
        a_max=100
    ))
    min_p = jnp.clip(base_min_p * (1 - 0.5 * logits_uncertainty), 0.01, 0.5)

    keys = jax.random.split(key, n_samples)

    samples = []
    for sample_key in keys:
        sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, key=sample_key)
        samples.append(sample)

    def score_sample(sample):
        log_prob = jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(sample, logits.shape[-1]))
        confidence_score = (
            (1 - metrics["logits_entropy"]) * 0.1 +
            (1 - metrics["attn_entropy"]) * 0.2 +
            (1 - metrics["logits_varentropy"]) * 0.3 +
            (1 - metrics["attn_varentropy"]) * 0.4 +
            metrics["agreement"] * 0.5 +
            metrics["interaction_strength"] * 0.6
        )
        return log_prob + confidence_score

    sample_scores = [score_sample(sample) for sample in samples]
    best_sample_idx = jnp.argmax(jnp.array(sample_scores))
    return samples[best_sample_idx]

# I am absolutely appaled that these random hyperparams are virtually impossible to beat with a more sophisticated approach.
# We are leaving it this way for now, but we should definitely be much better than this. Have some self respect.
def sample(gen_tokens: jax.Array, logits: jax.Array, attention_scores: jax.Array,
           temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, key=jax.random.PRNGKey(1337)) -> jax.Array:
    metrics = calculate_metrics(logits, attention_scores)
    #print(f'{metrics=}')
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < 0.1 and vent < 0.1:
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > 3.0 and vent < 0.1:
        # Insert a clarifying question token if not already present
        if not jnp.isin(gen_tokens[:,-1], 2564).any():
            return jnp.array([[2564]])  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = 1.3 + 0.2 * attn_ent  # Increase temperature based on attention entropy
            return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k, min_p=min_p, key=key)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < 5.0 and vent > 5.0:
        temp_adj = 1.2 + 0.3 * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, temperature * temp_adj), top_p=top_p, top_k=top_k_adj, min_p=min_p, key=key)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > 5.0 and vent > 5.0:
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = 2.0 + 0.5 * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, top_p - 0.2 * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, temperature * temp_adj), top_p=top_p_adj, top_k=top_k, min_p=min_p, key=key)

    # Middle ground: use adaptive sampling
    else:
        # Interpolate temperature based on entropy and varentropy
        #t = jnp.clip((ent + vent) / 10.0, 0.5, 2.0)
        # Adjust temperature and top_k based on attention metrics
        #temp_adj = t + 0.2 * attn_ent + 0.1 * attn_vent
        #top_k_adj = max(5, int(top_k * (1 + 0.3 * interaction_strength - 0.2 * agreement)))
        #return _sample(logits, temperature=temp_adj * temperature, top_p=top_p, top_k=top_k_adj, min_p=min_p, key=key)
        # Adaptive sample is still crazy pants. Leave the more stable code above here for now.
        return adaptive_sample(
            logits,
            metrics,
            gen_tokens,
            n_samples=5,
            base_temp=temperature,
            base_top_p=top_p,
            base_top_k=top_k,
            key=key
        )
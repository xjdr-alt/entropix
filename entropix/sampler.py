from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

class SamplerParams(NamedTuple):
    steer_tokens: jax.Array
    temp: float
    top_p: float
    min_p: float
    top_k: int
    

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

def _sample( logits: jax.Array, *, temperature: float | jax.Array, top_p: float | jax.Array, top_k: int | jax.Array, min_p: float | jax.Array,
            key=jax.random.PRNGKey(1337),) -> jax.Array:
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

def calculate_metrics(logits: jnp.ndarray, attn_scores: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = jax.nn.softmax(attn_scores, axis=-1)
    attn_entropy = -jnp.sum(attention_probs * jnp.log2(jnp.clip(attention_probs, 1e-10, 1.0)), axis=-1)
    attn_varentropy = jnp.var(attn_entropy, axis=1)

    mean_attention = jnp.mean(attention_probs, axis=1)
    agreement = jnp.mean(jnp.abs(attention_probs - mean_attention[:, None, :]), axis=(1, 2))

    interaction_strength = jnp.mean(jnp.abs(attn_scores), axis=(1, 2, 3))

    return {
        "logits_entropy": jnp.mean(entropy),
        "logits_varentropy": jnp.mean(varentropy),
        "attn_entropy": jnp.mean(attn_entropy),
        "attn_varentropy": jnp.mean(attn_varentropy),
        "agreement": jnp.mean(agreement),
        "interaction_strength": interaction_strength
    }

@chex.dataclass(kw_only=True, frozen=True)
class SamplerConfig:
    """
    Encapsulation of all available sampler hyperparameters.

    This should be a good starting point for baselining experiments.
    """

    temp: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03  # Turn this down to 0.01 to reduce the shoggoth

    low_ent_thresh: float = 0.1
    low_vent_thresh: float = 0.1
    med_ent_thresh: float = 3.0
    high_ent_thresh: float = 5.0
    high_vent_thresh: float = 5.0

    # TODO this is a bit of a nasty mess, but also makes all the hyperparameters visible
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    # TODO not convinced this should
    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5
    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6


def sample(gen_tokens: jax.Array, logits: jax.Array, attention_scores: jax.Array, cfg: SamplerConfig,
           clarifying_question_token: int = 2564, key=jax.random.PRNGKey(1337)) -> jax.Array:

    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if ent < cfg.low_ent_thresh and vent < cfg.low_vent_thresh:
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif ent > cfg.high_ent_thresh and vent < cfg.low_vent_thresh:
        # Insert a clarifying question token if not already present
        if not jnp.isin(gen_tokens[:,-1], clarifying_question_token).any():
            return jnp.array([[clarifying_question_token]])
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent  # Increase temperature based on attention entropy
            return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=cfg.top_k, min_p=cfg.min_p, key=key)

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif ent < cfg.high_ent_thresh and vent > cfg.high_vent_thresh:
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(logits, temperature=min(1.5, cfg.temp * temp_adj), top_p=cfg.top_p, top_k=top_k_adj, min_p=cfg.min_p, key=key)

    # High Entropy, High Varentropy: "resampling in the mist"
    elif ent > cfg.med_ent_thresh and vent > cfg.high_vent_thresh:
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.hehv_attn_ent_coef * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(logits, temperature=max(2.0, cfg.temp * temp_adj), top_p=top_p_adj, top_k=cfg.top_k, min_p=cfg.min_p, key=key)

    # Middle ground: use adaptive sampling
    else:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temp * (1 + cfg.ada_temp_logits * logits_uncertainty + cfg.ada_temp_attn * attn_uncertainty - cfg.ada_temp_agree * metrics["agreement"])
        top_p = jnp.clip(cfg.top_p * (1 + cfg.ada_top_p * metrics["attn_varentropy"]), 0.1, 1.0)
        top_k = int(jnp.clip(
            jnp.round(cfg.top_k * (1 + cfg.ada_top_k_int * metrics["interaction_strength"].item() - cfg.ada_top_k_agree * metrics["agreement"].item())),
            a_min=1,
            a_max=100
        ))
        min_p = jnp.clip(cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty), 0.01, 0.5)

        keys = jax.random.split(key, cfg.n_adaptive_samples)

        samples = []
        for sample_key in keys:
            sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, key=sample_key)
            samples.append(sample)

        def score_sample(sample):
            log_prob = jnp.sum(jax.nn.log_softmax(logits) * jax.nn.one_hot(sample, logits.shape[-1]))
            confidence_score = (
                (1 - metrics["logits_entropy"]) * cfg.ada_score_logits_ent +
                (1 - metrics["attn_entropy"]) * cfg.ada_score_attn_ent +
                (1 - metrics["logits_varentropy"]) * cfg.ada_score_logits_vent +
                (1 - metrics["attn_varentropy"]) * cfg.ada_score_attn_vent +
                metrics["agreement"] * cfg.ada_score_agree +
                metrics["interaction_strength"] * cfg.ada_score_int
            )
            return log_prob + confidence_score

        sample_scores = [score_sample(sample) for sample in samples]
        best_sample_idx = jnp.argmax(jnp.array(sample_scores))
        return samples[best_sample_idx]

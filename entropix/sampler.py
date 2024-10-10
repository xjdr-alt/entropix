from typing import Dict, Tuple

import math
import chex
import jax
import jax.numpy as jnp

LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E

# ANSI escape sequences for text formatting
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_ITALIC = "\033[3m"
ANSI_DIM = "\033[2m"

# Catppuccin color palette
COLORS = {
    "rosewater": (244, 194, 193),
    "flamingo": (242, 150, 160),
    "pink": (245, 140, 173),
    "mauve": (203, 166, 247),
    "red": (243, 139, 168),
    "maroon": (235, 160, 172),
    "peach": (250, 179, 135),
    "yellow": (249, 226, 175),
    "green": (166, 227, 161),
    "teal": (148, 226, 213),
    "sky": (137, 220, 235),
    "sapphire": (116, 199, 236),
    "blue": (137, 180, 250),
    "lavender": (180, 190, 254),
    "text": (205, 214, 244),
    "subtext1": (186, 194, 222),
    "subtext0": (166, 173, 200),
    "overlay2": (147, 153, 178),
    "overlay1": (127, 132, 156),
    "overlay0": (108, 112, 134),
    "surface2": (88, 91, 112),
    "surface1": (69, 71, 90),
    "surface0": (49, 50, 68),
    "base": (30, 30, 46),
    "mantle": (24, 24, 37),
    "crust": (17, 17, 27)
}

def blend_colors(color1: Tuple[int, int, int], color2: Tuple[int, int, int], weight: float = 0.5) -> Tuple[int, int, int]:
    # Use a power function to emphasize brighter colors
    emphasis = 2.0
    w1 = weight ** (1/emphasis)
    w2 = (1 - weight) ** (1/emphasis)

    blended = tuple(int(((c1/255)**emphasis * w1 + (c2/255)**emphasis * w2) ** (1/emphasis) * 255)
                    for c1, c2 in zip(color1, color2))

    # Ensure the result is within valid RGB range
    blended = tuple(max(0, min(255, c)) for c in blended)

    #print(f"Debug: Blend result: {blended} (color1: {color1}, color2: {color2}, weight: {weight})", file=sys.stderr, flush=True)
    return blended

def get_color_for_metric(metrics: Dict[str, float], config) -> Tuple[Tuple[int, int, int], str]:
    """Get color and formatting for metrics based on their values and thresholds."""
    ent = metrics["logits_entropy"]
    vent = metrics["logits_varentropy"]
    attn_ent = metrics["attn_entropy"]
    attn_vent = metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]

    color = COLORS["text"]  # Start with default text color
    formatting = ""

    # Logits Entropy
    if ent < config.low_logits_entropy_threshold:
        color = blend_colors(color, COLORS["blue"], 0.7)
    elif ent < config.medium_logits_entropy_threshold:
        color = blend_colors(color, COLORS["sky"], 0.7)
    elif ent < config.high_logits_entropy_threshold:
        color = blend_colors(color, COLORS["sapphire"], 0.7)
    else:
        color = blend_colors(color, COLORS["lavender"], 0.7)

    # Logits Varentropy
    if vent < config.low_logits_varentropy_threshold:
        color = blend_colors(color, COLORS["green"], 0.3)
    elif vent < config.medium_logits_varentropy_threshold:
        color = blend_colors(color, COLORS["teal"], 0.3)
    elif vent < config.high_logits_varentropy_threshold:
        color = blend_colors(color, COLORS["yellow"], 0.3)
    else:
        color = blend_colors(color, COLORS["peach"], 0.3)

    # Attention Entropy
    if attn_ent < config.low_attention_entropy_threshold:
        formatting += ANSI_BOLD
    elif attn_ent > config.high_attention_entropy_threshold:
        formatting += ANSI_ITALIC

    # Attention Varentropy
    if attn_vent < config.low_attention_varentropy_threshold:
        color = blend_colors(color, COLORS["rosewater"], 0.2)
    elif attn_vent < config.medium_attention_varentropy_threshold:
        color = blend_colors(color, COLORS["flamingo"], 0.2)
    elif attn_vent < config.high_attention_varentropy_threshold:
        color = blend_colors(color, COLORS["pink"], 0.2)
    else:
        color = blend_colors(color, COLORS["mauve"], 0.2)

    # Agreement
    if agreement < config.low_agreement_threshold:
        formatting += ANSI_DIM
    elif agreement > config.high_agreement_threshold:
        color = blend_colors(color, COLORS["red"], 0.2)

    # Interaction Strength
    if interaction_strength < config.low_interaction_strength_threshold:
        color = blend_colors(color, COLORS["surface2"], 0.1)
    elif interaction_strength < config.medium_interaction_strength_threshold:
        color = blend_colors(color, COLORS["surface1"], 0.1)
    elif interaction_strength < config.high_interaction_strength_threshold:
        color = blend_colors(color, COLORS["surface0"], 0.1)
    else:
        color = blend_colors(color, COLORS["base"], 0.1)

    return color, formatting



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

@chex.dataclass(kw_only=True, frozen=True)
class SamplerConfig:
    """
    Configuration for the sampling strategy, including threshold values for various metrics
    and adaptive sampling parameters.
    """

    # Sampling Hyperparameters
    temperature: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_probability: float = 0.03  # Minimum probability threshold for token selection

    # Logits Entropy Thresholds
    low_logits_entropy_threshold: float = 0.01
    medium_logits_entropy_threshold: float = 0.7
    high_logits_entropy_threshold: float = 2.1

    # Logits Varentropy Thresholds
    low_logits_varentropy_threshold: float = 0.05
    medium_logits_varentropy_threshold: float = 2.0
    high_logits_varentropy_threshold: float = 5.8

    # Attention Entropy Thresholds
    low_attention_entropy_threshold: float = 11.915
    medium_attention_entropy_threshold: float = 11.921
    high_attention_entropy_threshold: float = 11.926

    # Attention Varentropy Thresholds
    low_attention_varentropy_threshold: float = 0.001
    medium_attention_varentropy_threshold: float = 0.0045
    high_attention_varentropy_threshold: float = 0.009

    # Agreement Thresholds
    low_agreement_threshold: float = 2e-06
    medium_agreement_threshold: float = 4e-06
    high_agreement_threshold: float = 5e-06

    # Interaction Strength Thresholds
    low_interaction_strength_threshold: float = 0.2
    medium_interaction_strength_threshold: float = 0.247
    high_interaction_strength_threshold: float = 0.264

    # Offsets and Coefficients for Adjusting Sampling Parameters
    high_entropy_attention_offset: float = 1.3
    high_entropy_attention_coefficient: float = 0.2

    low_entropy_interaction_strength_offset: float = 1.2
    low_entropy_interaction_strength_coefficient: float = 0.3

    high_entropy_varentropy_attention_offset: float = 2.0
    high_entropy_varentropy_attention_coefficient: float = 0.5

    # Adaptive Sampling Parameters
    number_of_adaptive_samples: int = 5

    adaptive_temperature_logits_coefficient: float = 0.3
    adaptive_temperature_attention_coefficient: float = 0.2
    adaptive_temperature_agreement_coefficient: float = 0.2
    adaptive_top_p_coefficient: float = 0.1
    adaptive_top_k_interaction_coefficient: float = 0.3
    adaptive_top_k_agreement_coefficient: float = 0.2
    adaptive_min_p_coefficient: float = 0.5
    adaptive_score_logits_entropy_coefficient: float = 0.1
    adaptive_score_attention_entropy_coefficient: float = 0.2
    adaptive_score_logits_varentropy_coefficient: float = 0.3
    adaptive_score_attention_varentropy_coefficient: float = 0.4
    adaptive_score_agreement_coefficient: float = 0.5
    adaptive_score_interaction_strength_coefficient: float = 0.6


def sample(gen_tokens: jax.Array, logits: jax.Array, attention_scores: jax.Array, cfg: SamplerConfig,
           clarifying_question_token: int = 2564, key=jax.random.PRNGKey(1337)) -> jax.Array:

    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    color = get_color_for_metric(metrics ,cfg)
    #print(f'{metrics=}')

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32), color

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength < cfg.low_interaction_strength_threshold):
        # Insert a clarifying question token if not already present
        if not jnp.isin(gen_tokens[:, -1], clarifying_question_token).any():
            return jnp.array([[clarifying_question_token]]), color
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent  # Increase temperature based on attention entropy
            return _sample(
                logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_probability,
                key=key
            ), color

    # Low Entropy, High Varentropy: "exploring forks in the path"
    elif (ent < cfg.high_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength > cfg.low_interaction_strength_threshold):
        temp_adj = cfg.low_entropy_interaction_strength_offset + cfg.low_entropy_interaction_strength_coefficient * interaction_strength  # Increase temperature based on interaction strength
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))  # Increase top_k when agreement is low
        return _sample(
            logits,
            temperature=min(1.5, cfg.temperature * temp_adj),
            top_p=cfg.top_p,
            top_k=top_k_adj,
            min_p=cfg.min_probability,
            key=key
        ), color

    # High Entropy, High Varentropy: "resampling in the mist"
    elif (ent > cfg.medium_logits_entropy_threshold and
          vent > cfg.high_logits_varentropy_threshold and
          attn_ent > cfg.high_attention_entropy_threshold and
          attn_vent > cfg.high_attention_varentropy_threshold and
          agreement > cfg.high_agreement_threshold and
          interaction_strength > cfg.high_interaction_strength_threshold):
        # Use high temperature and adjusted top_p based on attention metrics
        temp_adj = cfg.high_entropy_varentropy_attention_offset + cfg.high_entropy_varentropy_attention_coefficient * attn_vent  # Increase temperature based on attention varentropy
        top_p_adj = max(0.5, cfg.top_p - cfg.high_entropy_attention_coefficient * attn_ent)  # Decrease top_p when attention entropy is high
        return _sample(
            logits,
            temperature=max(2.0, cfg.temperature * temp_adj),
            top_p=top_p_adj,
            top_k=cfg.top_k,
            min_p=cfg.min_probability,
            key=key
        ), color

    # Middle ground: use adaptive sampling
    else:
        logits_uncertainty = metrics["logits_entropy"] + metrics["logits_varentropy"]
        attn_uncertainty = metrics["attn_entropy"] + metrics["attn_varentropy"]

        temperature = cfg.temperature * (
            1 +
            cfg.adaptive_temperature_logits_coefficient * ent +
            cfg.adaptive_temperature_attention_coefficient * attn_ent -
            cfg.adaptive_temperature_agreement_coefficient * agreement
        )
        top_p = jnp.clip(
            cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent),
            0.1,
            1.0
        )
        top_k = int(jnp.clip(
            jnp.round(cfg.top_k * (
                1 +
                cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() -
                cfg.adaptive_top_k_agreement_coefficient * agreement.item()
            )),
            a_min=1,
            a_max=100
        ))
        min_p = jnp.clip(
            cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient * vent),
            0.01,
            0.5
        )

        keys = jax.random.split(key, cfg.number_of_adaptive_samples)

        samples = []
        for sample_key in keys:
            sample = _sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                key=sample_key
            )
            samples.append(sample)

        def score_sample(sample):
            log_prob = jnp.sum(
                jax.nn.log_softmax(logits) * jax.nn.one_hot(sample, logits.shape[-1]),
                axis=-1
            )
            confidence_score = (
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = jnp.array([score_sample(sample) for sample in samples])
        best_sample_idx = jnp.argmax(sample_scores)
        return samples[best_sample_idx], color

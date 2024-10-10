import torch
import torch.nn.functional as F
from typing import Tuple, Dict
from dataclasses import dataclass

# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def multinomial_sample_one(probs_sort: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    # Use torch.rand instead of Exponential distribution
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)

def _sample(logits: torch.Tensor, temperature=0.666, top_p=0.90, top_k=27, min_p: float = 0.0, generator: torch.Generator = None) -> torch.Tensor:
    bsz = logits.shape[0]
    logit = logits[:, -1]
    probs = F.softmax(logit / temperature, dim=-1)

    # Apply min_p sampling
    if min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)

    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # Apply top-p sampling
    mask = torch.where(probs_sum - probs_sort > top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = multinomial_sample_one(probs_sort, generator)
    # Convert next_token to int64 before using it in gather
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))
    return next_token_g.to(torch.int32)

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor) -> Dict[str, torch.Tensor]:
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)
    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(attention_probs * torch.log2(torch.clamp(attention_probs, 1e-10, 1.0)), dim=-1)
    attn_varentropy = torch.var(attn_entropy, dim=1)
    
    # Add a small epsilon to avoid NaN when all values are the same
    attn_varentropy = torch.where(torch.isnan(attn_varentropy), torch.zeros_like(attn_varentropy), attn_varentropy)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }

@dataclass
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

def sample(gen_tokens: torch.Tensor, logits: torch.Tensor, attention_scores: torch.Tensor,
           cfg: SamplerConfig,
           clarifying_question_token: int = 2564,
           generator: torch.Generator = torch.Generator(device=device).manual_seed(1337)) -> torch.Tensor:
    metrics = calculate_metrics(logits, attention_scores)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    agreement = metrics["agreement"]
    interaction_strength = metrics["interaction_strength"]
    color = get_color_for_metric(metrics, cfg)

    # Low Entropy, Low Varentropy: "flowing with unspoken intent"
    if (ent < cfg.low_logits_entropy_threshold and
        vent < cfg.low_logits_varentropy_threshold and
        attn_ent < cfg.low_attention_entropy_threshold and
        attn_vent < cfg.low_attention_varentropy_threshold and
        agreement < cfg.low_agreement_threshold and
        interaction_strength < cfg.low_interaction_strength_threshold):
        return torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)

    # High Entropy, Low Varentropy: "treading carefully, asking clarifying questions"
    elif (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold and
          agreement < cfg.low_agreement_threshold and
          interaction_strength < cfg.low_interaction_strength_threshold):
        # Insert a clarifying question token if not already present
        if not torch.isin(gen_tokens[:,-1], torch.tensor([clarifying_question_token], device=device)).any():
            return torch.tensor([[clarifying_question_token]], dtype=torch.int32, device=device)  # Assuming 2564 is our "ask clarifying question" token
        else:
            # If we've just asked a question, sample with slightly higher temperature
            temp_adj = cfg.high_entropy_attention_offset + cfg.high_entropy_attention_coefficient * attn_ent  # Increase temperature based on attention entropy
            return _sample(
                logits,
                temperature=min(1.5, cfg.temperature * temp_adj),
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                min_p=cfg.min_probability,
                generator=generator
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
            generator=generator
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
            generator=generator
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
        top_p = torch.clamp(cfg.top_p * (1 + cfg.adaptive_top_p_coefficient * attn_vent), 0.1, 1.0)
        top_k = int(torch.clamp(
            torch.round(torch.tensor(cfg.top_k) 
                        * (1 + cfg.adaptive_top_k_interaction_coefficient * interaction_strength.item() - 
                           cfg.adaptive_top_k_agreement_coefficient * agreement.item())),
            min=1,
            max=100
        ).item())
        min_p = torch.clamp(cfg.min_probability * (1 - cfg.adaptive_min_p_coefficient* vent), 0.01, 0.5)

        samples = []
        for _ in range(cfg.number_of_adaptive_samples):
            sample = _sample(logits, temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, generator=generator)
            samples.append(sample)

        def score_sample(sample):
            # Flatten the sample tensor and convert to long (int64)
            sample_flat = sample.flatten().to(torch.long)
            
            # Create one-hot encoding
            one_hot = F.one_hot(sample_flat, logits.shape[-1])
            
            # Reshape log_softmax output to match one_hot
            log_probs = F.log_softmax(logits, dim=-1).view(-1, logits.shape[-1])
            
            # Calculate log probability
            log_prob = torch.sum(log_probs * one_hot)
            
            confidence_score = (
                (1 - ent / cfg.high_logits_entropy_threshold) * cfg.adaptive_score_logits_entropy_coefficient +
                (1 - attn_ent / cfg.high_attention_entropy_threshold) * cfg.adaptive_score_attention_entropy_coefficient +
                (1 - vent / cfg.high_logits_varentropy_threshold) * cfg.adaptive_score_logits_varentropy_coefficient +
                (1 - attn_vent / cfg.high_attention_varentropy_threshold) * cfg.adaptive_score_attention_varentropy_coefficient +
                (agreement / cfg.high_agreement_threshold) * cfg.adaptive_score_agreement_coefficient +
                (interaction_strength / cfg.high_interaction_strength_threshold) * cfg.adaptive_score_interaction_strength_coefficient
            )
            return log_prob + confidence_score

        sample_scores = torch.stack([score_sample(sample) for sample in samples])
        best_sample_idx = torch.argmax(sample_scores)
        return samples[best_sample_idx], color
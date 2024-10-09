import mlx.core as mx
import mlx.nn as nn
from typing import Dict, Tuple, NamedTuple
import math


class SamplerConfig(NamedTuple):
  temperature: float = 0.666
  top_p: float = 0.90
  top_k: int = 27
  min_p: float = 0.0


def calculate_varentropy_logsoftmax(
  logits: mx.array, axis: int = -1
) -> Tuple[mx.array, mx.array]:
  logsoftmax = nn.log_softmax(logits, axis=axis)
  softmax = mx.exp(logsoftmax)
  entropy = -mx.sum(softmax * logsoftmax, axis=axis)
  varentropy = mx.sum(softmax * (logsoftmax + entropy.reshape(-1, 1)) ** 2, axis=axis)
  return entropy, varentropy


def multinomial_sample_one(probs_sort: mx.array) -> mx.array:
  # MLX doesn't have a direct equivalent to JAX's random.choice or PyTorch's multinomial
  # We'll implement a simple version using cumulative sum and uniform random
  cumsum = mx.cumsum(probs_sort, axis=-1)
  rand = mx.random.uniform(shape=(1,))
  return mx.argmax(cumsum > rand, axis=-1)


def _sample(
  logits: mx.array,
  temperature: float | mx.array,
  top_p: float | mx.array,
  top_k: int | mx.array,
  min_p: float | mx.array,
) -> mx.array:
  # Temperature scaling
  logits = logits / temperature

  # Top-k filtering
  if top_k > 0:
    top_k_logits = mx.topk(logits, k=min(top_k, logits.shape[-1]), axis=-1)
    min_top_k = mx.min(top_k_logits, axis=-1, keepdims=True)
    logits = mx.where(logits < min_top_k, -float("inf"), logits)

  # Top-p (nucleus) filtering
  sorted_logits = mx.sort(logits, axis=-1)  # Sort in ascending order
  sorted_logits = sorted_logits[..., ::-1]
  cumulative_probs = mx.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
  sorted_indices_to_remove = cumulative_probs > top_p
  sorted_indices_to_remove = mx.concatenate(
    [
      mx.zeros_like(sorted_indices_to_remove[..., :1]),
      sorted_indices_to_remove[..., :-1],
    ],
    axis=-1,
  )
  min_sorted_logits = mx.where(sorted_indices_to_remove, -float("inf"), sorted_logits)
  min_sorted_logit = mx.max(min_sorted_logits, axis=-1, keepdims=True)
  logits = mx.where(logits < min_sorted_logit, -float("inf"), logits)

  # Min-p filtering
  probs = nn.softmax(logits, axis=-1)
  min_p_mask = probs < min_p
  logits = mx.where(min_p_mask, -float("inf"), logits)

  return mx.random.categorical(nn.softmax(logits, axis=-1))


def calculate_metrics(
  logits: mx.array, attention_scores: mx.array
) -> Dict[str, mx.array]:
  entropy, varentropy = calculate_varentropy_logsoftmax(logits)
  attention_entropy = -mx.sum(
    attention_scores * mx.log(attention_scores + 1e-10), axis=-1
  )
  return {
    "entropy": entropy,
    "varentropy": varentropy,
    "attention_entropy": attention_entropy,
  }


def adaptive_sample(
  logits: mx.array,
  metrics: Dict[str, mx.array],
  base_temp: float = 0.666,
  min_temp: float = 0.1,
  max_temp: float = 2.0,
  base_top_p: float = 0.9,
  min_top_p: float = 0.1,
  max_top_p: float = 1.0,
) -> mx.array:
  # Implement adaptive sampling logic here
  # This is a placeholder and should be adapted based on your specific requirements
  temp_scale = mx.clip(metrics["entropy"] / mx.log(logits.shape[-1]), 0, 1)
  temperature = base_temp + (max_temp - min_temp) * temp_scale
  top_p = base_top_p + (max_top_p - min_top_p) * (1 - temp_scale)
  return _sample(logits, temperature=temperature, top_p=top_p, top_k=0, min_p=0.0)


def score_sample(sample: mx.array):
  # Implement sample scoring logic here
  # This is a placeholder and should be adapted based on your specific requirements
  return mx.sum(sample)  # Example scoring function


def sample(
  gen_tokens: mx.array,
  logits: mx.array,
  attention_scores: mx.array,
  cfg: SamplerConfig,
  adaptive: bool = False,
  num_samples: int = 1,
) -> mx.array:
  metrics = calculate_metrics(logits, attention_scores)

  if adaptive:
    return adaptive_sample(logits, metrics)

  samples = [
    _sample(logits, cfg.temperature, cfg.top_p, cfg.top_k, cfg.min_p)
    for _ in range(num_samples)
  ]

  if num_samples > 1:
    scores = [score_sample(s) for s in samples]
    best_sample = samples[mx.argmax(mx.array(scores))]
    return best_sample
  else:
    return samples[0]

from typing import NamedTuple, Optional, Tuple, List
import jax
import jax.numpy as jnp
import jax.random as random
import math
import tyro
from pathlib import Path
from functools import partial
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.tokenizer import Tokenizer
from entropix.weights import load_weights
from entropix.utils import precompute_freqs_cis, build_attn_mask, sample

def calculate_perplexity(xfmr_weights, model_params, tokenizer, text):
    tokens = jnp.array([tokenizer.encode(text, bos=False, eos=False)], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, 0)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    logits, _ = xfmr(xfmr_weights, model_params, tokens, 0, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    token_log_probs = jnp.take_along_axis(log_probs[:, :-1], tokens[:, 1:, None], axis=-1).squeeze(-1)
    return jnp.exp(-jnp.mean(token_log_probs))

def generate(xfmr_weights, model_params, tokenizer, prompt, max_tokens, temperature=0.8, seed=None):
    rng = random.PRNGKey(seed) if seed is not None else random.PRNGKey(0)
    tokens = tokenizer.encode(prompt, bos=False, eos=False)
    gen_tokens = []
    cur_pos = 0
    tokens = jnp.array([tokens], jnp.int32)
    bsz, seqlen = tokens.shape
    attn_mask = build_attn_mask(seqlen, cur_pos)
    freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    
    for _ in range(max_tokens):
        logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache, attn_mask=attn_mask)
        rng, subkey = random.split(rng)
        next_token = sample(logits, temperature=temperature, key=subkey)
        gen_tokens.append(next_token.item())
        tokens = next_token
        cur_pos += 1
        if next_token.item() in tokenizer.stop_tokens:
            break
    
    return tokenizer.decode(gen_tokens)

def calculate_bleu(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score between reference and hypothesis."""
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing)

def calculate_distinct_ngrams(text: str, n: int) -> float:
    """Calculate the ratio of distinct n-grams to total n-grams."""
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    distinct_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    return distinct_ngrams / total_ngrams if total_ngrams > 0 else 0

def calculate_metrics(reference: str, generated: str) -> dict:
    """Calculate various metrics for the generated text."""
    return {
        "bleu": calculate_bleu(reference, generated),
        "distinct_1grams": calculate_distinct_ngrams(generated, 1),
        "distinct_2grams": calculate_distinct_ngrams(generated, 2),
        "distinct_3grams": calculate_distinct_ngrams(generated, 3),
    }

def generate_and_evaluate(xfmr_weights, model_params, tokenizer, prompt, reference, max_tokens=100, num_samples=5, temperature=0.8):
    results = []
    for i in range(num_samples):
        generated_text = generate(xfmr_weights, model_params, tokenizer, prompt, max_tokens, temperature, seed=i)
        perplexity = calculate_perplexity(xfmr_weights, model_params, tokenizer, generated_text)
        metrics = calculate_metrics(reference, generated_text)
        results.append({
            "generated_text": generated_text,
            "perplexity": float(perplexity),
            **metrics
        })
    return results

def main():
    print(f"Using device: {jax.devices()[0]}")
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights()
    tokenizer = Tokenizer('entropix/tokenizer.model')

    prompt = "Long time ago somewhere inside the earth"
    reference = "Long time ago somewhere inside the earth, there was a hidden world of wonders waiting to be discovered."
    
    temperatures = [0.6, 0.7, 0.8, 0.9]
    for temp in temperatures:
        print(f"\nEvaluating with temperature {temp}")
        results = generate_and_evaluate(xfmr_weights, model_params, tokenizer, prompt, reference, max_tokens=150, num_samples=5, temperature=temp)
        
        avg_metrics = {key: sum(r[key] for r in results) / len(results) for key in results[0] if key != 'generated_text'}
        print("\nAverage Metrics:")
        for key, value in avg_metrics.items():
            print(f"{key.capitalize()}: {value}")

if __name__ == '__main__':
    tyro.cli(main)
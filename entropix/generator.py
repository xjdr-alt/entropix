from entropix.model import KVCache, xfmr
from entropix.rope import precompute_freqs_cis
from entropix.sampler import sample, SamplerConfig
from entropix.stats import AttnStats
from entropix.sampler import SamplerConfig
from entropix.model import xfmr
from entropix.sampler import _sample
import jax.numpy as jnp
import jax
from typing import NamedTuple

class InitialState(NamedTuple):
    tokens: jax.Array
    kvcache: KVCache
    attn_stats: AttnStats
    freqs_cis: jax.Array
    attn_mask: jax.Array
    stop_tokens: jax.Array
    sampler_cfg: SamplerConfig
    logits_cache: jax.Array


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
    mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
    if seqlen > 1:
        mask = jnp.full((seqlen, seqlen), float('-inf'))
        mask = jnp.triu(mask, k=1)
        mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
    return mask


def initialize(model_params, tokens, max_gen_len):    
    tokens = jnp.array([tokens], dtype=jnp.int32)
    bsz, seqlen = tokens.shape
    max_total_len = seqlen + max_gen_len
    attn_mask = build_attn_mask(seqlen, 0)
    freqs_cis = precompute_freqs_cis(model_params)
    kvcache = KVCache.new(model_params, bsz, max_total_len)
    attn_stats = AttnStats.new(model_params, bsz, max_total_len)
    logits_cache = jnp.zeros((bsz, max_total_len, model_params.vocab_size), dtype=jnp.float32)
    stop = jnp.array([128001, 128008, 128009], dtype=jnp.int32)
    sampler_cfg = SamplerConfig()
    return {
        'tokens': tokens,
        'kvcache': kvcache,
        'attn_stats': attn_stats,
        'freqs_cis': freqs_cis,
        'attn_mask': attn_mask,
        'stop_tokens': stop,
        'sampler_cfg': sampler_cfg,
        'logits_cache': logits_cache,
    }

def generate(xfmr_weights, model_params, tokenizer, initial_state, max_gen_len):
    kvcache = initial_state['kvcache']
    attn_stats = initial_state['attn_stats']
    attn_mask = initial_state['attn_mask']
    freqs_cis = initial_state['freqs_cis']
    stop_tokens = initial_state['stop_tokens']
    sampler_cfg = initial_state['sampler_cfg']
    tokens = initial_state['tokens']

    prompt_len = tokens.shape[1]

    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, 0, freqs_cis[:prompt_len], kvcache, attn_stats, attn_mask=attn_mask)
    cur_pos, max_total_len = prompt_len, prompt_len + max_gen_len    
    next_token = jnp.argmax(logits[:,-1], axis=-1, keepdims=True).astype(jnp.int32)
    gen_tokens = next_token
    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    while cur_pos < max_total_len:
      cur_pos += 1
      logits, kvcache, scores, attn_stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache, attn_stats)
      next_token = sample(gen_tokens, logits, scores, cfg=sampler_cfg)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
      if jnp.isin(next_token, stop_tokens).any():
        break


def vanilla_generate(xfmr_weights, model_params, tokenizer, initial_state, n_gen_tokens, rng):
    

    kvcache = initial_state['kvcache']
    attn_stats = initial_state['attn_stats']
    attn_mask = initial_state['attn_mask']
    freqs_cis = initial_state['freqs_cis']
    sampler_cfg = initial_state['sampler_cfg']
    logits_cache = initial_state['logits_cache']
    tokens = initial_state['tokens']

    prompt_len = tokens.shape[1]

    logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, 0, freqs_cis[:prompt_len], kvcache, attn_stats, attn_mask=attn_mask)
    logits_cache = logits_cache.at[:, :prompt_len, :].set(logits)
    cur_pos, max_total_len = prompt_len, prompt_len + n_gen_tokens
    next_token = _sample(logits_cache[:, cur_pos:cur_pos+1, :], temperature=sampler_cfg.temp, min_p=sampler_cfg.min_p, top_k=sampler_cfg.top_k, top_p=sampler_cfg.top_p, key=rng)
    gen_tokens = next_token

    print(tokenizer.decode([next_token.item()]), end='', flush=True)
    while cur_pos < max_total_len:
      cur_pos += 1
      logits, kvcache, scores, attn_stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache, attn_stats)
      logits_cache = logits_cache.at[:, cur_pos:cur_pos+1, :].set(logits)
      next_token = _sample(logits, temperature=sampler_cfg.temp, min_p=sampler_cfg.min_p, top_k=sampler_cfg.top_k, top_p=sampler_cfg.top_p, key=rng)
      gen_tokens = jnp.concatenate((gen_tokens, next_token))
      print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
    return gen_tokens, logits_cache, attn_stats


@jax.jit
def generate_next_token(xfmr_weights, model_params, rng, input_ids, kvcache, attn_stats, attn_mask, freqs_cis, temperature=1.0):
    """
    Generates the next token for a single sequence.

    Args:
        xfmr_weights: Transformer weights.
        model_params: Model parameters.
        rng: JAX random number generator key.
        input_ids: Current input token IDs.
        kvcache: KVCache instance for caching keys and values.
        attn_stats: AttnStats instance for attention statistics.
        attn_mask: Attention mask.
        freqs_cis: Precomputed frequencies for rotary embeddings.
        temperature: Sampling temperature.

    Returns:
        next_token: The next generated token ID.
        updated_kvcache: Updated KVCache after generating the token.
        updated_attn_stats: Updated AttnStats after generating the token.
    """
    # Get logits from the model
    logits, updated_kvcache, scores, updated_attn_stats = xfmr(
        xfmr_weights,
        model_params,
        input_ids,
        input_ids.shape[-1] - 1,  # current position
        freqs_cis[input_ids.shape[-1] - 1 : input_ids.shape[-1]],
        kvcache,
        attn_stats,
        attn_mask=attn_mask
    )
    
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Sample the next token using categorical distribution
    next_token = jax.random.categorical(rng, scaled_logits).reshape(-1, 1)
    
    return next_token, updated_kvcache, updated_attn_stats

@jax.jit
def generate_sequence_batch(xfmr_weights, model_params, tokenizer, rng, input_ids, max_length, temperature=1.0):
    """
    Generates a single sequence by autoregressively sampling tokens.

    Args:
        xfmr_weights: Transformer weights.
        model_params: Model parameters.
        tokenizer: Tokenizer instance.
        rng: JAX random number generator key.
        input_ids: Initial input token IDs of shape (batch_size, prompt_length).
        max_length: Maximum number of tokens to generate.
        temperature: Sampling temperature.

    Returns:
        generated_sequences: Generated token IDs of shape (batch_size, prompt_length + max_length).
        final_kvcache: Final KVCache after generation.
        final_attn_stats: Final AttnStats after generation.
    """
    # Initialize the generation state using _initialize
    initial_state = _initialize(model_params, input_ids, max_length)
    
    tokens = initial_state['tokens']
    kvcache = initial_state['kvcache']
    attn_stats = initial_state['attn_stats']
    freqs_cis = initial_state['freqs_cis']
    attn_mask = initial_state['attn_mask']
    stop_tokens = initial_state['stop_tokens']
    sampler_cfg = initial_state['sampler_cfg']
    
    generated_sequences = tokens
    rng = jax.random.split(rng, max_length)[0]  # Split once for the entire generation
    
    for pos in range(tokens.shape[1], initial_state['kvcache'].k.shape[2]):
        # Generate next token
        next_token, kvcache, attn_stats = generate_next_token(
            xfmr_weights,
            model_params,
            rng,
            generated_sequences,
            kvcache,
            attn_stats,
            attn_mask,
            freqs_cis,
            temperature
        )
        generated_sequences = jnp.concatenate([generated_sequences, next_token], axis=-1)
        
        # Early stopping if a stop token is generated
        if jnp.isin(next_token, stop_tokens).any():
            break

    return generated_sequences, kvcache, attn_stats

@jax.jit
def gather_logprobs(logprobs, eval_token_ids):
    """
    Gathers log probabilities for the generated tokens.

    Args:
        logprobs: Log probabilities from the model of shape (batch_size, vocab_size).
        eval_token_ids: Generated token IDs of shape (batch_size, max_length).

    Returns:
        sequence_log_probs: Log probabilities of the generated tokens of shape (batch_size, max_length).
    """
    return jnp.take_along_axis(logprobs, eval_token_ids[..., None], axis=-1).squeeze(-1)


@jax.jit
def compute_sequence_scores(token_logprobs):
    """
    Computes the sequence scores as the negative sum of log probabilities.

    Args:
        token_logprobs: Log probabilities of the generated tokens of shape (batch_size, max_length).

    Returns:
        sequence_scores: Negative sum of log probabilities of shape (batch_size,).
    """
    return -jnp.sum(token_logprobs, axis=-1)


@jax.jit
def calculate_logits(xfmr_weights, model_params, generated_sequences, freqs_cis, kvcache, attn_stats, attn_mask):
    """
    Calculates logits for the generated sequences.

    Args:
        xfmr_weights: Transformer weights.
        model_params: Model parameters.
        generated_sequences: Generated token IDs of shape (batch_size, total_length).
        freqs_cis: Precomputed frequencies for rotary embeddings.
        kvcache: KVCache instance.
        attn_stats: AttnStats instance.
        attn_mask: Attention mask.

    Returns:
        logits: Logits from the model of shape (batch_size, vocab_size).
    """
    logits, _, _, _ = xfmr(
        xfmr_weights,
        model_params,
        generated_sequences,
        generated_sequences.shape[-1] - 1,  # Current position
        freqs_cis[generated_sequences.shape[-1] - 1 : generated_sequences.shape[-1]],
        kvcache,
        attn_stats,
        attn_mask=attn_mask
    )
    return logits


def generate_and_evaluate_sequences(xfmr_weights, model_params, tokenizer, rng, N: int, max_length: int, input_ids: jax.Array, temperature: float) -> jax.Array:
    """
    Generates N autoregressive continuations of the input prompt and evaluates their log-probabilities.

    Args:
        xfmr_weights: Transformer weights.
        model_params: Model parameters.
        tokenizer: Tokenizer instance for encoding and decoding.
        rng: JAX random number generator key.
        N: Number of sequences to generate.
        max_length: Maximum number of tokens to generate for each sequence.
        input_ids: Input token IDs as a JAX array of shape (1, prompt_length).
        temperature: Sampling temperature to control randomness.

    Returns:
        evaluated_sequences: A JAX array of shape (N, 1 + prompt_length + max_length) where the first column
                             contains the negative log-probabilities and the remaining columns contain the
                             generated token IDs.
    """
    # Repeat the input_ids N times to create a batch
    batch_input_ids = jnp.repeat(input_ids, N, axis=0)  # Shape: (N, prompt_length)

    # Split RNG keys for each sequence in the batch
    rng_keys = jax.random.split(rng, N)

    # Vectorize the generate_sequence_batch function to handle N sequences in parallel
    vectorized_generate = jax.vmap(
        generate_sequence_batch,
        in_axes=(None, None, None, 0, 0, None),
        out_axes=(0, 0, 0)
    )

    # Generate sequences in batch
    generated_sequences, final_kvcache, final_attn_stats = vectorized_generate(
        xfmr_weights,
        model_params,
        tokenizer,
        rng_keys,
        batch_input_ids,
        max_length,
        temperature
    )

    # Compute frequencies cis for the last position
    last_pos = generated_sequences.shape[-1] - 1
    freqs_cis = precompute_freqs_cis(model_params)[last_pos:last_pos+1]

    # Compute logits for the generated sequences
    logits = calculate_logits(
        xfmr_weights,
        model_params,
        generated_sequences,
        freqs_cis,
        final_kvcache,
        final_attn_stats,
        build_attn_mask(generated_sequences.shape[-1], 0)
    )

    # Compute log probabilities
    logprobs = jax.nn.log_softmax(logits, axis=-1)  # Shape: (N, vocab_size)

    # Extract the generated token IDs beyond the prompt
    eval_token_ids = generated_sequences[:, -max_length:]  # Shape: (N, max_length)

    # Gather the log probabilities of the generated tokens
    sequence_log_probs = gather_logprobs(logprobs, eval_token_ids)  # Shape: (N, max_length)

    # Calculate the sequence scores as the negative sum of log probabilities
    sequence_scores = compute_sequence_scores(sequence_log_probs)  # Shape: (N,)

    # Combine the scores with the generated sequences
    evaluated_sequences = jnp.concatenate([sequence_scores[:, None], generated_sequences], axis=-1)  # Shape: (N, 1 + prompt_length + max_length)

    return evaluated_sequences
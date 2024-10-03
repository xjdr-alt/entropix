import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import jax
import jax.numpy as jnp
from entropix.config import LLAMA_1B_PARAMS
from entropix.weights import load_weights
from entropix.tokenizer import Tokenizer
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.utils import precompute_freqs_cis, build_attn_mask, sample

@pytest.fixture
def model_setup():
    model_params = LLAMA_1B_PARAMS
    xfmr_weights = load_weights()
    tokenizer = Tokenizer('entropix/tokenizer.model')
    return model_params, xfmr_weights, tokenizer

def test_generation_pipeline(model_setup):
    model_params, xfmr_weights, tokenizer = model_setup
    
    prompt = "Once upon a time"
    tokens = tokenizer.encode(prompt, bos=False, eos=False)
    
    def generate(tokens, max_new_tokens=10):
        gen_tokens = []
        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
        
        for _ in range(max_new_tokens):
            logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[cur_pos:cur_pos+seqlen], kvcache, attn_mask=attn_mask)
            next_token = sample(logits)
            gen_tokens.append(next_token.item())
            tokens = next_token
            cur_pos += 1
            if next_token.item() in tokenizer.stop_tokens:
                break
        
        return gen_tokens
    
    generated_tokens = generate(tokens)
    generated_text = tokenizer.decode(generated_tokens)
    
    assert len(generated_tokens) > 0
    assert isinstance(generated_text, str)
    assert len(generated_text) > 0

def test_generation_with_different_prompts(model_setup):
    model_params, xfmr_weights, tokenizer = model_setup
    
    prompts = [
        "The quick brown fox",
        "In a galaxy far, far away",
        "It was the best of times",
    ]
    
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, bos=False, eos=False)
        
        def generate(tokens, max_new_tokens=10):
            gen_tokens = []
            cur_pos = 0
            tokens = jnp.array([tokens], jnp.int32)
            bsz, seqlen = tokens.shape
            attn_mask = build_attn_mask(seqlen, cur_pos)
            freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
            kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
            
            for _ in range(max_new_tokens):
                logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[cur_pos:cur_pos+seqlen], kvcache, attn_mask=attn_mask)
                next_token = sample(logits)
                gen_tokens.append(next_token.item())
                tokens = next_token
                cur_pos += 1
                if next_token.item() in tokenizer.stop_tokens:
                    break
            
            return gen_tokens
        
        generated_tokens = generate(tokens)
        generated_text = tokenizer.decode(generated_tokens)
        
        assert len(generated_tokens) > 0
        assert isinstance(generated_text, str)
        assert len(generated_text) > 0
        assert generated_text != prompt  # Ensure the model generated new text

def test_generation_respects_max_length(model_setup):
    model_params, xfmr_weights, tokenizer = model_setup
    
    prompt = "This is a test prompt"
    tokens = tokenizer.encode(prompt, bos=False, eos=False)
    max_new_tokens = 5
    
    def generate(tokens, max_new_tokens):
        gen_tokens = []
        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        freqs_cis = precompute_freqs_cis(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
        kvcache = KVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
        
        for _ in range(max_new_tokens):
            logits, kvcache = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[cur_pos:cur_pos+seqlen], kvcache, attn_mask=attn_mask)
            next_token = sample(logits)
            gen_tokens.append(next_token.item())
            tokens = next_token
            cur_pos += 1
            if next_token.item() in tokenizer.stop_tokens:
                break
        
        return gen_tokens
    
    generated_tokens = generate(tokens, max_new_tokens)
    
    assert len(generated_tokens) <= max_new_tokens

# Add more integration tests as needed
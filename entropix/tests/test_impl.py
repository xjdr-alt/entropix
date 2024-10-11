# This code tests the torch and jax implementations at float32 with JIT compilation for jax except for attention
import pytest
import jax
import jax.numpy as jnp
import torch
import torch.nn.functional as F
import numpy as np
import ml_dtypes
from pathlib import Path

from entropix.config import LLAMA_1B_PARAMS, ModelParams
from entropix.tokenizer import Tokenizer
from entropix.torch_kvcache import KVCache as TorchKVCache
from entropix.torch_model import (
    rms_norm as torch_rms_norm,
    attention as torch_attention,
    feed_forward as torch_feed_forward,
    apply_rotary_emb as torch_apply_rotary_emb,
)
from entropix.torch_weights import (
    load_weights as load_torch_weights,
    LayerWeights as TorchLayerWeights,
    XfmrWeights as TorchXfmrWeights,
)
from entropix.torch_main import (
    build_attn_mask as build_attn_mask_torch,
    precompute_freqs_cis as precompute_freqs_cis_torch,
)

from entropix.kvcache import KVCache as JaxKVCache
from entropix.model import (
    rms_norm as jax_rms_norm,
    attention as jax_attention,
    feed_forward as jax_feed_forward,
    apply_rotary_emb as jax_apply_rotary_emb,
)
from entropix.weights import (
    XfmrWeights as JaxXfmrWeights,
    load_weights as load_jax_weights,
    LayerWeights as JaxLayerWeights,
)
from entropix.main import build_attn_mask as build_attn_mask_jax, precompute_freqs_cis as precompute_freqs_cis_jax
from typing import NamedTuple


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTestSetup(NamedTuple):
    model_params: ModelParams
    torch_weights: TorchXfmrWeights
    jax_weights: JaxXfmrWeights
    torch_tokens: torch.Tensor
    jax_tokens: jax.Array
    bsz: int
    seqlen: int
    cur_pos: int
    attn_mask_torch: torch.Tensor
    attn_mask_jax: jax.Array
    freqs_cis_torch: torch.Tensor
    freqs_cis_jax: jax.Array


def compare_outputs(
    torch_output: torch.Tensor,
    jax_output: jax.Array,
    atol: float = 1e-3,
    rtol: float = 1e-5,
) -> None:
    jax_output_np = np.array(jax_output)
    if torch_output.dtype == torch.bfloat16:
        torch_output_np = torch_output.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
    else:
        torch_output_np = torch_output.cpu().numpy()

    assert jax_output_np.shape == torch_output_np.shape, f"Shapes do not match: {jax_output_np.shape} vs {torch_output_np.shape}"

    np.testing.assert_allclose(torch_output_np, jax_output_np, atol=atol, rtol=rtol)


def get_inputs(shape, torch_dtype, jax_dtype):
    data = np.random.randn(*shape)
    torch_out = torch.tensor(data, dtype=torch_dtype).to(device)
    jax_out = jnp.array(data, jax_dtype)
    jax_out = jax.device_put(jax_out, jax.devices("gpu")[0])
    return torch_out, jax_out


def build_fresh_kvcache(bsz, model_params):
    torch_kvcache = TorchKVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim).to(device)
    jax_kvcache = JaxKVCache.new(model_params.n_layers, bsz, model_params.max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)
    return torch_kvcache, jax_kvcache


@pytest.fixture(scope="module")
def model_setup():
    model_params = LLAMA_1B_PARAMS
    weight_path = Path("../weights/1B-Instruct")
    torch_weights = load_torch_weights(weight_path)
    jax_weights = load_jax_weights(weight_path)

    torch_float32_weights = []
    for layer_idx, torch_layer in enumerate(torch_weights.layer_weights):
        torch_layer_fp32 = TorchLayerWeights(
            wq=torch_layer.wq.float().to(device),
            wk=torch_layer.wk.float().to(device),
            wv=torch_layer.wv.float().to(device),
            wo=torch_layer.wo.float().to(device),
            w1=torch_layer.w1.float().to(device),
            w2=torch_layer.w2.float().to(device),
            w3=torch_layer.w3.float().to(device),
            ffn_norm=torch_layer.ffn_norm.float().to(device),
            attention_norm=torch_layer.attention_norm.float().to(device),
        )
        torch_float32_weights.append(torch_layer_fp32)

    torch_float32_weights = TorchXfmrWeights(
        layer_weights=torch_float32_weights,
        tok_embeddings=torch_weights.tok_embeddings.float().to(device),
        norm=torch_weights.norm.float().to(device),
        output=torch_weights.output.float().to(device),
    )

    jax_float32_weights = []
    for layer_idx, jax_layer in enumerate(jax_weights.layer_weights):
        jax_layer_fp32 = JaxLayerWeights(
            wq=jax_layer.wq.astype(jnp.float32),
            wk=jax_layer.wk.astype(jnp.float32),
            wv=jax_layer.wv.astype(jnp.float32),
            wo=jax_layer.wo.astype(jnp.float32),
            w1=jax_layer.w1.astype(jnp.float32),
            w2=jax_layer.w2.astype(jnp.float32),
            w3=jax_layer.w3.astype(jnp.float32),
            ffn_norm=jax_layer.ffn_norm.astype(jnp.float32),
            attention_norm=jax_layer.attention_norm.astype(jnp.float32),
        )
        jax_float32_weights.append(jax_layer_fp32)

    jax_float32_weights = JaxXfmrWeights(
        layer_weights=jax_float32_weights,
        tok_embeddings=jax_weights.tok_embeddings.astype(jnp.float32),
        norm=jax_weights.norm.astype(jnp.float32),
        output=jax_weights.output.astype(jnp.float32),
    )

    tokenizer = Tokenizer("tokenizer.model")
    input_text = "Test Prompt"
    tokens = tokenizer.encode(input_text, bos=False, eos=False, allowed_special="all")

    torch_tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    jax_tokens = jnp.array([tokens], jnp.int32)

    bsz, seqlen = torch_tokens.shape
    cur_pos = 0

    attn_mask_torch = build_attn_mask_torch(seqlen, cur_pos)
    attn_mask_jax = build_attn_mask_jax(seqlen, cur_pos)

    freqs_cis_torch = precompute_freqs_cis_torch(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)
    freqs_cis_jax = precompute_freqs_cis_jax(model_params.head_dim, model_params.max_seq_len, model_params.rope_theta, model_params.use_scaled_rope)

    return ModelTestSetup(
        model_params=model_params,
        torch_weights=torch_float32_weights,
        jax_weights=jax_float32_weights,
        torch_tokens=torch_tokens,
        jax_tokens=jax_tokens,
        bsz=bsz,
        seqlen=seqlen,
        cur_pos=cur_pos,
        attn_mask_torch=attn_mask_torch,
        attn_mask_jax=attn_mask_jax,
        freqs_cis_torch=freqs_cis_torch,
        freqs_cis_jax=freqs_cis_jax,
    )


def test_attn_mask(model_setup: ModelTestSetup):
    compare_outputs(model_setup.attn_mask_torch, model_setup.attn_mask_jax)


def test_freqs_cis(model_setup: ModelTestSetup):
    compare_outputs(model_setup.freqs_cis_torch, model_setup.freqs_cis_jax)


def test_token_embeddings(model_setup: ModelTestSetup):
    with torch.no_grad():
        torch_emb = model_setup.torch_weights.tok_embeddings
        jax_emb = model_setup.jax_weights.tok_embeddings
        jax_h = jax_emb[model_setup.jax_tokens]
        torch_h = torch_emb[model_setup.torch_tokens]
        compare_outputs(torch_h, jax_h)


def test_rms_norm(model_setup: ModelTestSetup):
    with torch.no_grad():
        torch_in, jax_in = get_inputs(
            (
                model_setup.bsz,
                model_setup.seqlen,
                model_setup.model_params.n_local_heads * model_setup.model_params.head_dim,
            ),
            torch.float32,
            jnp.float32,
        )
        torch_w = model_setup.torch_weights.layer_weights[0].attention_norm
        jax_w = model_setup.jax_weights.layer_weights[0].attention_norm

        torch_rms = torch_rms_norm(x=torch_in, w=torch_w)
        jax_rms = jax_rms_norm(x=jax_in, w=jax_w)
        compare_outputs(torch_rms, jax_rms)


def test_rope(model_setup: ModelTestSetup):
    with torch.no_grad():
        torch_xq, jax_xq = get_inputs(
            (
                model_setup.bsz,
                model_setup.seqlen,
                model_setup.model_params.n_local_heads,
                model_setup.model_params.head_dim,
            ),
            torch.bfloat16,
            jnp.bfloat16,
        )
        torch_xk, jax_xk = get_inputs(
            (
                model_setup.bsz,
                model_setup.seqlen,
                model_setup.model_params.n_local_kv_heads,
                model_setup.model_params.head_dim,
            ),
            torch.bfloat16,
            jnp.bfloat16,
        )

        torch_xq, torch_xk = torch_apply_rotary_emb(torch_xq, torch_xk, model_setup.freqs_cis_torch[: model_setup.seqlen])
        jax_xq, jax_xk = jax_apply_rotary_emb(jax_xq, jax_xk, model_setup.freqs_cis_jax[: model_setup.seqlen])

        compare_outputs(torch_xq, jax_xq)
        compare_outputs(torch_xk, jax_xk)


def test_attention(model_setup: ModelTestSetup):
    torch_h, jax_h = get_inputs(
        (
            model_setup.bsz,
            model_setup.seqlen,
            model_setup.model_params.n_local_heads * model_setup.model_params.head_dim,
        ),
        torch.float32,
        jnp.float32,
    )
    torch_kvcache, jax_kvcache = build_fresh_kvcache(model_setup.bsz, model_setup.model_params)
    with torch.no_grad():
        idx = 5
        torch_attn_out, torch_kvcache, torch_scores = torch_attention(
            x=torch_h,
            layer_weights=model_setup.torch_weights.layer_weights[idx],
            model_params=model_setup.model_params,
            cur_pos=model_setup.cur_pos,
            layer_idx=idx,
            freqs_cis=model_setup.freqs_cis_torch[: model_setup.seqlen],
            kvcache=torch_kvcache,
            attn_mask=model_setup.attn_mask_torch,
        )
        with jax.disable_jit():
            jax_attn_out, jax_kvcache, jax_scores = jax_attention(
                x=jax_h,
                layer_weights=model_setup.jax_weights.layer_weights[idx],
                model_params=model_setup.model_params,
                cur_pos=model_setup.cur_pos,
                layer_idx=idx,
                freqs_cis=model_setup.freqs_cis_jax[: model_setup.seqlen],
                kvcache=jax_kvcache,
                attn_mask=model_setup.attn_mask_jax,
            )

    compare_outputs(torch_attn_out, jax_attn_out)
    compare_outputs(torch_scores, jax_scores)
    compare_outputs(torch_kvcache.k, jax_kvcache.k)
    compare_outputs(torch_kvcache.v, jax_kvcache.v)


def test_feedforward(model_setup: ModelTestSetup):
    torch_x, jax_x = get_inputs(
        (
            model_setup.bsz,
            model_setup.seqlen,
            model_setup.model_params.n_local_heads * model_setup.model_params.head_dim,
        ),
        torch.float32,
        jnp.float32,
    )
    layer_idx = 0
    torch_layer = model_setup.torch_weights.layer_weights[layer_idx]
    jax_layer = model_setup.jax_weights.layer_weights[layer_idx]
    torch_rms = torch_rms_norm(x=torch_x, w=torch_layer.attention_norm)
    jax_rms = jax_rms_norm(x=jax_x, w=jax_layer.attention_norm)
    torch_ff_out = torch_feed_forward(torch_rms, torch_layer)
    jax_ff_out = jax_feed_forward(jax_rms, jax_layer)
    compare_outputs(torch_ff_out, jax_ff_out)


def test_each_layer(model_setup: ModelTestSetup):
    with torch.no_grad():
        for idx in range(model_setup.model_params.n_layers):
            torch_kvcache, jax_kvcache = build_fresh_kvcache(model_setup.bsz, model_setup.model_params)
            torch_h, jax_h = get_inputs(
                (
                    model_setup.bsz,
                    model_setup.seqlen,
                    model_setup.model_params.n_local_heads * model_setup.model_params.head_dim,
                ),
                torch.float32,
                jnp.float32,
            )
            compare_outputs(torch_h, jax_h)
            torch_layer = model_setup.torch_weights.layer_weights[idx]
            jax_layer = model_setup.jax_weights.layer_weights[idx]

            torch_norm_x = torch_rms_norm(x=torch_h, w=torch_layer.attention_norm)
            jax_norm_x = jax_rms_norm(x=jax_h, w=jax_layer.attention_norm)
            compare_outputs(torch_norm_x, jax_norm_x)

            torch_h_attn, torch_kvcache, torch_scores = torch_attention(
                x=torch_norm_x,
                layer_weights=torch_layer,
                model_params=model_setup.model_params,
                cur_pos=model_setup.cur_pos,
                layer_idx=idx,
                freqs_cis=model_setup.freqs_cis_torch[: model_setup.seqlen],
                kvcache=torch_kvcache,
                attn_mask=model_setup.attn_mask_torch,
            )
            with jax.disable_jit():
                jax_h_attn, jax_kvcache, jax_scores = jax_attention(
                    x=jax_norm_x,
                    layer_weights=jax_layer,
                    model_params=model_setup.model_params,
                    cur_pos=model_setup.cur_pos,
                    layer_idx=idx,
                    freqs_cis=model_setup.freqs_cis_jax[: model_setup.seqlen],
                    kvcache=jax_kvcache,
                    attn_mask=model_setup.attn_mask_jax,
                )
            compare_outputs(torch_h_attn, jax_h_attn)
            compare_outputs(torch_scores, jax_scores)
            compare_outputs(torch_h, jax_h)

            torch_h = torch_h + torch_h_attn
            jax_h = jax_h + jax_h_attn
            compare_outputs(torch_h, jax_h)
            torch_ff_norm_out = torch_rms_norm(x=torch_h, w=torch_layer.ffn_norm)
            jax_ff_norm_out = jax_rms_norm(x=jax_h, w=jax_layer.ffn_norm)
            compare_outputs(torch_ff_norm_out, jax_ff_norm_out)

            torch_ffn_out = torch_feed_forward(torch_ff_norm_out, torch_layer)
            jax_ffn_out = jax_feed_forward(jax_ff_norm_out, jax_layer)
            compare_outputs(torch_ffn_out, jax_ffn_out)

            torch_h = torch_h + torch_ffn_out
            jax_h = jax_h + jax_ffn_out
            compare_outputs(torch_h, jax_h)


def test_last_linear_head(model_setup: ModelTestSetup):
    with torch.no_grad():
        torch_h, jax_h = get_inputs(
            (
                model_setup.bsz,
                model_setup.seqlen,
                model_setup.model_params.n_local_heads * model_setup.model_params.head_dim,
            ),
            torch.float32,
            jnp.float32,
        )
        torch_norm_x = torch_rms_norm(x=torch_h, w=model_setup.torch_weights.norm)
        jax_norm_x = jax_rms_norm(x=jax_h, w=model_setup.jax_weights.norm)
        compare_outputs(torch_norm_x, jax_norm_x)
        torch_out = F.linear(torch_norm_x, model_setup.torch_weights.output)
        jax_out = jnp.dot(jax_norm_x, model_setup.jax_weights.output.T)
        compare_outputs(torch_out, jax_out)


if __name__ == "__main__":
    pytest.main([__file__])

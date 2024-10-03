import os
import tyro
import torch
import ml_dtypes
import jax.numpy as jnp
from pathlib import Path

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

def translate_key(in_key: str):
    out_key = in_key.replace('.weight', '')
    if out_key.startswith('model.'):
        out_key = out_key.replace('model.', '')
        if out_key.endswith('input_layernorm'):
            out_key = out_key.replace('input_layernorm', 'attention_norm')
        elif out_key.endswith('mlp.down_proj'):
            out_key = out_key.replace('mlp.down_proj', 'feed_forward.w2')
        elif out_key.endswith('mlp.gate_proj'):
            out_key = out_key.replace('mlp.gate_proj', 'feed_forward.w1')
        elif out_key.endswith('mlp.up_proj'):
            out_key = out_key.replace('mlp.up_proj', 'feed_forward.w3')
        elif out_key.endswith('post_attention_layernorm'):
            out_key = out_key.replace('post_attention_layernorm', 'ffn_norm')
        elif out_key.endswith('self_attn.k_proj'):
            out_key = out_key.replace('self_attn.k_proj', 'attention.wk')
        elif out_key.endswith('self_attn.o_proj'):
            out_key = out_key.replace('self_attn.o_proj', 'attention.wo')
        elif out_key.endswith('self_attn.q_proj'):
            out_key = out_key.replace('self_attn.q_proj', 'attention.wq')
        elif out_key.endswith('self_attn.v_proj'):
            out_key = out_key.replace('self_attn.v_proj', 'attention.wv')
        elif out_key.endswith('down_proj'):
            out_key = out_key.replace('down_proj', 'w2')
        elif out_key.endswith('gate_proj'):
            out_key = out_key.replace('gate_proj', 'w1')
        elif out_key.endswith('up_proj'):
            out_key = out_key.replace('up_proj', 'w3')
        elif out_key == 'embed_tokens':
            out_key = 'tok_embeddings'
        elif out_key == 'norm':
            out_key = 'norm'
        else:
            print(f"Don't know how to handle {in_key=}")
    elif out_key == 'lm_head':
        out_key = 'output'
    else:
        print(f"Don't know how to handle {in_key=}")
    return f'{out_key}.weight'


def reverse_permute(tensor: torch.Tensor, n_heads: int = 32, dim1:int = 4096, dim2: int = 4096) -> torch.Tensor:
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)



def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    if not str(filename).endswith("/modeling_deepseek.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    imports.remove("flash_attn")
    return imports


def main(model_id: str, out_dir: Path):
    t_path = Path.home() / '.hf_token'
    token = t_path.read_text().strip()
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)


    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
      token = t_path.read_text().strip()
      hf_model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16, offload_folder="/tmp/offload", token=token)
      with torch.no_grad():
        state_dict = hf_model.state_dict()
        for hf_name, param in state_dict.items():
            print(f' {hf_name}: {param.shape=}')
            name = translate_key(hf_name)
            if name.endswith('wq.weight'):
                param = reverse_permute(param, n_heads=32, dim1=2048, dim2=2048)  # 1B
                #param = reverse_permute(param, n_heads=24, dim1=3072, dim2=3072)  # 3B
                #param = reverse_permute(param, n_heads=32, dim1=4096, dim2=4096)  # 7B
                #param = reverse_permute(param, n_heads=64, dim1=8192, dim2=8192)   # 70B
                #param = reverse_permute(param, n_heads=96, dim1=12288, dim2=12288)   # 123B
                #param = reverse_permute(param, n_heads=128, dim1=16384, dim2=16384) # 405B
                #param = reverse_permute(param, n_heads=128, dim1=12288, dim2=12288) # DSV2
                #param = reverse_permute(param, n_heads=64, dim1=12288, dim2=12288) # Commandr+
                #param = reverse_permute(param, n_heads=48, dim1=6144, dim2=6144)    # Mixtral8x22B
            elif name.endswith('wk.weight'): #wk.weight
                param = reverse_permute(param, n_heads=8, dim1=512, dim2=2048)  # 1B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=3072)  # 3B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=4096)  # 7B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=8192)   # 70B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=12288)   # 123B
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=16384)  # 405B
                #param = reverse_permute(param, n_heads=128, dim1=12288, dim2=12288)  # DSV2
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=12288)  # Commandr+
                #param = reverse_permute(param, n_heads=8, dim1=1024, dim2=6144)    # Mixtral8x22B
            else:
                pass
            bf16_np_out = param.cpu().view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
            bf16_out = jnp.asarray(bf16_np_out, dtype=jnp.bfloat16).reshape(*param.shape)
            print(f'Writing {hf_name} as {name} to {out_dir}/{name}.npy')
            jnp.save(f'{out_dir}/{name}.npy', bf16_out)


if __name__ == "__main__":
    tyro.cli(main)
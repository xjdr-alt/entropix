import os
import tyro
import torch
import ml_dtypes
import jax.numpy as jnp
from pathlib import Path

from transformers import AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports


def translate_key(in_key):
    out_key = in_key.removeprefix("model.").removesuffix(".weight")

    match out_key.split("."):
        case [key] if key in {"lm_head", "embed_tokens", "norm"}:
            out_key = out_key.replace("embed_tokens", "tok_embeddings")
            out_key = out_key.replace("lm_head", "output")

        case [*_, key] if key.endswith("layernorm"):
            out_key = out_key.replace("layernorm", "norm")
            out_key = out_key.replace("input", "attention")
            out_key = out_key.replace("post_attention", "ffn")

        case [*_, "self_attn", qkvo_proj]:
            qkvo = qkvo_proj[0]
            out_key = out_key.replace("self_attn", "attention")
            out_key = out_key.replace(qkvo_proj, f"w{qkvo}")

        case [*_, proj] if proj.endswith("proj"):
            kind = proj[:-5]
            n = {"gate": 1, "down": 2, "up": 3}[kind]
            out_key = out_key.replace(proj, f"w{n}")
            out_key = out_key.replace("mlp", "feed_forward")

        case _:
            print(f"Don't know how to handle {in_key=}")

    return f"{out_key}.weight"


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
    t_paths_candidates = [
        Path.home() / '.hf_token',
        Path.home() / '.cache' / 'huggingface' / 'token'
    ]
    token = None
    for t_path in t_paths_candidates:
        if t_path.exists():
            token = t_path.read_text().strip()
            break
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

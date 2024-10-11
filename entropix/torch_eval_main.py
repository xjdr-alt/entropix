from typing import List, Tuple

from pathlib import Path

import torch
import torch.nn.functional as F
import tyro
from tqdm import tqdm

from lm_eval import simple_evaluate
from lm_eval.evaluator_utils import TaskOutput
from lm_eval.api.model import LM

from entropix.config import LLAMA_1B_PARAMS
from entropix.torch_kvcache import KVCache
from entropix.torch_model import xfmr
from entropix.torch_sampler import SamplerConfig, sample
from entropix.tokenizer import Tokenizer
from entropix.torch_weights import load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Device selection, tree is like first apple silicion, then cuda, fallback is cpu.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    SCALE_FACTOR = 8.0
    LOW_FREQ_FACTOR = 1.0
    HIGH_FREQ_FACTOR = 4.0
    OLD_CONTEXT_LEN = 8192  # original llama3 length

    low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
    high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

    def scale_freq(freq: torch.Tensor) -> torch.Tensor:
        wavelen = 2 * torch.pi / freq

        # Calculate smooth factor
        smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
        smooth = torch.clamp(smooth, 0.0, 1.0)  # Ensure smooth is between 0 and 1

        # Calculate scaled frequency
        scaled = (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

        # Apply conditional scaling
        scaled = torch.where(
            wavelen < high_freq_wavelen,
            freq,  # No scaling
            torch.where(
                wavelen > low_freq_wavelen,
                freq / SCALE_FACTOR,  # Apply scaling factor
                scaled  # Apply smooth scaling
            )
        )
        return scaled

    scaled_freqs = torch.vmap(scale_freq)(freqs)
    
    return scaled_freqs

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    if use_scaled:
        freqs = apply_scaling(freqs)

    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)  # Shape: (end, 1)
    freqs = freqs.unsqueeze(0)  # Shape: (1, dim//2)
    freqs = t * freqs  # Broadcasting to shape: (end, dim//2)
    return torch.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> torch.Tensor:
  mask = None
  if seqlen > 1:
      mask = torch.full((seqlen, seqlen), float("-inf"))
      mask = torch.triu(mask, diagonal=1)
      mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
  return mask

class CustomLLaMAModel(LM):
    def __init__(self, weights_path: Path):
        super().__init__()
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = load_weights(weights_path.absolute())
        self.tokenizer = Tokenizer('entropix/tokenizer.model')
        self.freqs_cis = precompute_freqs_cis(
            self.model_params.head_dim,
            self.model_params.max_seq_len,
            self.model_params.rope_theta,
            self.model_params.use_scaled_rope
        )
        self.sampler_cfg = SamplerConfig()

    def _model_generate(self, context_tokens: List[int], max_tokens: int):
        gen_tokens = None
        cur_pos = 0
        tokens = torch.tensor([context_tokens], dtype=torch.long).to(device)
        bsz, seqlen = tokens.shape
        attn_mask = build_attn_mask(seqlen, cur_pos)
        kvcache = KVCache.new(self.model_params.n_layers, bsz, self.model_params.max_seq_len, self.model_params.n_local_kv_heads, self.model_params.head_dim)

        logits, kvcache, _, _ = xfmr(self.xfmr_weights, self.model_params, tokens, cur_pos, self.freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)
        next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
        gen_tokens = next_token

        cur_pos = seqlen
        stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)

        while cur_pos < min(self.model_params.max_seq_len, len(context_tokens) + max_tokens):
            cur_pos += 1
            logits, kvcache, scores, _ = xfmr(self.xfmr_weights, self.model_params, next_token, cur_pos, self.freqs_cis[cur_pos:cur_pos+1], kvcache)
            next_token, _ = sample(gen_tokens, logits, scores, cfg=self.sampler_cfg)
            gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
            if torch.isin(next_token, stop).any():
                break

        return gen_tokens, logits

    def generate_until(self, requests) -> List[str]:
        res = []
        for request in tqdm(requests):
            context = request.args[0]
            until = request.args[1]
            context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
            generated_tokens, _ = self._model_generate(context_tokens, self.max_gen_toks)
            for t in generated_tokens:
                decoded = self.tokenizer.decode(t.tolist())
                for stop_seq in until:
                    if stop_seq in decoded:
                        stop_index = decoded.index(stop_seq)
                        decoded = decoded[:stop_index]
                        break

                res.append(decoded)

        return res

    def loglikelihood(self, requests):
      res = []
      for request in requests:
        context = request.args[0]
        continuation = request.args[1]
        context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
        continuation_tokens = self.tokenizer.encode(continuation, bos=False, eos=False, allowed_special='all')
        all_tokens = context_tokens + continuation_tokens
        tokens, logits = self._model_generate(context_tokens, len(continuation_tokens))
        continuation_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        greedy_tokens = torch.argmax(continuation_logprobs, axis=-1)
        max_equal = torch.all(greedy_tokens == torch.tensor(continuation_tokens))
        selected_logprobs = continuation_logprobs[torch.arange(len(continuation_tokens)), torch.tensor(continuation_tokens)]
        logprob_sum = torch.sum(selected_logprobs)

        res.append((float(logprob_sum), bool(max_equal)))
      return res


    def loglikelihood_rolling(self, requests):
        res = []
        for request in requests:
            context = request.args[0]
            continuation = request.args[1]

            context_tokens = self.tokenizer.encode(context, bos=False, eos=False, allowed_special='all')
            continuation_tokens = self.tokenizer.encode(continuation, bos=False, eos=False, allowed_special='all')
            all_tokens = context_tokens + continuation_tokens

            tokens, logits = self._model_generate(context_tokens, len(continuation_tokens))

            token_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            selected_logprobs = token_logprobs[torch.arange(len(continuation_tokens)), torch.tensor(continuation_tokens)]

            res.append(selected_logprobs.tolist())

        return res

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id()

    @property
    def max_length(self) -> int:
        return self.model_params.max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return 12  # The original implementation doesn't use batching

    @property
    def device(self) -> str:
        return 'cuda'  # Adjust if using GPU/TPU

def main(
    weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct'),
    tasks: List[str] = ["gpqa"],
    num_fewshot: int = 5,
):
    model = CustomLLaMAModel(weights_path)

    results = simple_evaluate(
        model=model,
        tasks=tasks,
    )

    # Create a TaskOutput object
    output = TaskOutput(results)

    # Print the results in a formatted way
    print(output.formatted())

    # If you want to save the results to a file/
    output.save_json("leaderboard.json")

if __name__ == '__main__':
    tyro.cli(main)
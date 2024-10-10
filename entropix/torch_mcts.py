import torch
import torch.nn.functional as F
from typing import Tuple

from entropix.torch_main import calculate_varentropy_logsoftmax, _sample

class MCTSSearch:
    def __init__(self, cxfmr, xfmr_weights, model_params, freqs_cis, kvcache):
        self.cxfmr = cxfmr
        self.xfmr_weights = xfmr_weights
        self.model_params = model_params
        self.freqs_cis = freqs_cis
        self.kvcache = kvcache
        self.max_depth = 6
        self.n_branches = 5

    def _is_normal_range(self, ent: float, vent: float) -> bool:
        return ent < 5.0 and vent < 5.0

    def simulate_path(self, token: torch.Tensor, cur_pos: int, depth: int = 0) -> Tuple[torch.Tensor, bool]:
        if depth >= self.max_depth:
            return token, False

        next_logits, _ = self.cxfmr(self.xfmr_weights, self.model_params, token.unsqueeze(0), 
                                    cur_pos + depth + 1, 
                                    self.freqs_cis[cur_pos + depth + 1:cur_pos + depth + 2], 
                                    self.kvcache)
        next_ent, next_vent = calculate_varentropy_logsoftmax(next_logits)
        
        if self._is_normal_range(next_ent.item(), next_vent.item()):
            return token, True

        next_token = _sample(next_logits, temperature=1.0)
        return self.simulate_path(next_token.squeeze(), cur_pos, depth + 1)

    def search(self, logits: torch.Tensor, cur_pos: int) -> torch.Tensor:
        # Select initial candidates
        candidates = []
        for _ in range(self.n_branches):
            candidates.append(_sample(logits, temperature=2))
        
        for candidate in candidates:
            # Remove extra dimensions to get a 1D tensor
            candidate_token = candidate.squeeze()
            final_token, success = self.simulate_path(candidate_token, cur_pos)
            if success:
                return final_token.unsqueeze(0).unsqueeze(0)

        # If no path leads to normal range, return the first candidate
        return candidates[0]
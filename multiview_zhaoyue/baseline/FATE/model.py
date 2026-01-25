from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
@dataclass
class FateConfig:
    hidden_size: int
    attention_size: int = 150
    num_heads: int = 5
    topk_ratio: float = 0.1
class FateAttentionScorer(nn.Module):
    def __init__(self, cfg: FateConfig):
        super().__init__()
        self.cfg = cfg
        self.W1 = nn.Linear(cfg.hidden_size, cfg.attention_size, bias=False)
        self.W2 = nn.Linear(cfg.attention_size, cfg.num_heads, bias=False)
    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.tanh(self.W1(hidden_state))
        t = self.W2(t)
        mask = attention_mask.unsqueeze(-1).to(dtype=t.dtype)
        t = t.masked_fill(mask == 0, float("-inf"))
        t = F.softmax(t, dim=1)
        A = t.transpose(1, 2)
        outputs = A @ hidden_state
        outputs = torch.flatten(outputs, start_dim=1)
        k = float(self.cfg.topk_ratio)
        topk = max(int(outputs.size(1) * k), 1)
        outputs = torch.topk(torch.abs(outputs), topk, dim=1)[0]
        score = outputs.mean(dim=1).float()
        return score, A
class FateModel(nn.Module):
    def __init__(self, backbone: nn.Module, cfg: FateConfig):
        super().__init__()
        self.backbone = backbone
        self.scorer = FateAttentionScorer(cfg)
    def forward(self, features: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.backbone(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            return_dict=True,
        )
        hidden_state = out.last_hidden_state
        score, A = self.scorer(hidden_state, features["attention_mask"])
        return score, A

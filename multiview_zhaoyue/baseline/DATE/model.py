from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraModel
def _get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = (name or "gelu").lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    return F.gelu
def generate_pseudo_masks(
    seq_len: int = 128,
    n_masks: int = 50,
    perc_of_els: float = 0.5,
    seed: int = 42,
) -> List[List[bool]]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    masks: List[List[bool]] = []
    target_true = int(seq_len * perc_of_els)
    def _mask_key(m: List[bool]) -> Tuple[int, ...]:
        return tuple(1 if x else 0 for x in m)
    seen = set()
    while len(masks) < n_masks:
        idx = torch.randperm(seq_len, generator=rng)[:target_true]
        m = [False for _ in range(seq_len)]
        for i in idx.tolist():
            m[i] = True
        key = _mask_key(m)
        if key in seen:
            continue
        seen.add(key)
        masks.append(m)
    return masks
class DateDiscriminatorPredictions(nn.Module):
    def __init__(self, config: ElectraConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.act = _get_activation(getattr(config, "hidden_act", "gelu"))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.dense(hidden_states)
        x = self.act(x)
        logits = self.dense_prediction(x).squeeze(-1)
        return logits
class DateRMDHead(nn.Module):
    def __init__(self, hidden_size: int, n_masks: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_masks)
    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        cls_vec = sequence_output[:, 0, :]
        x = F.relu(self.fc1(cls_vec))
        return self.fc2(x)
@dataclass
class DateConfig:
    seq_len: int = 128
    n_masks: int = 50
    vocab_size: int = 30522
    gen_hidden_size: int = 16
    gen_num_layers: int = 1
    gen_num_heads: int = 2
    disc_hidden_size: int = 256
    disc_num_layers: int = 4
    disc_num_heads: int = 4
    hidden_act: str = "gelu"
    embedding_size: int = 128
    dropout: float = 0.5
class DateModel(nn.Module):
    def __init__(
        self,
        cfg: DateConfig,
        random_generator: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.cfg = cfg
        self.random_generator = random_generator
        gen_config = ElectraConfig(
            vocab_size=cfg.vocab_size,
            embedding_size=cfg.embedding_size,
            hidden_size=cfg.gen_hidden_size,
            num_hidden_layers=cfg.gen_num_layers,
            num_attention_heads=cfg.gen_num_heads,
            intermediate_size=max(4 * cfg.gen_hidden_size, 64),
            hidden_act=cfg.hidden_act,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.generator = ElectraForMaskedLM(gen_config)
        disc_config = ElectraConfig(
            vocab_size=cfg.vocab_size,
            embedding_size=cfg.embedding_size,
            hidden_size=cfg.disc_hidden_size,
            num_hidden_layers=cfg.disc_num_layers,
            num_attention_heads=cfg.disc_num_heads,
            intermediate_size=max(4 * cfg.disc_hidden_size, 256),
            hidden_act=cfg.hidden_act,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.discriminator_backbone = ElectraModel(disc_config)
        self.rtd_head = DateDiscriminatorPredictions(disc_config)
        self.rmd_head = DateRMDHead(disc_config.hidden_size, cfg.n_masks)
        self._rng = torch.Generator()
        self._rng.manual_seed(seed)
        self.masks = generate_pseudo_masks(seq_len=cfg.seq_len, n_masks=cfg.n_masks, seed=seed)
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.Tensor] = None,
        rmd_labels: Optional[torch.Tensor] = None,
        replace_tokens: bool = True,
    ):
        g_out = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=mlm_labels,
        )
        d_input_ids = input_ids
        d_labels = None
        if replace_tokens and mlm_labels is not None:
            with torch.no_grad():
                logits = g_out.logits
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(input_ids.size(0), -1)
                if self.random_generator:
                    sampled = torch.randint(
                        low=5,
                        high=self.cfg.vocab_size - 1,
                        size=sampled.shape,
                        device=sampled.device,
                    )
                mask = mlm_labels.ne(-100)
                d_input_ids = input_ids.clone()
                d_input_ids[mask] = sampled[mask]
                correct = sampled.eq(mlm_labels) & mask
                d_labels = mask.long()
                d_labels[correct] = 0
        else:
            d_input_ids = input_ids
            d_labels = torch.zeros_like(input_ids)
        disc_outputs = self.discriminator_backbone(
            input_ids=d_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = disc_outputs.last_hidden_state
        rtd_logits = self.rtd_head(sequence_output)
        rmd_logits = self.rmd_head(sequence_output)
        rtd_loss = None
        if d_labels is not None:
            active = attention_mask.bool()
            rtd_loss = F.binary_cross_entropy_with_logits(rtd_logits[active], d_labels.float()[active])
        rmd_loss = None
        if rmd_labels is not None:
            rmd_loss = F.cross_entropy(rmd_logits, rmd_labels)
        return {
            "g_loss": g_out.loss,
            "rtd_loss": rtd_loss,
            "rmd_loss": rmd_loss,
            "rtd_logits": rtd_logits,
            "rmd_logits": rmd_logits,
        }
@torch.no_grad()
def compute_rtd_anomaly_score(
    model: DateModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    model.eval()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        mlm_labels=None,
        rmd_labels=None,
        replace_tokens=False,
    )
    rtd_logits = outputs["rtd_logits"]
    rtd_prob = torch.sigmoid(rtd_logits)
    active = attention_mask.bool()
    if rtd_prob.dim() == 1:
        rtd_prob = rtd_prob.unsqueeze(0)
    scores = []
    for i in range(rtd_prob.size(0)):
        m = active[i]
        if m.sum().item() == 0:
            scores.append(torch.tensor(0.0, device=rtd_prob.device))
        else:
            scores.append(rtd_prob[i][m].mean())
    return torch.stack(scores, dim=0)

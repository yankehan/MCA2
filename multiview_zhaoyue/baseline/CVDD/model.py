import torch
import torch.nn as nn
import torch.nn.functional as F
class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int = 100, n_attention_heads: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_attention_heads, bias=False)
    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden = hidden.transpose(0, 1)
        x = torch.tanh(self.W1(hidden))
        logits = self.W2(x)
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be a 2D tensor")
            b, s = hidden.shape[0], hidden.shape[1]
            if attention_mask.shape == (b, s):
                mask_bt = attention_mask
            elif attention_mask.shape == (s, b):
                mask_bt = attention_mask.transpose(0, 1)
            else:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match (batch, seq)=({b}, {s})"
                )
            mask_bt = mask_bt.to(dtype=torch.bool)
            logits = logits.masked_fill(~mask_bt.unsqueeze(-1), float("-inf"))
        x = F.softmax(logits, dim=1)
        A = x.transpose(1, 2)
        M = A @ hidden
        return M, A
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int, freeze: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embedding_size = embedding_size
        if freeze:
            for p in self.embedding.parameters():
                p.requires_grad = False
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        return self.embedding(x)
class FrozenBertEncoder(nn.Module):
    def __init__(self, pretrained_model_name: str = "bert-base-uncased", cache_dir: str = None):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(pretrained_model_name, cache_dir=cache_dir)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.embedding_size = int(self.bert.config.hidden_size)
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        self.bert.eval()
        x_bt = x.transpose(0, 1)
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must be a 2D tensor")
            if attention_mask.shape == x.shape:
                attention_mask = attention_mask.transpose(0, 1)
            elif attention_mask.shape == x_bt.shape:
                pass
            else:
                raise ValueError(
                    f"attention_mask shape {tuple(attention_mask.shape)} does not match input_ids shapes "
                    f"{tuple(x_bt.shape)} (batch, seq) or {tuple(x.shape)} (seq, batch)"
                )
        with torch.no_grad():
            out = self.bert(input_ids=x_bt, attention_mask=attention_mask)
            hidden = out.last_hidden_state
        return hidden.transpose(0, 1)
class CVDDNet(nn.Module):
    def __init__(self, pretrained_model: TokenEmbedding, attention_size: int = 100, n_attention_heads: int = 1):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.embedding_size
        self.attention_size = attention_size
        self.n_attention_heads = n_attention_heads
        self.self_attention = SelfAttention(
            hidden_size=self.hidden_size,
            attention_size=attention_size,
            n_attention_heads=n_attention_heads,
        )
        self.c = nn.Parameter((torch.rand(1, n_attention_heads, self.hidden_size) - 0.5) * 2)
        self.cosine_sim = nn.CosineSimilarity(dim=2)
        self.alpha = 0.0
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None):
        hidden = self.pretrained_model(x, attention_mask=attention_mask)
        M, A = self.self_attention(hidden, attention_mask=attention_mask)
        cosine_dists = 0.5 * (1 - self.cosine_sim(M, self.c))
        context_weights = F.softmax(-self.alpha * cosine_dists, dim=1)
        return cosine_dists, context_weights, A

import torch
import torch.nn as nn
class DeviationLoss(nn.Module):
    def __init__(self, confidence_margin: float = 5.0, ref_size: int = 5000):
        super().__init__()
        self.confidence_margin = float(confidence_margin)
        self.ref_size = int(ref_size)
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.dtype != torch.float32:
            y_true = y_true.float()
        ref = torch.normal(
            mean=0.0,
            std=torch.ones(self.ref_size, device=y_pred.device),
        )
        dev = (y_pred - torch.mean(ref)) / (torch.std(ref) + 1e-8)
        inlier_loss = torch.abs(dev)
        outlier_loss = torch.abs((self.confidence_margin - dev).clamp_(min=0.0))
        return torch.mean((1.0 - y_true) * inlier_loss + y_true * outlier_loss)

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from .model import CVDDNet
class CVDDTrainer:
    def __init__(
        self,
        optimizer_name: str = "adam",
        lr: float = 1e-3,
        n_epochs: int = 50,
        lr_milestones: Tuple[int, ...] = (),
        batch_size: int = 64,
        lambda_p: float = 1.0,
        alpha_scheduler: str = "logarithmic",
        weight_decay: float = 0.5e-6,
        device: str = "cuda",
        n_jobs_dataloader: int = 0,
        show_progress: bool = True,
        desc_prefix: str = "",
    ):
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.lambda_p = lambda_p
        self.alpha_scheduler = alpha_scheduler
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.show_progress = show_progress
        self.desc_prefix = desc_prefix
        self.alpha_milestones = np.arange(1, 6) * int(n_epochs / 5) if n_epochs >= 5 else np.array([])
        if alpha_scheduler == "soft":
            self.alphas = [0.0] * 5
        elif alpha_scheduler == "linear":
            self.alphas = np.linspace(0.2, 1.0, 5).tolist()
        elif alpha_scheduler == "logarithmic":
            self.alphas = np.logspace(-4, 0, 5).tolist()
        else:
            self.alphas = [100.0] * 4
        self.train_time: Optional[float] = None
        self.test_time: Optional[float] = None
        self.c: Optional[List] = None
        self.train_dists = None
        self.test_dists = None
        self.test_auc: float = 0.0
        self.test_auprc: float = 0.0
        self.test_scores = None
        self.test_att_weights = None
    def train(self, dataset, net: CVDDNet):
        net = net.to(self.device)
        n_attention_heads = net.n_attention_heads
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        net.c.data = torch.from_numpy(
            initialize_context_vectors(net, train_loader, self.device)[np.newaxis, :]
        ).to(self.device)
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
        start_time = time.time()
        net.train()
        alpha_i = 0
        for epoch in range(self.n_epochs):
            scheduler.step()
            if epoch in set(self.alpha_milestones.tolist()):
                if alpha_i < len(self.alphas):
                    net.alpha = float(self.alphas[alpha_i])
                    alpha_i += 1
            epoch_loss = 0.0
            n_batches = 0
            dists_per_head = ()
            it = train_loader
            if self.show_progress:
                it = tqdm(train_loader, desc=f"{self.desc_prefix}train epoch {epoch+1}/{self.n_epochs}", leave=False)
            for data in it:
                if len(data) == 5:
                    _, text_batch, _, _, attention_mask = data
                    attention_mask = attention_mask.to(self.device)
                else:
                    _, text_batch, _, _ = data
                    attention_mask = None
                text_batch = text_batch.to(self.device)
                optimizer.zero_grad()
                cosine_dists, context_weights, _ = net(text_batch, attention_mask=attention_mask)
                scores = context_weights * cosine_dists
                I = torch.eye(n_attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P
                dists_per_head += (cosine_dists.detach().cpu().numpy(),)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()
                epoch_loss += float(loss.item())
                n_batches += 1
            if n_batches > 0:
                self.train_dists = np.concatenate(dists_per_head)
        self.train_time = time.time() - start_time
        self.c = np.squeeze(net.c.detach().cpu().numpy()).tolist()
        return net
    def test(self, dataset, net: CVDDNet, ad_score: str = "context_dist_mean"):
        net = net.to(self.device)
        n_attention_heads = net.n_attention_heads
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        idx_label_score_head = []
        dists_per_head = ()
        att_weights = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            it = test_loader
            if self.show_progress:
                it = tqdm(test_loader, desc=f"{self.desc_prefix}test", leave=False)
            for data in it:
                if len(data) == 5:
                    idx, text_batch, label_batch, _, attention_mask = data
                    attention_mask = attention_mask.to(self.device)
                else:
                    idx, text_batch, label_batch, _ = data
                    attention_mask = None
                text_batch = text_batch.to(self.device)
                label_batch = label_batch.to(self.device)
                cosine_dists, _, A = net(text_batch, attention_mask=attention_mask)
                scores = cosine_dists
                _, best_att_head = torch.min(scores, dim=1)
                dists_per_head += (cosine_dists.detach().cpu().numpy(),)
                ad_scores = torch.mean(cosine_dists, dim=1)
                idx_label_score_head += list(
                    zip(
                        idx,
                        label_batch.detach().cpu().numpy().tolist(),
                        ad_scores.detach().cpu().numpy().tolist(),
                        best_att_head.detach().cpu().numpy().tolist(),
                    )
                )
                att_weights += A[range(len(idx)), best_att_head].detach().cpu().numpy().tolist()
        self.test_time = time.time() - start_time
        self.test_dists = np.concatenate(dists_per_head) if len(dists_per_head) else None
        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights
        if len(idx_label_score_head) > 0:
            _, labels, scores, _ = zip(*idx_label_score_head)
            labels = np.array(labels)
            scores = np.array(scores)
            if np.sum(labels) > 0:
                if ad_score == "context_dist_mean":
                    self.test_auc = float(roc_auc_score(labels, scores))
                    self.test_auprc = float(average_precision_score(labels, scores))
                elif ad_score == "context_best" and self.test_dists is not None:
                    best = 0.0
                    best_auprc = 0.0
                    for context in range(n_attention_heads):
                        auc_candidate = float(roc_auc_score(labels, self.test_dists[:, context]))
                        auprc_candidate = float(average_precision_score(labels, self.test_dists[:, context]))
                        best = max(best, auc_candidate)
                        best_auprc = max(best_auprc, auprc_candidate)
                    self.test_auc = best
                    self.test_auprc = best_auprc
                else:
                    self.test_auc = 0.0
                    self.test_auprc = 0.0
            else:
                self.test_auc = 0.0
                self.test_auprc = 0.0
        else:
            self.test_auc = 0.0
            self.test_auprc = 0.0
def initialize_context_vectors(net: CVDDNet, train_loader, device: str):
    X = ()
    for data in train_loader:
        if len(data) == 5:
            _, text, _, _, attention_mask = data
            attention_mask = attention_mask.to(device)
        else:
            _, text, _, _ = data
            attention_mask = None
        text = text.to(device)
        X_batch = net.pretrained_model(text, attention_mask=attention_mask)
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        X_batch[torch.isnan(X_batch)] = 0
        X += (X_batch.detach().cpu().numpy(),)
    X = np.concatenate(X) if len(X) else np.zeros((1, net.hidden_size), dtype=np.float32)
    kmeans = KMeans(n_clusters=net.n_attention_heads).fit(X)
    centers = kmeans.cluster_centers_
    centers = centers / np.linalg.norm(centers, ord=2, axis=1, keepdims=True)
    return centers

from __future__ import annotations
import os
import time
from typing import Tuple
import numpy as np
import torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
@torch.no_grad()
def _predict_scores(model, dataloader, device: torch.device, use_amp: bool = True):
    model.eval()
    scores = []
    labels = []
    amp_ctx = torch.cuda.amp.autocast(enabled=bool(use_amp and device.type == "cuda"))
    for input_ids, attention_mask, y in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with amp_ctx:
            out, _ = model({"input_ids": input_ids, "attention_mask": attention_mask})
        scores.append(out.detach().float().cpu())
        labels.append(y.detach().float().cpu())
    return torch.cat(scores, dim=0).numpy(), torch.cat(labels, dim=0).numpy()
def run_fate_one_dataset(
    dataset_name: str,
    data_root: str,
    out_dir: str,
    device: str = "cuda",
    seed: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_len: int = 128,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    few_shot_anomalies: int = 10,
    attention_size: int = 150,
    num_heads: int = 5,
    topk_ratio: float = 0.1,
    include_regularization: bool = True,
    num_workers: int = 0,
    suppress_internal_output: bool = True,
) -> Tuple[float, float, float]:
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score, average_precision_score
    from transformers import AutoModel, AutoTokenizer
    from .dataset import build_fate_dataloaders, build_fate_tensors_from_jsonl
    from .loss import DeviationLoss
    from .model import FateConfig, FateModel
    del out_dir
    set_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    pin_memory = torch_device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    cfg = FateConfig(
        hidden_size=int(backbone.config.hidden_size),
        attention_size=int(attention_size),
        num_heads=int(num_heads),
        topk_ratio=float(topk_ratio),
    )
    model = FateModel(backbone=backbone, cfg=cfg).to(torch_device)
    tensors = build_fate_tensors_from_jsonl(
        dataset_name=dataset_name,
        data_root=data_root,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        few_shot_anomalies=few_shot_anomalies,
        seed=seed,
    )
    train_loader, test_loader = build_fate_dataloaders(
        tensors=tensors,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = DeviationLoss()
    use_amp = torch_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    head_eye = torch.eye(int(num_heads), device=torch_device)
    t0 = time.time()
    model.train()
    for _epoch in tqdm(range(int(num_epochs)), desc=f"[{dataset_name}] epochs", leave=False, mininterval=0.5):
        for input_ids, attention_mask, y in train_loader:
            input_ids = input_ids.to(torch_device, non_blocking=True)
            attention_mask = attention_mask.to(torch_device, non_blocking=True)
            y = y.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out, A = model({"input_ids": input_ids, "attention_mask": attention_mask})
                loss_main = criterion(out, y)
                if include_regularization:
                    CCT = A @ A.transpose(1, 2)
                    loss_reg = torch.mean((CCT - head_eye) ** 2)
                    loss = loss_main + loss_reg
                else:
                    loss = loss_main
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    scores, labels = _predict_scores(model, test_loader, device=torch_device, use_amp=use_amp)
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    t1 = time.time()
    total_time = float(t1 - t0)
    if not suppress_internal_output:
        print(f"[FATE] {dataset_name} AUROC={auroc:.6f} AUPRC={auprc:.6f} total_run_time_sec={total_time:.2f}")
    return auroc, auprc, total_time
def run_fate_one_dataset_efficiency(
    dataset_name: str,
    data_root: str,
    out_dir: str,
    device: str = "cuda",
    seed: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_len: int = 128,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    few_shot_anomalies: int = 10,
    attention_size: int = 150,
    num_heads: int = 5,
    topk_ratio: float = 0.1,
    include_regularization: bool = True,
    num_workers: int = 0,
    suppress_internal_output: bool = True,
) -> Tuple[float, float, float, float]:
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModel, AutoTokenizer
    from .dataset import build_fate_dataloaders, build_fate_tensors_from_jsonl
    from .loss import DeviationLoss
    from .model import FateConfig, FateModel
    del out_dir
    total_t0 = time.time()
    set_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    pin_memory = torch_device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    cfg = FateConfig(
        hidden_size=int(backbone.config.hidden_size),
        attention_size=int(attention_size),
        num_heads=int(num_heads),
        topk_ratio=float(topk_ratio),
    )
    model = FateModel(backbone=backbone, cfg=cfg).to(torch_device)
    tensors = build_fate_tensors_from_jsonl(
        dataset_name=dataset_name,
        data_root=data_root,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        few_shot_anomalies=few_shot_anomalies,
        seed=seed,
    )
    train_loader, test_loader = build_fate_dataloaders(
        tensors=tensors,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = DeviationLoss()
    use_amp = torch_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    head_eye = torch.eye(int(num_heads), device=torch_device)
    train_t0 = total_t0
    model.train()
    for _epoch in tqdm(range(int(num_epochs)), desc=f"[{dataset_name}] epochs", leave=False, mininterval=0.5):
        for input_ids, attention_mask, y in train_loader:
            input_ids = input_ids.to(torch_device, non_blocking=True)
            attention_mask = attention_mask.to(torch_device, non_blocking=True)
            y = y.to(torch_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out, A = model({"input_ids": input_ids, "attention_mask": attention_mask})
                loss_main = criterion(out, y)
                if include_regularization:
                    CCT = A @ A.transpose(1, 2)
                    loss_reg = torch.mean((CCT - head_eye) ** 2)
                    loss = loss_main + loss_reg
                else:
                    loss = loss_main
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    train_t1 = time.time()
    train_time_sec = float(train_t1 - train_t0)
    test_t0 = time.time()
    scores, labels = _predict_scores(model, test_loader, device=torch_device, use_amp=use_amp)
    auroc = float(roc_auc_score(labels, scores))
    test_t1 = time.time()
    test_time_sec = float(test_t1 - test_t0)
    total_time_sec = float(train_time_sec + test_time_sec)
    if not suppress_internal_output:
        print(
            f"[FATE] {dataset_name} AUROC={auroc:.6f} train_time(s)={train_time_sec:.2f} "
            f"test_time(s)={test_time_sec:.2f} total_time(s)={total_time_sec:.2f}"
        )
    return auroc, train_time_sec, test_time_sec, total_time_sec

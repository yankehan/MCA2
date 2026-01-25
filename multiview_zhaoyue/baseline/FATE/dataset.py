from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset
def read_jsonl_texts(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj.get("text", "")))
            labels.append(int(obj.get("label", 0)))
    return texts, labels
def maybe_generate_jsonl_from_npz(
    dataset_name: str,
    data_root: str,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> None:
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if os.path.exists(train_jsonl) and os.path.exists(test_jsonl):
        return
    npz_path = os.path.join(data_root, f"{dataset_name}.npz")
    if not os.path.exists(npz_path):
        return
    data = np.load(npz_path, allow_pickle=True)
    texts = data["data"]
    labels = data["label"]
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    if len(normal_idx) == 0:
        raise ValueError(f"No normal samples (label=0) found in {npz_path}")
    if len(anomaly_idx) == 0:
        raise ValueError(f"No anomaly samples (label=1) found in {npz_path}")
    rng = np.random.RandomState(seed)
    rng.shuffle(normal_idx)
    n_train = int(len(normal_idx) * train_ratio)
    train_normal_idx = normal_idx[:n_train]
    test_normal_idx = normal_idx[n_train:]
    train_texts = [str(texts[i]) for i in train_normal_idx]
    train_labels = [0 for _ in train_normal_idx]
    test_idx = np.concatenate([test_normal_idx, anomaly_idx])
    rng.shuffle(test_idx)
    test_texts = [str(texts[i]) for i in test_idx]
    test_labels = [int(labels[i]) for i in test_idx]
    def dump_jsonl(path: str, xs: List[str], ys: List[int]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for x, y in zip(xs, ys):
                f.write(json.dumps({"text": x, "label": int(y)}, ensure_ascii=False) + "\n")
    dump_jsonl(train_jsonl, train_texts, train_labels)
    dump_jsonl(test_jsonl, test_texts, test_labels)
@dataclass
class FateDataTensors:
    train_input_ids: torch.LongTensor
    train_attention_mask: torch.LongTensor
    train_labels: torch.FloatTensor
    test_input_ids: torch.LongTensor
    test_attention_mask: torch.LongTensor
    test_labels: torch.FloatTensor
class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, labels: torch.Tensor, batch_size: int, seed: int = 42):
        super().__init__(None)
        self.labels = labels.detach().cpu().numpy().astype(int)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.normal_idx = np.where(self.labels == 0)[0]
        self.outlier_idx = np.where(self.labels == 1)[0]
        if len(self.normal_idx) == 0 or len(self.outlier_idx) == 0:
            raise ValueError("BalancedBatchSampler requires both normal and outlier samples in training")
        self.steps_per_epoch = max(1, int(len(self.labels) // self.batch_size))
        self.n_normal = self.batch_size // 2
        self.n_outlier = self.batch_size - self.n_normal
    def __len__(self):
        return self.steps_per_epoch
    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        normal_perm = rng.permutation(self.normal_idx)
        outlier_perm = rng.permutation(self.outlier_idx)
        n_ptr = 0
        o_ptr = 0
        for _ in range(self.steps_per_epoch):
            batch = []
            for _ in range(self.n_normal):
                if n_ptr >= len(normal_perm):
                    normal_perm = rng.permutation(self.normal_idx)
                    n_ptr = 0
                batch.append(int(normal_perm[n_ptr]))
                n_ptr += 1
            for _ in range(self.n_outlier):
                if o_ptr >= len(outlier_perm):
                    outlier_perm = rng.permutation(self.outlier_idx)
                    o_ptr = 0
                batch.append(int(outlier_perm[o_ptr]))
                o_ptr += 1
            rng.shuffle(batch)
            yield batch
def build_fate_tensors_from_jsonl(
    dataset_name: str,
    data_root: str,
    tokenizer,
    max_seq_len: int = 128,
    few_shot_anomalies: int = 10,
    seed: int = 42,
) -> FateDataTensors:
    maybe_generate_jsonl_from_npz(dataset_name=dataset_name, data_root=data_root, seed=seed)
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if not os.path.exists(train_jsonl) or not os.path.exists(test_jsonl):
        raise FileNotFoundError(
            f"缺少数据文件: {dataset_name}\n"
            f"需要存在: {train_jsonl} 和 {test_jsonl}\n"
            f"或者存在: {os.path.join(data_root, dataset_name + '.npz')} 以便自动生成 jsonl"
        )
    train_texts, train_labels = read_jsonl_texts(train_jsonl)
    if any(int(l) != 0 for l in train_labels):
        raise ValueError(f"训练集必须全部为正常样本(label=0)，但 {train_jsonl} 中存在异常")
    test_texts, test_labels = read_jsonl_texts(test_jsonl)
    test_normals = [t for t, y in zip(test_texts, test_labels) if int(y) == 0]
    test_anoms = [t for t, y in zip(test_texts, test_labels) if int(y) == 1]
    if len(test_anoms) == 0:
        raise ValueError(f"测试集必须包含异常样本(label=1)，但 {test_jsonl} 中没有")
    rng = np.random.RandomState(seed)
    k = min(int(few_shot_anomalies), len(test_anoms))
    anom_idx = rng.choice(len(test_anoms), size=k, replace=False)
    fewshot_anoms = [test_anoms[i] for i in anom_idx]
    remaining_anoms = [t for i, t in enumerate(test_anoms) if i not in set(anom_idx.tolist())]
    train_all_texts = list(train_texts) + fewshot_anoms
    train_all_labels = [0] * len(train_texts) + [1] * len(fewshot_anoms)
    eval_texts = list(test_normals) + list(remaining_anoms)
    eval_labels = [0] * len(test_normals) + [1] * len(remaining_anoms)
    train_enc = tokenizer(
        train_all_texts,
        max_length=int(max_seq_len),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    test_enc = tokenizer(
        eval_texts,
        max_length=int(max_seq_len),
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return FateDataTensors(
        train_input_ids=train_enc["input_ids"].long(),
        train_attention_mask=train_enc["attention_mask"].long(),
        train_labels=torch.tensor(train_all_labels, dtype=torch.float32),
        test_input_ids=test_enc["input_ids"].long(),
        test_attention_mask=test_enc["attention_mask"].long(),
        test_labels=torch.tensor(eval_labels, dtype=torch.float32),
    )
def build_fate_dataloaders(
    tensors: FateDataTensors,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    train_ds = TensorDataset(tensors.train_input_ids, tensors.train_attention_mask, tensors.train_labels)
    train_sampler = BalancedBatchSampler(tensors.train_labels, batch_size=int(train_batch_size), seed=seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    test_ds = TensorDataset(tensors.test_input_ids, tensors.test_attention_mask, tensors.test_labels)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    return train_loader, test_loader

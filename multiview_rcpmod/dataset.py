import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
class MultiViewEmbeddingDataset(Dataset):

    def __init__(self, data_name, view_names, split='train', embeddings_dir='embeddings',
                 normalize=True, device='cpu'):
        self.data_name = data_name
        self.view_names = view_names
        self.split = split
        self.embeddings_dir = embeddings_dir
        self.normalize = normalize
        self.device = device
        self.embeddings = {}
        self.norm_stats = {}
        for view_name in view_names:
            emb = self.load_embedding(view_name)
            if normalize:
                emb, stats = self._normalize(emb)
                self.norm_stats[view_name] = stats
            self.embeddings[view_name] = torch.tensor(emb, dtype=torch.float32)
        n_samples = [emb.shape[0] for emb in self.embeddings.values()]
        assert len(set(n_samples)) == 1, "All views must have same number of samples"
        self.n_samples = n_samples[0]
        self.labels = self.load_labels()
        if split == 'train':
            unique_labels = torch.unique(self.labels)
            if len(unique_labels) > 1 or (len(unique_labels) == 1 and unique_labels[0] != 0):
                print(f"警告: 训练集应该只包含正常数据(label=0)，但发现标签: {unique_labels.tolist()}")
    def load_embedding(self, view_name):
        file_name = f"{view_name}_{self.data_name}_{self.split}.npy"
        file_path = os.path.join(self.embeddings_dir, self.data_name, f"{self.data_name}-{self.split}", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        return np.load(file_path)
    def load_labels(self):
        data_path = f'data/{self.data_name}_{self.split}_data.jsonl'
        if os.path.exists(data_path):
            labels = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    labels.append(item['label'])
            labels = np.array(labels)
            unique_labels = np.unique(labels)
            print(f"✓ 加载标签成功: {data_path}")
            print(f"  标签数量: {len(labels)}, 类别: {unique_labels}, 分布: {np.bincount(labels)}")
            return torch.tensor(labels, dtype=torch.long)
        else:
            raise FileNotFoundError(
                f"❌ 数据文件不存在: {data_path}\n"
                f"   请确保数据文件在正确的位置。\n"
                f"   当前工作目录: {os.getcwd()}"
            )
    def _normalize(self, embeddings):
        min_val = np.min(embeddings, axis=0, keepdims=True)
        max_val = np.max(embeddings, axis=0, keepdims=True)
        normalized = (embeddings - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val}
        return normalized, stats
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        views_dict = {
            view_name: self.embeddings[view_name][idx]
            for view_name in self.view_names
        }
        label = self.labels[idx]
        return views_dict, label
def collate_multiview(batch):
    views_dicts, labels = zip(*batch)
    view_names = list(views_dicts[0].keys())
    batched_views = {}
    for view_name in view_names:
        embeddings = [views_dict[view_name] for views_dict in views_dicts]
        batched_views[view_name] = torch.stack(embeddings)
    batched_labels = torch.stack(labels)
    return batched_views, batched_labels
def create_multiview_dataloader(data_name, view_names, split='train', batch_size=256,
                               shuffle=True, embeddings_dir='embeddings',
                               normalize=True, device='cpu'):
    dataset = MultiViewEmbeddingDataset(
        data_name=data_name,
        view_names=view_names,
        split=split,
        embeddings_dir=embeddings_dir,
        normalize=normalize,
        device=device
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_multiview,
        num_workers=0
    )
    return dataloader, dataset
def load_all_views_full(data_name, view_names, split='train', embeddings_dir='embeddings',
                       normalize=True, device='cpu'):
    dataset = MultiViewEmbeddingDataset(
        data_name=data_name,
        view_names=view_names,
        split=split,
        embeddings_dir=embeddings_dir,
        normalize=normalize,
        device=device
    )
    views_dict = {
        view_name: dataset.embeddings[view_name].to(device)
        for view_name in view_names
    }
    labels = dataset.labels.to(device)
    norm_stats = dataset.norm_stats
    return views_dict, labels, norm_stats
def get_view_dimensions(data_name, view_names, split='train', embeddings_dir='embeddings'):
    view_dims = {}
    for view_name in view_names:
        file_name = f"{view_name}_{data_name}_{split}.npy"
        file_path = os.path.join(embeddings_dir, data_name, f"{data_name}-{split}", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        emb = np.load(file_path, mmap_mode='r')
        view_dims[view_name] = emb.shape[1]
    return view_dims

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
class NCMODDataset(Dataset):
    def __init__(self, data_name, view_names, split='train', embeddings_dir='embeddings',
                 normalize=True, device='cpu', num_neibs=8):
        self.data_name = data_name
        self.view_names = view_names
        self.split = split
        self.embeddings_dir = embeddings_dir
        self.normalize = normalize
        self.device = device
        self.num_neibs = num_neibs
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
        self.neibs_global = [[] for _ in range(self.n_samples)]
        self.neibs_local = [[] for _ in range(self.n_samples)]
        self.weights_global = [[] for _ in range(self.n_samples)]
    def load_embedding(self, view_name):
        file_name = f"{view_name}_{self.data_name}_{self.split}.npy"
        file_path = os.path.join(self.embeddings_dir, self.data_name, f"{self.data_name}-{self.split}", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        try:
            return np.load(file_path)
        except Exception as e:
            raise IOError(f"Failed to load embedding file {file_path}: {e}") from e
    def load_labels(self):
        data_path = f'data/{self.data_name}_{self.split}_data.jsonl'
        if os.path.exists(data_path):
            labels = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Malformed JSON at line {line_num} in {data_path}: {e}"
                        ) from e
                    if 'label' not in item:
                        raise KeyError(
                            f"Missing 'label' key at line {line_num} in {data_path}"
                        )
                    labels.append(item['label'])
            labels = np.array(labels)
            return torch.tensor(labels, dtype=torch.long)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    def _normalize(self, embeddings):
        min_val = np.min(embeddings, axis=0, keepdims=True)
        max_val = np.max(embeddings, axis=0, keepdims=True)
        normalized = (embeddings - min_val) / (max_val - min_val + 1e-8)
        stats = {'min': min_val, 'max': max_val}
        return normalized, stats
    def set_knn(self, neibs_global, neibs_local, weights_global):
        self.neibs_global = neibs_global
        self.neibs_local = neibs_local
        self.weights_global = weights_global
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        views_dict = {
            view_name: self.embeddings[view_name][idx]
            for view_name in self.view_names
        }
        if len(self.neibs_global[idx]) > 0:
            global_idx = np.random.choice(len(self.neibs_global[idx]))
            id_global_neighbor = self.neibs_global[idx][global_idx]
            weight_global = self.weights_global[idx][global_idx]
            id_local_neighbor = np.random.choice(self.neibs_local[idx])
        else:
            id_global_neighbor = idx
            id_local_neighbor = idx
            weight_global = 0.0
        neighbor_global = {
            view_name: self.embeddings[view_name][id_global_neighbor]
            for view_name in self.view_names
        }
        neighbor_local = {
            view_name: self.embeddings[view_name][id_local_neighbor]
            for view_name in self.view_names
        }
        return views_dict, neighbor_global, neighbor_local, weight_global, self.labels[idx], idx
def load_all_views(data_name, view_names, split='train', embeddings_dir='embeddings',
                   normalize=True, device='cpu', num_neibs=8):
    dataset = NCMODDataset(
        data_name=data_name,
        view_names=view_names,
        split=split,
        embeddings_dir=embeddings_dir,
        normalize=normalize,
        device=device,
        num_neibs=num_neibs
    )
    views_dict = {
        view_name: dataset.embeddings[view_name].to(device)
        for view_name in view_names
    }
    labels = dataset.labels.to(device)
    return dataset, views_dict, labels

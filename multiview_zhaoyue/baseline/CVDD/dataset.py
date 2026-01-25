import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", re.UNICODE)
def simple_tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    return _TOKEN_PATTERN.findall(text)
@dataclass
class SimpleVocab:
    stoi: Dict[str, int]
    itos: List[str]
    @property
    def vocab_size(self) -> int:
        return len(self.itos)
    def encode(self, text: str) -> torch.LongTensor:
        tokens = simple_tokenize(text)
        unk = self.stoi["<unk>"]
        ids = [self.stoi.get(t, unk) for t in tokens]
        return torch.tensor(ids, dtype=torch.long)
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                out.append(self.itos[i])
        return " ".join(out)
def build_vocab_from_jsonl(train_jsonl_path: str, min_freq: int = 2, max_vocab_size: int = 50000,
                           train_only_normal: bool = True) -> SimpleVocab:
    counts: Dict[str, int] = {}
    with open(train_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if train_only_normal and int(obj.get("label", 0)) != 0:
                continue
            for t in simple_tokenize(obj.get("text", "")):
                counts[t] = counts.get(t, 0) + 1
    items = [(t, c) for t, c in counts.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    items = items[: max(0, max_vocab_size - 2)]
    itos = ["<pad>", "<unk>"] + [t for t, _ in items]
    stoi = {t: i for i, t in enumerate(itos)}
    return SimpleVocab(stoi=stoi, itos=itos)
class JsonlTextDataset(Dataset):
    def __init__(self, jsonl_path: str, vocab: SimpleVocab, max_len: int = 256, train_only_normal: bool = False):
        self.jsonl_path = jsonl_path
        self.vocab = vocab
        self.max_len = max_len
        self.train_only_normal = train_only_normal
        self.samples: List[Tuple[int, torch.LongTensor, float]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = float(obj.get("label", 0))
                if self.train_only_normal and int(label) != 0:
                    continue
                ids = vocab.encode(obj.get("text", ""))
                if ids.numel() == 0:
                    ids = torch.tensor([vocab.stoi["<unk>"]], dtype=torch.long)
                if ids.numel() > max_len:
                    ids = ids[:max_len]
                self.samples.append((i, ids, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        sample_idx, ids, label = self.samples[idx]
        return sample_idx, ids, torch.tensor(label, dtype=torch.float32)
class JsonlBertDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_len: int = 256, train_only_normal: bool = False):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_only_normal = train_only_normal
        self.samples: List[Tuple[int, torch.LongTensor, torch.LongTensor, float]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = float(obj.get("label", 0))
                if self.train_only_normal and int(label) != 0:
                    continue
                encoded = tokenizer(
                    obj.get("text", ""),
                    truncation=True,
                    max_length=max_len,
                    add_special_tokens=True,
                    return_attention_mask=True,
                )
                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
                attn = torch.tensor(encoded["attention_mask"], dtype=torch.long)
                if input_ids.numel() == 0:
                    input_ids = torch.tensor([tokenizer.unk_token_id], dtype=torch.long)
                    attn = torch.tensor([1], dtype=torch.long)
                self.samples.append((i, input_ids, attn, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        sample_idx, ids, attn, label = self.samples[idx]
        return sample_idx, ids, attn, torch.tensor(label, dtype=torch.float32)
def collate_cvdd(batch):
    idxs, seqs, labels = zip(*batch)
    lengths = [s.numel() for s in seqs]
    max_len = max(lengths)
    pad_id = 0
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, : s.numel()] = s
    text_batch = padded.t().contiguous()
    label_batch = torch.stack(labels)
    weight_batch = torch.empty(0)
    return list(idxs), text_batch, label_batch, weight_batch
def collate_cvdd_bert(batch):
    idxs, seqs, attns, labels = zip(*batch)
    lengths = [s.numel() for s in seqs]
    max_len = max(lengths)
    pad_id = 0
    padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, (s, a) in enumerate(zip(seqs, attns)):
        padded[i, : s.numel()] = s
        padded_attn[i, : a.numel()] = a
    text_batch = padded.t().contiguous()
    attn_batch = padded_attn.t().contiguous()
    label_batch = torch.stack(labels)
    weight_batch = torch.empty(0)
    return list(idxs), text_batch, label_batch, weight_batch, attn_batch
class BucketBatchSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, shuffle: bool = True, drop_last: bool = True,
                 bucket_size_multiplier: int = 100):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = max(batch_size, batch_size * bucket_size_multiplier)
    def __iter__(self):
        import random
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.bucket_size):
            bucket = indices[i:i + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx])
            if self.shuffle:
                chunks = [bucket[j:j + self.batch_size] for j in range(0, len(bucket), self.batch_size)]
                random.shuffle(chunks)
                for c in chunks:
                    if len(c) < self.batch_size and self.drop_last:
                        continue
                    yield c
            else:
                for j in range(0, len(bucket), self.batch_size):
                    c = bucket[j:j + self.batch_size]
                    if len(c) < self.batch_size and self.drop_last:
                        continue
                    yield c
    def __len__(self):
        n = len(self.lengths) // self.batch_size
        return n
class CVDDJsonlDataset:
    def __init__(self, data_root: str, dataset_name: str, min_freq: int = 2, max_vocab_size: int = 50000,
                 max_len: int = 256, use_bert_tokenizer: bool = True, bert_name: str = "bert-base-uncased",
                 bert_cache_dir: str = None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        train_path = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
        test_path = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Missing train jsonl: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing test jsonl: {test_path}")
        self.use_bert_tokenizer = use_bert_tokenizer
        if use_bert_tokenizer:
            from transformers import AutoTokenizer
            self.encoder = AutoTokenizer.from_pretrained(bert_name, cache_dir=bert_cache_dir)
            self.train_set = JsonlBertDataset(train_path, self.encoder, max_len=max_len, train_only_normal=True)
            self.test_set = JsonlBertDataset(test_path, self.encoder, max_len=max_len, train_only_normal=False)
        else:
            self.encoder = build_vocab_from_jsonl(
                train_jsonl_path=train_path,
                min_freq=min_freq,
                max_vocab_size=max_vocab_size,
                train_only_normal=True,
            )
            self.train_set = JsonlTextDataset(train_path, self.encoder, max_len=max_len, train_only_normal=True)
            self.test_set = JsonlTextDataset(test_path, self.encoder, max_len=max_len, train_only_normal=False)
    def loaders(self, batch_size: int, shuffle_train: bool = True, shuffle_test: bool = False,
                num_workers: int = 0):
        if self.use_bert_tokenizer:
            train_lengths = [int(s[1].numel()) for s in self.train_set.samples]
            test_lengths = [int(s[1].numel()) for s in self.test_set.samples]
            train_sampler = BucketBatchSampler(train_lengths, batch_size=batch_size, shuffle=shuffle_train,
                                              drop_last=True)
            test_sampler = BucketBatchSampler(test_lengths, batch_size=batch_size, shuffle=shuffle_test,
                                             drop_last=False)
            train_loader = DataLoader(
                dataset=self.train_set,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=collate_cvdd_bert,
            )
            test_loader = DataLoader(
                dataset=self.test_set,
                batch_sampler=test_sampler,
                num_workers=num_workers,
                collate_fn=collate_cvdd_bert,
            )
        else:
            train_lengths = [int(s[1].numel()) for s in self.train_set.samples]
            test_lengths = [int(s[1].numel()) for s in self.test_set.samples]
            train_sampler = BucketBatchSampler(train_lengths, batch_size=batch_size, shuffle=shuffle_train,
                                              drop_last=True)
            test_sampler = BucketBatchSampler(test_lengths, batch_size=batch_size, shuffle=shuffle_test,
                                             drop_last=False)
            train_loader = DataLoader(
                dataset=self.train_set,
                batch_sampler=train_sampler,
                num_workers=num_workers,
                collate_fn=collate_cvdd,
            )
            test_loader = DataLoader(
                dataset=self.test_set,
                batch_sampler=test_sampler,
                num_workers=num_workers,
                collate_fn=collate_cvdd,
            )
        return train_loader, test_loader

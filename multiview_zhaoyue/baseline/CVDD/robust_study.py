import os
import sys
import time
from typing import List, Optional
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
ROBUST_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
def _set_random_seed(seed: int) -> None:
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        return
if __package__ is None or __package__ == "":
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from multiview_zhaoyue.baseline.CVDD.cvdd import CVDD
    from multiview_zhaoyue.baseline.CVDD.dataset import BucketBatchSampler, collate_cvdd, collate_cvdd_bert
else:
    from .cvdd import CVDD
    from .dataset import BucketBatchSampler, collate_cvdd, collate_cvdd_bert
def _save_dataset_rows_excel_or_csv(out_dir: str, dataset: str, rows: List[dict]):
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"CVDD_robust_{dataset}.xlsx")
    out_csv = os.path.join(out_dir, f"CVDD_robust_{dataset}.csv")
    try:
        import pandas as pd
        import numpy as np
        expected_levels = [int(x * 100) for x in ROBUST_LEVELS]
        df = pd.DataFrame(rows)
        if not df.empty:
            if 'train_anom_pct' not in df.columns and 'contam_ratio' in df.columns:
                df['train_anom_pct'] = (df['contam_ratio'].astype(float) * 100).round().astype(int)
            if 'method' not in df.columns:
                df['method'] = 'CVDD'
            summary_df = df.pivot_table(
                index=['dataset', 'method'],
                columns='train_anom_pct',
                values='auc',
                aggfunc='mean',
            )
            for lvl in expected_levels:
                if lvl not in summary_df.columns:
                    summary_df[lvl] = np.nan
            summary_df = summary_df[expected_levels]
            col_rename = {lvl: f"{lvl}%异常AUC" for lvl in expected_levels}
            summary_df = summary_df.rename(columns=col_rename).reset_index()
            summary_df = summary_df.rename(columns={'dataset': '数据集名称', 'method': '我们的方法名称'})
            summary_df = summary_df.sort_values(['数据集名称', '我们的方法名称']).reset_index(drop=True)
        else:
            cols = ['数据集名称', '我们的方法名称'] + [f"{lvl}%异常AUC" for lvl in expected_levels]
            summary_df = pd.DataFrame(columns=cols)
        summary_df.to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        import csv
        if len(rows) == 0:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                f.write("")
            return out_csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return out_csv
class _InMemoryBertDataset:
    def __init__(self, samples, lengths):
        self.samples = samples
        self.lengths = lengths
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        return self.samples[idx]
class _InMemoryTextDataset:
    def __init__(self, samples, lengths):
        self.samples = samples
        self.lengths = lengths
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx: int):
        return self.samples[idx]
class RobustCVDDDataset:
    def __init__(
        self,
        train_samples,
        test_samples,
        train_lengths,
        test_lengths,
        use_bert_tokenizer: bool,
        encoder,
        max_len: int = 256,
        bert_name: str = "bert-base-uncased",
        bert_cache_dir: str = None,
    ):
        self.use_bert_tokenizer = bool(use_bert_tokenizer)
        self.encoder = encoder
        self.bert_name = bert_name
        self.bert_cache_dir = bert_cache_dir
        self.max_len = max_len
        if self.use_bert_tokenizer:
            self.train_set = _InMemoryBertDataset(train_samples, train_lengths)
            self.test_set = _InMemoryBertDataset(test_samples, test_lengths)
        else:
            self.train_set = _InMemoryTextDataset(train_samples, train_lengths)
            self.test_set = _InMemoryTextDataset(test_samples, test_lengths)
    def loaders(self, batch_size: int, shuffle_train: bool = True, shuffle_test: bool = False, num_workers: int = 0):
        from torch.utils.data import DataLoader
        if self.use_bert_tokenizer:
            train_sampler = BucketBatchSampler(self.train_set.lengths, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
            test_sampler = BucketBatchSampler(self.test_set.lengths, batch_size=batch_size, shuffle=shuffle_test, drop_last=False)
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
            train_sampler = BucketBatchSampler(self.train_set.lengths, batch_size=batch_size, shuffle=shuffle_train, drop_last=True)
            test_sampler = BucketBatchSampler(self.test_set.lengths, batch_size=batch_size, shuffle=shuffle_test, drop_last=False)
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
def _load_jsonl_records(jsonl_path: str):
    import json
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            label = int(obj.get("label", 0))
            records.append((i, text, label))
    return records
def _encode_records_bert(records, tokenizer, max_len: int):
    import torch
    samples = []
    lengths = []
    for idx, text, label in records:
        encoded = tokenizer(
            text,
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
        samples.append((idx, input_ids, attn, torch.tensor(float(label), dtype=torch.float32)))
        lengths.append(int(input_ids.numel()))
    return samples, lengths
def _encode_records_text(records, vocab, max_len: int):
    import torch
    samples = []
    lengths = []
    for idx, text, label in records:
        ids = vocab.encode(text)
        if ids.numel() == 0:
            ids = torch.tensor([vocab.stoi["<unk>"]], dtype=torch.long)
        if ids.numel() > max_len:
            ids = ids[:max_len]
        samples.append((idx, ids, torch.tensor(float(label), dtype=torch.float32)))
        lengths.append(int(ids.numel()))
    return samples, lengths
def _build_robust_split_indices(test_labels: List[int], anomaly_train_ratio: float, seed: int):
    import numpy as np
    test_norm_idx = [i for i, y in enumerate(test_labels) if int(y) == 0]
    test_anom_idx = [i for i, y in enumerate(test_labels) if int(y) == 1]
    n_test_anom = int(len(test_anom_idx))
    n_move = int(float(anomaly_train_ratio) * n_test_anom)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n_test_anom) if n_test_anom > 0 else np.array([], dtype=np.int64)
    n_keep = int(0.8 * n_test_anom)
    keep_pos = perm[:n_keep]
    pool_pos = perm[n_keep:]
    keep_anom = [test_anom_idx[int(p)] for p in keep_pos.tolist()] if n_keep > 0 else []
    pool_anom = [test_anom_idx[int(p)] for p in pool_pos.tolist()] if pool_pos.size > 0 else []
    if n_move > int(len(pool_anom)):
        n_move = int(len(pool_anom))
    if n_move > 0:
        pool_perm = rng.permutation(len(pool_anom))
        move_anom = [pool_anom[int(p)] for p in pool_perm[:n_move].tolist()]
    else:
        move_anom = []
    return test_norm_idx, move_anom, keep_anom
def run_one_dataset_robust(
    dataset_name: str,
    data_root: str,
    device: str = "cuda",
    seed: int = 42,
    anomaly_train_ratio: float = 0.0,
    embedding_size: int = 100,
    attention_size: int = 150,
    n_attention_heads: int = 3,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    lambda_p: float = 1.0,
    max_len: int = 256,
    min_freq: int = 2,
    max_vocab_size: int = 50000,
    use_bert_tokenizer: bool = True,
    bert_name: str = "bert-base-uncased",
    bert_cache_dir: str = None,
):
    train_path = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_path = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train jsonl: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test jsonl: {test_path}")
    train_records_all = _load_jsonl_records(train_path)
    test_records_all = _load_jsonl_records(test_path)
    train_records_normal = [r for r in train_records_all if int(r[2]) == 0]
    test_records_normal = [r for r in test_records_all if int(r[2]) == 0]
    test_records_anom = [r for r in test_records_all if int(r[2]) == 1]
    test_labels_all = [int(r[2]) for r in test_records_all]
    test_norm_idx, move_anom_idx, keep_anom_idx = _build_robust_split_indices(
        test_labels=test_labels_all,
        anomaly_train_ratio=float(anomaly_train_ratio),
        seed=int(seed),
    )
    move_anom_records = [test_records_all[i] for i in move_anom_idx]
    keep_anom_records = [test_records_all[i] for i in keep_anom_idx]
    train_records = list(train_records_normal) + list(move_anom_records)
    test_records = list(test_records_normal) + list(keep_anom_records)
    if use_bert_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_name, cache_dir=bert_cache_dir)
        train_samples, train_lengths = _encode_records_bert(train_records, tokenizer, max_len=max_len)
        test_samples, test_lengths = _encode_records_bert(test_records, tokenizer, max_len=max_len)
        encoder = tokenizer
    else:
        if __package__ is None or __package__ == "":
            from multiview_zhaoyue.baseline.CVDD.dataset import build_vocab_from_jsonl
        else:
            from .dataset import build_vocab_from_jsonl
        vocab = build_vocab_from_jsonl(
            train_jsonl_path=train_path,
            min_freq=min_freq,
            max_vocab_size=max_vocab_size,
            train_only_normal=True,
        )
        train_samples, train_lengths = _encode_records_text(train_records, vocab, max_len=max_len)
        test_samples, test_lengths = _encode_records_text(test_records, vocab, max_len=max_len)
        encoder = vocab
    dataset = RobustCVDDDataset(
        train_samples=train_samples,
        test_samples=test_samples,
        train_lengths=train_lengths,
        test_lengths=test_lengths,
        use_bert_tokenizer=use_bert_tokenizer,
        encoder=encoder,
        max_len=max_len,
        bert_name=bert_name,
        bert_cache_dir=bert_cache_dir,
    )
    model = CVDD(ad_score="context_dist_mean")
    model.set_network(
        dataset=dataset,
        embedding_size=embedding_size,
        attention_size=attention_size,
        n_attention_heads=n_attention_heads,
        freeze_embedding=False,
        bert_name=bert_name,
        bert_cache_dir=bert_cache_dir,
    )
    _set_random_seed(seed)
    t0 = time.time()
    model.train(
        dataset=dataset,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lambda_p=lambda_p,
        alpha_scheduler="logarithmic",
        device=device,
        show_progress=True,
        desc_prefix=f"[{dataset_name}] [train_anom_{int(float(anomaly_train_ratio) * 100)}%] ",
    )
    model.test(
        dataset=dataset,
        device=device,
        show_progress=True,
        desc_prefix=f"[{dataset_name}] [train_anom_{int(float(anomaly_train_ratio) * 100)}%] ",
    )
    t1 = time.time()
    stats = {
        "train_normal": int(len(train_records_normal)),
        "train_anomaly": int(len(move_anom_records)),
        "test_normal": int(len(test_records_normal)),
        "test_anomaly": int(len(keep_anom_records)),
        "total_anomaly": int(len(test_records_anom)),
    }
    return {
        "dataset": dataset_name,
        "method": "CVDD",
        "seed": int(seed),
        "contam_ratio": float(anomaly_train_ratio),
        "train_anom_pct": int(float(anomaly_train_ratio) * 100),
        "contam_tag": f"train_anom_{int(float(anomaly_train_ratio) * 100)}%",
        "train_normal": stats["train_normal"],
        "train_anomaly": stats["train_anomaly"],
        "test_normal": stats["test_normal"],
        "test_anomaly": stats["test_anomaly"],
        "auc": float(model.results.get("test_auc") or 0.0),
        "total_time_sec": float(t1 - t0),
    }
def main(datasets: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="Robust study for CVDD baseline (semi-supervised contamination)")
    parser.add_argument("--datasets", type=str, default="covid_fake,bbc,email_spam")
    parser.add_argument("--device", type=str, default=os.environ.get("CVDD_DEVICE", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的seed列表，例如 41,42,43；为空则使用 --seed")
    args = parser.parse_args()
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    data_root = os.path.join(repo_root, "data")
    out_dir = os.path.join(repo_root, "multiview_zhaoyue", "baseline", "CVDD")
    device = args.device
    for ds in tqdm(datasets, desc="datasets", leave=True):
        rows = []
        for contam_ratio in ROBUST_LEVELS:
            for seed in seeds:
                row = run_one_dataset_robust(
                    dataset_name=ds,
                    data_root=data_root,
                    device=device,
                    seed=int(seed),
                    anomaly_train_ratio=float(contam_ratio),
                )
                print(
                    f"[CVDD Robust] dataset={row.get('dataset')} seed={int(seed)} contam={float(contam_ratio):.2f} "
                    f"auc={float(row.get('auc') or 0.0):.6f} total_time_sec={float(row.get('total_time_sec') or 0.0):.2f}"
                )
                rows.append(row)
        seeds_tag = "_".join(str(s) for s in seeds)
        out_key = f"{ds}_seeds_{seeds_tag}" if len(seeds) > 1 else f"{ds}_seed_{seeds[0]}"
        _save_dataset_rows_excel_or_csv(out_dir=out_dir, dataset=out_key, rows=rows)
if __name__ == "__main__":
    main()

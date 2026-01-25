import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROBUST_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
def _save_dataset_rows_excel(out_dir: str, dataset: str, rows: list) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"FATE_robust_{dataset}.xlsx")
    try:
        import pandas as pd
        import numpy as np
        expected_levels = [int(x * 100) for x in ROBUST_LEVELS]
        df = pd.DataFrame(rows)
        if not df.empty:
            if 'train_anom_pct' not in df.columns and 'contam_ratio' in df.columns:
                df['train_anom_pct'] = (df['contam_ratio'].astype(float) * 100).round().astype(int)
            if 'method' not in df.columns:
                df['method'] = 'FATE'
            summary_df = df.pivot_table(
                index=['dataset', 'method'],
                columns='train_anom_pct',
                values='auroc',
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
        out_csv = os.path.join(out_dir, f"FATE_robust_{dataset}.csv")
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
def _read_jsonl_texts(path: str):
    import json
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(str(obj.get("text", "")))
            labels.append(int(obj.get("label", 0)))
    return texts, labels
def _build_robust_split_texts(dataset_name: str, data_root: str, seed: int, anomaly_train_ratio: float):
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if not os.path.exists(train_jsonl) or not os.path.exists(test_jsonl):
        raise FileNotFoundError(
            f"缺少数据文件: {dataset_name}\n"
            f"需要存在: {train_jsonl} 和 {test_jsonl}"
        )
    train_texts, train_labels = _read_jsonl_texts(train_jsonl)
    if any(int(l) != 0 for l in train_labels):
        raise ValueError(f"训练集必须全部为正常样本(label=0)，但 {train_jsonl} 中存在异常")
    test_texts, test_labels = _read_jsonl_texts(test_jsonl)
    test_normals = [t for t, y in zip(test_texts, test_labels) if int(y) == 0]
    test_anoms = [t for t, y in zip(test_texts, test_labels) if int(y) == 1]
    if len(test_anoms) == 0:
        raise ValueError(f"测试集必须包含异常样本(label=1)，但 {test_jsonl} 中没有")
    import numpy as np
    rng = np.random.RandomState(int(seed))
    n_anom = int(len(test_anoms))
    n_move = int(float(anomaly_train_ratio) * n_anom)
    n_keep = int(0.8 * n_anom)
    perm = rng.permutation(n_anom) if n_anom > 0 else np.array([], dtype=np.int64)
    keep_pos = perm[:n_keep]
    pool_pos = perm[n_keep:]
    remaining_anoms = [test_anoms[int(i)] for i in keep_pos.tolist()] if n_keep > 0 else []
    pool_anoms = [test_anoms[int(i)] for i in pool_pos.tolist()] if pool_pos.size > 0 else []
    if n_move > int(len(pool_anoms)):
        n_move = int(len(pool_anoms))
    if n_move > 0:
        pool_perm = rng.permutation(len(pool_anoms))
        moved_anoms = [pool_anoms[int(i)] for i in pool_perm[:n_move].tolist()]
    else:
        moved_anoms = []
    train_all_texts = list(train_texts) + list(moved_anoms)
    train_all_labels = [0] * len(train_texts) + [1] * len(moved_anoms)
    eval_texts = list(test_normals) + list(remaining_anoms)
    eval_labels = [0] * len(test_normals) + [1] * len(remaining_anoms)
    stats = {
        "train_normal": int(len(train_texts)),
        "train_anomaly": int(len(moved_anoms)),
        "test_normal": int(len(test_normals)),
        "test_anomaly": int(len(remaining_anoms)),
        "total_anomaly": int(len(test_anoms)),
    }
    return train_all_texts, train_all_labels, eval_texts, eval_labels, stats
def run_fate_one_dataset_robust(
    dataset_name: str,
    data_root: str,
    device: str = "cuda",
    seed: int = 42,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_seq_len: int = 128,
    train_batch_size: int = 16,
    eval_batch_size: int = 64,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
    anomaly_train_ratio: float = 0.0,
    attention_size: int = 150,
    num_heads: int = 5,
    topk_ratio: float = 0.1,
    include_regularization: bool = True,
    num_workers: int = 0,
    suppress_internal_output: bool = True,
):
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer
    if __package__ is None or __package__ == "":
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from multiview_zhaoyue.baseline.FATE.dataset import BalancedBatchSampler
        from multiview_zhaoyue.baseline.FATE.loss import DeviationLoss
        from multiview_zhaoyue.baseline.FATE.model import FateConfig, FateModel
        from multiview_zhaoyue.baseline.FATE.trainer import _predict_scores, set_seed
    else:
        from .dataset import BalancedBatchSampler
        from .loss import DeviationLoss
        from .model import FateConfig, FateModel
        from .trainer import _predict_scores, set_seed
    set_seed(int(seed))
    import torch
    torch_device = torch.device(device if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
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
    train_texts, train_labels, eval_texts, eval_labels, stats = _build_robust_split_texts(
        dataset_name=dataset_name,
        data_root=data_root,
        seed=int(seed),
        anomaly_train_ratio=float(anomaly_train_ratio),
    )
    train_enc = tokenizer(
        train_texts,
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
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32)
    test_labels_t = torch.tensor(eval_labels, dtype=torch.float32)
    train_ds = TensorDataset(train_enc["input_ids"].long(), train_enc["attention_mask"].long(), train_labels_t)
    test_ds = TensorDataset(test_enc["input_ids"].long(), test_enc["attention_mask"].long(), test_labels_t)
    if int(stats["train_anomaly"]) > 0:
        train_sampler = BalancedBatchSampler(train_labels_t, batch_size=int(train_batch_size), seed=int(seed))
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=int(train_batch_size),
            shuffle=True,
            num_workers=int(num_workers),
            pin_memory=bool(pin_memory),
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    criterion = DeviationLoss()
    use_amp = torch_device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    head_eye = torch.eye(int(num_heads), device=torch_device)
    t0 = time.time()
    model.train()
    for _epoch in range(int(num_epochs)):
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
    t1 = time.time()
    total_time = float(t1 - t0)
    if not suppress_internal_output:
        print(
            f"[FATE Robust] {dataset_name} contam={float(anomaly_train_ratio):.2f} "
            f"AUROC={auroc:.6f} total_run_time_sec={total_time:.2f}"
        )
    row = {
        "dataset": str(dataset_name),
        "method": "FATE",
        "seed": int(seed),
        "contam_ratio": float(anomaly_train_ratio),
        "train_anom_pct": int(float(anomaly_train_ratio) * 100),
        "contam_tag": f"train_anom_{int(float(anomaly_train_ratio) * 100)}%",
        "train_normal": int(stats.get("train_normal") or 0),
        "train_anomaly": int(stats.get("train_anomaly") or 0),
        "test_normal": int(stats.get("test_normal") or 0),
        "test_anomaly": int(stats.get("test_anomaly") or 0),
        "auroc": float(auroc),
        "total_run_time_sec": float(total_time),
        "model_name": str(model_name),
        "max_seq_len": int(max_seq_len),
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "num_epochs": int(num_epochs),
        "lr": float(learning_rate),
        "attention_size": int(attention_size),
        "num_heads": int(num_heads),
        "topk_ratio": float(topk_ratio),
        "regularization": bool(include_regularization),
        "device": str(device),
    }
    return row
def main():
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="Robust study for FATE baseline on multiview_zhaoyue datasets")
    parser.add_argument("--datasets", type=str, default="covid_fake,bbc,email_spam", help="逗号分隔的数据集列表；为空则跑默认10个")
    parser.add_argument("--device", type=str, default=os.environ.get("FATE_DEVICE", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的seed列表，例如 41,42,43；为空则使用 --seed")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--attention_size", type=int, default=150)
    parser.add_argument("--num_heads", type=int, default=5)
    parser.add_argument("--topk_ratio", type=float, default=0.1)
    parser.add_argument("--no_regularization", action="store_true")
    args = parser.parse_args()
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    data_root = os.path.join(repo_root, "data")
    out_dir = os.path.join(repo_root, "multiview_zhaoyue", "baseline", "FATE")
    failures = []
    for ds in tqdm(datasets, desc="datasets", leave=True):
        rows = []
        for contam_ratio in ROBUST_LEVELS:
            for seed in seeds:
                try:
                    row = run_fate_one_dataset_robust(
                        dataset_name=ds,
                        data_root=data_root,
                        device=args.device,
                        seed=int(seed),
                        model_name=args.model_name,
                        max_seq_len=args.max_seq_len,
                        train_batch_size=args.train_batch_size,
                        eval_batch_size=args.eval_batch_size,
                        num_epochs=args.num_epochs,
                        learning_rate=args.lr,
                        anomaly_train_ratio=float(contam_ratio),
                        attention_size=args.attention_size,
                        num_heads=args.num_heads,
                        topk_ratio=args.topk_ratio,
                        include_regularization=(not args.no_regularization),
                        num_workers=0,
                        suppress_internal_output=True,
                    )
                    rows.append(row)
                    print(
                        f"[FATE Robust] dataset={ds} seed={int(seed)} contam={float(contam_ratio):.2f} "
                        f"AUROC={float(row.get('auroc') or 0.0):.6f} total_run_time_sec={float(row.get('total_run_time_sec') or 0.0):.2f}"
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    failures.append((ds, int(seed), float(contam_ratio), f"{type(e).__name__}: {e}"))
                    print(
                        f"[FATE Robust] dataset={ds} seed={int(seed)} contam={float(contam_ratio):.2f} "
                        f"FAILED: {type(e).__name__}: {e}"
                    )
        seeds_tag = "_".join(str(s) for s in seeds)
        out_key = f"{ds}_seeds_{seeds_tag}" if len(seeds) > 1 else f"{ds}_seed_{seeds[0]}"
        _save_dataset_rows_excel(out_dir=out_dir, dataset=out_key, rows=rows)
    if failures:
        print("[FATE Robust] finished with failures:")
        for ds, seed, contam, err in failures:
            print(f"[FATE Robust] dataset={ds} seed={seed} contam={contam} error={err}")
if __name__ == "__main__":
    main()

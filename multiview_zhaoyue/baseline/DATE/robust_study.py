import os
import sys
import time
ROBUST_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
def _save_dataset_rows_excel(out_dir: str, dataset: str, rows: list) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"DATE_robust_{dataset}.xlsx")
    try:
        import pandas as pd
        import numpy as np
        expected_levels = [int(x * 100) for x in ROBUST_LEVELS]
        df = pd.DataFrame(rows)
        if not df.empty:
            if 'train_anom_pct' not in df.columns and 'contam_ratio' in df.columns:
                df['train_anom_pct'] = (df['contam_ratio'].astype(float) * 100).round().astype(int)
            if 'method' not in df.columns:
                df['method'] = 'DATE'
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
        out_csv = os.path.join(out_dir, f"DATE_robust_{dataset}.csv")
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
def _read_jsonl_texts_and_labels(path: str):
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
def _dump_txt(lines, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in lines:
            t = (t or "").replace("\n", " ").strip()
            if not t:
                continue
            f.write(t + "\n")
def _build_robust_txts(
    dataset_name: str,
    data_root: str,
    work_dir: str,
    seed: int,
    anomaly_train_ratio: float,
):
    train_jsonl = os.path.join(data_root, f"{dataset_name}_train_data.jsonl")
    test_jsonl = os.path.join(data_root, f"{dataset_name}_test_data.jsonl")
    if not os.path.exists(train_jsonl) or not os.path.exists(test_jsonl):
        raise FileNotFoundError(
            f"缺少数据文件: {dataset_name}\n"
            f"需要存在: {train_jsonl} 和 {test_jsonl}"
        )
    train_texts, train_labels = _read_jsonl_texts_and_labels(train_jsonl)
    if any(int(l) != 0 for l in train_labels):
        raise ValueError(
            f"训练集必须全部为正常样本(label=0)，但 {train_jsonl} 中出现了: {sorted(set(train_labels))}"
        )
    test_texts, test_labels = _read_jsonl_texts_and_labels(test_jsonl)
    test_inliers = [t for t, y in zip(test_texts, test_labels) if int(y) == 0]
    test_outliers = [t for t, y in zip(test_texts, test_labels) if int(y) == 1]
    if len(test_outliers) == 0:
        raise ValueError(f"测试集必须包含异常样本(label=1)，但 {test_jsonl} 中没有")
    import numpy as np
    rng = np.random.RandomState(int(seed))
    n_out = int(len(test_outliers))
    n_move = int(float(anomaly_train_ratio) * n_out)
    n_keep = int(0.8 * n_out)
    perm = rng.permutation(n_out) if n_out > 0 else np.array([], dtype=np.int64)
    keep_pos = perm[:n_keep]
    pool_pos = perm[n_keep:]
    keep_outliers = [test_outliers[int(i)] for i in keep_pos.tolist()] if n_keep > 0 else []
    pool_outliers = [test_outliers[int(i)] for i in pool_pos.tolist()] if pool_pos.size > 0 else []
    if n_move > int(len(pool_outliers)):
        n_move = int(len(pool_outliers))
    if n_move > 0:
        pool_perm = rng.permutation(len(pool_outliers))
        move_outliers = [pool_outliers[int(i)] for i in pool_perm[:n_move].tolist()]
    else:
        move_outliers = []
    os.makedirs(work_dir, exist_ok=True)
    train_txt = os.path.join(work_dir, "train.txt")
    test_txt = os.path.join(work_dir, "test.txt")
    outliers_txt = os.path.join(work_dir, "outliers.txt")
    robust_train = list(train_texts) + list(move_outliers)
    _dump_txt(robust_train, train_txt)
    _dump_txt(test_inliers, test_txt)
    _dump_txt(keep_outliers, outliers_txt)
    stats = {
        "train_normal": int(len(train_texts)),
        "train_anomaly": int(len(move_outliers)),
        "test_normal": int(len(test_inliers)),
        "test_anomaly": int(len(keep_outliers)),
        "total_anomaly": int(len(test_outliers)),
    }
    return train_txt, test_txt, outliers_txt, stats
def run_date_one_dataset_robust(
    dataset_name: str,
    data_root: str,
    out_dir: str,
    device: str = "cuda",
    seed: int = 42,
    anomaly_train_ratio: float = 0.0,
    seq_len: int = 128,
    num_train_epochs: int = 20,
    train_batch_size: int = 16,
    eval_batch_size: int = 16,
    anomaly_batch_size: int = 16,
    max_lr: float = 1e-5,
    min_lr: float = 1e-4,
    warmup_steps: int = 1000,
    weight_decay: float = 0.1,
    disc_drop: float = 0.5,
    disc_hid_layers: int = 4,
    disc_hid_size: int = 256,
    gen_hid_layers: int = 1,
    gen_hid_size: int = 16,
    rtd_loss_weight: int = 50,
    rmd_loss_weight: int = 100,
    mlm_loss_weight: int = 1,
    log_every_n_epochs: int = 5,
    suppress_internal_output: bool = True,
):
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    if __package__ is None or __package__ == "":
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from multiview_zhaoyue.baseline.DATE.trainer import _make_mlm_inputs, _read_txt_lines, set_seed
        from multiview_zhaoyue.baseline.DATE.model import DateConfig, DateModel, compute_rtd_anomaly_score
    else:
        from .trainer import _make_mlm_inputs, _read_txt_lines, set_seed
        from .model import DateConfig, DateModel, compute_rtd_anomaly_score
    work_dir = os.path.join(out_dir, "_work_robust", dataset_name, f"seed_{int(seed)}", f"train_anom_{int(float(anomaly_train_ratio) * 100)}%")
    train_txt, test_txt, outliers_txt, stats = _build_robust_txts(
        dataset_name=dataset_name,
        data_root=data_root,
        work_dir=work_dir,
        seed=int(seed),
        anomaly_train_ratio=float(anomaly_train_ratio),
    )
    set_seed(int(seed))
    use_cuda = str(device).lower().startswith("cuda") and torch.cuda.is_available()
    torch_device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    max_length = int(seq_len) + 2
    cfg = DateConfig(
        seq_len=int(seq_len),
        n_masks=50,
        vocab_size=int(getattr(tokenizer, "vocab_size", 30522)),
        gen_hidden_size=int(gen_hid_size),
        gen_num_layers=int(gen_hid_layers),
        disc_hidden_size=int(disc_hid_size),
        disc_num_layers=int(disc_hid_layers),
        dropout=float(disc_drop),
    )
    model = DateModel(cfg=cfg, random_generator=True, seed=int(seed)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(max_lr), weight_decay=float(weight_decay))
    train_texts = _read_txt_lines(train_txt)
    test_inlier_texts = _read_txt_lines(test_txt)
    test_outlier_texts = _read_txt_lines(outliers_txt)
    def _collate_texts(batch):
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids")
    train_loader = DataLoader(train_texts, batch_size=int(train_batch_size), shuffle=True, num_workers=0, collate_fn=_collate_texts)
    eval_in_loader = DataLoader(test_inlier_texts, batch_size=int(eval_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    eval_out_loader = DataLoader(test_outlier_texts, batch_size=int(anomaly_batch_size), shuffle=False, num_workers=0, collate_fn=_collate_texts)
    t0 = time.time()
    model.train()
    rng = np.random.RandomState(int(seed))
    def _eval_current_model_auroc() -> float:
        model.eval()
        in_scores = []
        out_scores = []
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids in eval_in_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                in_scores.append((1.0 - s).detach().cpu().numpy())
            for input_ids, attention_mask, token_type_ids in eval_out_loader:
                input_ids = input_ids.to(torch_device)
                attention_mask = attention_mask.to(torch_device)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(torch_device)
                s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
                out_scores.append((1.0 - s).detach().cpu().numpy())
        in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
        out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
        y_true = np.concatenate([np.ones_like(in_scores_np), np.zeros_like(out_scores_np)], axis=0)
        y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
        return float(roc_auc_score(y_true, y_score))
    for _epoch in range(int(num_train_epochs)):
        train_iter = train_loader
        for _step, (input_ids, attention_mask, token_type_ids) in enumerate(train_iter):
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            mask_idx = int(rng.randint(0, len(model.masks)))
            pseudo_mask = model.masks[mask_idx]
            masked_input, mlm_labels = _make_mlm_inputs(input_ids, attention_mask, tokenizer, pseudo_mask)
            rmd_labels = torch.full((input_ids.size(0),), mask_idx, device=torch_device, dtype=torch.long)
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                mlm_labels=mlm_labels,
                rmd_labels=rmd_labels,
                replace_tokens=True,
            )
            loss = 0.0
            if outputs.get("g_loss") is not None:
                loss = loss + float(mlm_loss_weight) * outputs["g_loss"]
            if outputs.get("rtd_loss") is not None:
                loss = loss + float(rtd_loss_weight) * outputs["rtd_loss"]
            if outputs.get("rmd_loss") is not None:
                loss = loss + float(rmd_loss_weight) * outputs["rmd_loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if int(log_every_n_epochs) > 0 and ((_epoch + 1) % int(log_every_n_epochs) == 0):
            auroc_mid = _eval_current_model_auroc()
            model.train()
            elapsed = float(time.time() - t0)
            msg = (
                f"[DATE Robust] dataset={dataset_name} contam={float(anomaly_train_ratio):.2f} "
                f"epoch={_epoch + 1}/{int(num_train_epochs)} AUROC={auroc_mid:.6f} elapsed_sec={elapsed:.2f}"
            )
            if suppress_internal_output:
                print(msg)
            else:
                print(msg)
    model.eval()
    in_scores = []
    out_scores = []
    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids in eval_in_loader:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            in_scores.append((1.0 - s).detach().cpu().numpy())
        for input_ids, attention_mask, token_type_ids in eval_out_loader:
            input_ids = input_ids.to(torch_device)
            attention_mask = attention_mask.to(torch_device)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(torch_device)
            s = compute_rtd_anomaly_score(model, input_ids, attention_mask, token_type_ids)
            out_scores.append((1.0 - s).detach().cpu().numpy())
    in_scores_np = np.concatenate(in_scores, axis=0) if len(in_scores) else np.array([])
    out_scores_np = np.concatenate(out_scores, axis=0) if len(out_scores) else np.array([])
    y_true = np.concatenate([np.ones_like(in_scores_np), np.zeros_like(out_scores_np)], axis=0)
    y_score = np.concatenate([in_scores_np, out_scores_np], axis=0)
    auroc = float(roc_auc_score(y_true, y_score))
    t1 = time.time()
    row = {
        "dataset": dataset_name,
        "method": "DATE",
        "seed": int(seed),
        "contam_ratio": float(anomaly_train_ratio),
        "train_anom_pct": int(float(anomaly_train_ratio) * 100),
        "contam_tag": f"train_anom_{int(float(anomaly_train_ratio) * 100)}%",
        "train_normal": int(stats.get("train_normal") or 0),
        "train_anomaly": int(stats.get("train_anomaly") or 0),
        "test_normal": int(stats.get("test_normal") or 0),
        "test_anomaly": int(stats.get("test_anomaly") or 0),
        "auroc": float(auroc),
        "total_run_time_sec": float(t1 - t0),
    }
    return row
def main():
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="Robust study for DATE baseline on multiview_zhaoyue datasets")
    parser.add_argument("--datasets", type=str, default="covid_fake,bbc,email_spam")
    parser.add_argument("--device", type=str, default=os.environ.get("DATE_DEVICE", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的seed列表，例如 41,42,43；为空则使用 --seed")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--anomaly_batch_size", type=int, default=16)
    parser.add_argument("--max_lr", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--disc_drop", type=float, default=0.5)
    parser.add_argument("--disc_hid_layers", type=int, default=4)
    parser.add_argument("--disc_hid_size", type=int, default=256)
    parser.add_argument("--gen_hid_layers", type=int, default=1)
    parser.add_argument("--gen_hid_size", type=int, default=16)
    parser.add_argument("--rtd_loss_weight", type=int, default=50)
    parser.add_argument("--rmd_loss_weight", type=int, default=100)
    parser.add_argument("--mlm_loss_weight", type=int, default=1)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_sleep_sec", type=float, default=30.0)
    parser.add_argument("--keep_internal_output", action="store_true", default=True)
    args = parser.parse_args()
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    EPOCHS_BY_DATASET = {
        "bbc": args.num_train_epochs,
        "olid": args.num_train_epochs,
        "agnews": 5,
        "movie_review": args.num_train_epochs,
        "N24News": 5,
        "email_spam": args.num_train_epochs,
        "smsspam": args.num_train_epochs,
        "covid_fake": args.num_train_epochs,
        "liar2": args.num_train_epochs,
        "hate_speech": 10,
    }
    BATCH_SIZE_BY_DATASET = {
        "bbc": args.train_batch_size,
        "olid": args.train_batch_size,
        "agnews": 128,
        "movie_review": args.train_batch_size,
        "N24News": 64,
        "email_spam": args.train_batch_size,
        "smsspam": args.train_batch_size,
        "covid_fake": args.train_batch_size,
        "liar2": args.train_batch_size,
        "hate_speech": args.train_batch_size,
    }
    repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    data_root = os.path.join(repo_root, "data")
    out_dir = os.path.join(repo_root, "multiview_zhaoyue", "baseline", "DATE")
    failures = []
    for ds in tqdm(datasets, desc="datasets", leave=True):
        ds_epochs = int(EPOCHS_BY_DATASET.get(ds, args.num_train_epochs))
        ds_bs = int(BATCH_SIZE_BY_DATASET.get(ds, args.train_batch_size))
        rows = []
        for contam_ratio in ROBUST_LEVELS:
            for seed in seeds:
                ok = False
                last_err = None
                retries = max(1, int(args.max_retries))
                for attempt in range(1, retries + 1):
                    try:
                        row = run_date_one_dataset_robust(
                            dataset_name=ds,
                            data_root=data_root,
                            out_dir=out_dir,
                            device=args.device,
                            seed=int(seed),
                            anomaly_train_ratio=float(contam_ratio),
                            seq_len=args.seq_len,
                            num_train_epochs=ds_epochs,
                            train_batch_size=ds_bs,
                            eval_batch_size=ds_bs,
                            anomaly_batch_size=ds_bs,
                            max_lr=args.max_lr,
                            min_lr=args.min_lr,
                            warmup_steps=args.warmup_steps,
                            weight_decay=args.weight_decay,
                            disc_drop=args.disc_drop,
                            disc_hid_layers=args.disc_hid_layers,
                            disc_hid_size=args.disc_hid_size,
                            gen_hid_layers=args.gen_hid_layers,
                            gen_hid_size=args.gen_hid_size,
                            rtd_loss_weight=args.rtd_loss_weight,
                            rmd_loss_weight=args.rmd_loss_weight,
                            mlm_loss_weight=args.mlm_loss_weight,
                            suppress_internal_output=(not args.keep_internal_output),
                        )
                        print(
                            f"[DATE Robust] dataset={ds} seed={int(seed)} contam={float(contam_ratio):.2f} "
                            f"AUROC={float(row.get('auroc') or 0.0):.6f} total_run_time_sec={float(row.get('total_run_time_sec') or 0.0):.2f}"
                        )
                        rows.append(row)
                        ok = True
                        break
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        last_err = e
                        print(
                            f"[DATE Robust] dataset={ds} seed={int(seed)} contam={float(contam_ratio):.2f} "
                            f"attempt={attempt}/{retries} FAILED: {type(e).__name__}: {e}"
                        )
                        if attempt < retries:
                            try:
                                time.sleep(float(args.retry_sleep_sec))
                            except Exception:
                                pass
                if not ok:
                    failures.append((ds, int(seed), float(contam_ratio), repr(last_err)))
        seeds_tag = "_".join(str(s) for s in seeds)
        out_key = f"{ds}_seeds_{seeds_tag}" if len(seeds) > 1 else f"{ds}_seed_{seeds[0]}"
        _save_dataset_rows_excel(out_dir=out_dir, dataset=out_key, rows=rows)
    if failures:
        print("[DATE Robust] finished with failures:")
        for ds, seed, contam, err in failures:
            print(f"[DATE Robust] dataset={ds} seed={seed} contam={contam} error={err}")
if __name__ == "__main__":
    main()

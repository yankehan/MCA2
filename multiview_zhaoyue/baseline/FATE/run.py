import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
def _save_one_dataset_excel(out_dir: str, dataset: str, row: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"FATE_{dataset}.xlsx")
    try:
        import pandas as pd
        pd.DataFrame([row]).to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        out_csv = os.path.join(out_dir, f"FATE_{dataset}.csv")
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return out_csv
def _save_one_dataset_excel_auprc(out_dir: str, dataset: str, row: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"auprc_FATE_{dataset}.xlsx")
    try:
        import pandas as pd
        pd.DataFrame([row]).to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        out_csv = os.path.join(out_dir, f"auprc_FATE_{dataset}.csv")
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return out_csv
def main():
    import argparse
    import statistics
    from tqdm import tqdm
    if __package__ is None or __package__ == "":
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from multiview_zhaoyue.baseline.FATE.trainer import run_fate_one_dataset
    else:
        from .trainer import run_fate_one_dataset
    parser = argparse.ArgumentParser(description="Run FATE baseline on multiview_zhaoyue datasets")
    parser.add_argument("--datasets", type=str, default="olid,covid_fake,liar2,hate_speech,email_spam,smsspam,bbc,movie_review,N24News,agnews")
    parser.add_argument("--device", type=str, default=os.environ.get("FATE_DEVICE", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的seed列表，例如 41,42,43；为空则使用 --seed")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--train_batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--few_shot_anomalies", type=int, default=10)
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
    multi_seed_mode = args.seeds is not None
    seeds_str = args.seeds.replace(" ", "") if args.seeds else ""
    for ds in tqdm(datasets, desc="datasets", leave=True):
        aurocs = []
        auprcs = []
        times = []
        seeds_ok = []
        for seed in seeds:
            try:
                t0 = time.time()
                auroc, auprc, total_time_sec = run_fate_one_dataset(
                    dataset_name=ds,
                    data_root=data_root,
                    out_dir=out_dir,
                    device=args.device,
                    seed=seed,
                    model_name=args.model_name,
                    max_seq_len=args.max_seq_len,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    num_epochs=args.num_epochs,
                    learning_rate=args.lr,
                    few_shot_anomalies=args.few_shot_anomalies,
                    attention_size=args.attention_size,
                    num_heads=args.num_heads,
                    topk_ratio=args.topk_ratio,
                    include_regularization=(not args.no_regularization),
                    num_workers=0,
                    suppress_internal_output=True,
                )
                _ = time.time() - t0
                row = {
                    "dataset": ds,
                    "auroc": float(auroc),
                    "total_run_time_sec": float(total_time_sec),
                    "seed": int(seed),
                    "few_shot_anomalies": int(args.few_shot_anomalies),
                    "model_name": str(args.model_name),
                    "max_seq_len": int(args.max_seq_len),
                    "train_batch_size": int(args.train_batch_size),
                    "eval_batch_size": int(args.eval_batch_size),
                    "num_epochs": int(args.num_epochs),
                    "lr": float(args.lr),
                    "attention_size": int(args.attention_size),
                    "num_heads": int(args.num_heads),
                    "topk_ratio": float(args.topk_ratio),
                    "regularization": bool(not args.no_regularization),
                    "device": str(args.device),
                }
                row_auprc = {
                    "dataset": ds,
                    "auprc": float(auprc),
                    "total_run_time_sec": float(total_time_sec),
                    "seed": int(seed),
                    "few_shot_anomalies": int(args.few_shot_anomalies),
                    "model_name": str(args.model_name),
                    "max_seq_len": int(args.max_seq_len),
                    "train_batch_size": int(args.train_batch_size),
                    "eval_batch_size": int(args.eval_batch_size),
                    "num_epochs": int(args.num_epochs),
                    "lr": float(args.lr),
                    "attention_size": int(args.attention_size),
                    "num_heads": int(args.num_heads),
                    "topk_ratio": float(args.topk_ratio),
                    "regularization": bool(not args.no_regularization),
                    "device": str(args.device),
                }
                if multi_seed_mode:
                    aurocs.append(float(auroc))
                    auprcs.append(float(auprc))
                    times.append(float(total_time_sec))
                    seeds_ok.append(int(seed))
                else:
                    out_key = f"{ds}_seed{seed}"
                    _save_one_dataset_excel(out_dir=out_dir, dataset=out_key, row=row)
                    _save_one_dataset_excel_auprc(out_dir=out_dir, dataset=out_key, row=row_auprc)
                print(f"[FATE] dataset={ds} seed={seed} AUROC={auroc:.6f} AUPRC={auprc:.6f} total_run_time_sec={total_time_sec:.2f}")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                failures.append((ds, seed, f"{type(e).__name__}: {e}"))
                print(f"[FATE] dataset={ds} seed={seed} FAILED: {type(e).__name__}: {e}")
        if multi_seed_mode and aurocs:
            auroc_mean = float(statistics.mean(aurocs))
            auroc_std = float(statistics.stdev(aurocs)) if len(aurocs) > 1 else 0.0
            auprc_mean = float(statistics.mean(auprcs))
            auprc_std = float(statistics.stdev(auprcs)) if len(auprcs) > 1 else 0.0
            time_mean = float(statistics.mean(times))
            time_std = float(statistics.stdev(times)) if len(times) > 1 else 0.0
            row = {
                "dataset": ds,
                "auroc_mean": auroc_mean,
                "auroc_std": auroc_std,
                "total_run_time_sec_mean": time_mean,
                "total_run_time_sec_std": time_std,
                "seeds": ",".join(str(s) for s in seeds),
                "n_success": int(len(aurocs)),
                "n_total": int(len(seeds)),
                "few_shot_anomalies": int(args.few_shot_anomalies),
                "model_name": str(args.model_name),
                "max_seq_len": int(args.max_seq_len),
                "train_batch_size": int(args.train_batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "num_epochs": int(args.num_epochs),
                "lr": float(args.lr),
                "attention_size": int(args.attention_size),
                "num_heads": int(args.num_heads),
                "topk_ratio": float(args.topk_ratio),
                "regularization": bool(not args.no_regularization),
                "device": str(args.device),
            }
            row_auprc = {
                "dataset": ds,
                "auprc_mean": auprc_mean,
                "auprc_std": auprc_std,
                "total_run_time_sec_mean": time_mean,
                "total_run_time_sec_std": time_std,
                "seeds": ",".join(str(s) for s in seeds),
                "n_success": int(len(auprcs)),
                "n_total": int(len(seeds)),
                "few_shot_anomalies": int(args.few_shot_anomalies),
                "model_name": str(args.model_name),
                "max_seq_len": int(args.max_seq_len),
                "train_batch_size": int(args.train_batch_size),
                "eval_batch_size": int(args.eval_batch_size),
                "num_epochs": int(args.num_epochs),
                "lr": float(args.lr),
                "attention_size": int(args.attention_size),
                "num_heads": int(args.num_heads),
                "topk_ratio": float(args.topk_ratio),
                "regularization": bool(not args.no_regularization),
                "device": str(args.device),
            }
            out_key = f"{ds}_{seeds_str}" if seeds_str else ds
            _save_one_dataset_excel(out_dir=out_dir, dataset=out_key, row=row)
            _save_one_dataset_excel_auprc(out_dir=out_dir, dataset=out_key, row=row_auprc)
            print(
                f"[FATE] dataset={ds} seeds={','.join(str(s) for s in seeds_ok)} "
                f"AUROC_mean={auroc_mean:.6f} AUROC_std={auroc_std:.6f} AUPRC_mean={auprc_mean:.6f} AUPRC_std={auprc_std:.6f}"
            )
    if failures:
        print("[FATE] finished with failures:")
        for ds, seed, err in failures:
            print(f"[FATE] dataset={ds} seed={seed} error={err}")
if __name__ == "__main__":
    main()

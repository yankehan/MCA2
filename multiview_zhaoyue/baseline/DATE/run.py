import os
import sys
import time
def _save_one_dataset_excel(out_dir: str, dataset: str, row: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"DATE_{dataset}.xlsx")
    try:
        import pandas as pd
        pd.DataFrame([row]).to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        out_csv = os.path.join(out_dir, f"DATE_{dataset}.csv")
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader()
            w.writerow(row)
        return out_csv
def _save_one_dataset_excel_auprc(out_dir: str, dataset: str, row: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"auprc_DATE_{dataset}.xlsx")
    try:
        import pandas as pd
        pd.DataFrame([row]).to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        out_csv = os.path.join(out_dir, f"auprc_DATE_{dataset}.csv")
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
        from multiview_zhaoyue.baseline.DATE.trainer import run_date_one_dataset
    else:
        from .trainer import run_date_one_dataset
    parser = argparse.ArgumentParser(description="Run DATE baseline on multiview_zhaoyue datasets")
    parser.add_argument("--datasets", type=str, default="olid,covid_fake,liar2,hate_speech,email_spam,smsspam,bbc,movie_review,N24News,agnews")
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
    multi_seed_mode = args.seeds is not None
    seeds_str = args.seeds.replace(" ", "") if args.seeds else ""
    for ds in tqdm(datasets, desc="datasets", leave=True):
        ds_epochs = int(EPOCHS_BY_DATASET.get(ds, args.num_train_epochs))
        ds_bs = int(BATCH_SIZE_BY_DATASET.get(ds, args.train_batch_size))
        aurocs = []
        auprcs = []
        times = []
        seeds_ok = []
        for seed in seeds:
            ok = False
            last_err = None
            retries = max(1, int(args.max_retries))
            for attempt in range(1, retries + 1):
                try:
                    auroc, auprc, total_time_sec = run_date_one_dataset(
                        dataset_name=ds,
                        data_root=data_root,
                        out_dir=out_dir,
                        device=args.device,
                        seed=int(seed),
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
                    row = {
                        "dataset": ds,
                        "auroc": float(auroc),
                        "total_run_time_sec": float(total_time_sec),
                        "seed": int(seed),
                    }
                    row_auprc = {
                        "dataset": ds,
                        "auprc": float(auprc),
                        "total_run_time_sec": float(total_time_sec),
                        "seed": int(seed),
                    }
                    if multi_seed_mode:
                        aurocs.append(float(auroc))
                        auprcs.append(float(auprc))
                        times.append(float(total_time_sec))
                        seeds_ok.append(int(seed))
                    else:
                        _save_one_dataset_excel(out_dir=out_dir, dataset=ds, row=row)
                        _save_one_dataset_excel_auprc(out_dir=out_dir, dataset=ds, row=row_auprc)
                    print(f"[DATE] dataset={ds} seed={seed} AUROC={auroc:.6f} AUPRC={auprc:.6f} total_run_time_sec={total_time_sec:.2f}")
                    ok = True
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_err = e
                    print(
                        f"[DATE] dataset={ds} seed={seed} attempt={attempt}/{retries} FAILED: {type(e).__name__}: {e}"
                    )
                    if attempt < retries:
                        try:
                            time.sleep(float(args.retry_sleep_sec))
                        except Exception:
                            pass
            if not ok:
                failures.append((ds, int(seed), repr(last_err)))
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
                "seq_len": int(args.seq_len),
                "num_train_epochs": int(ds_epochs),
                "train_batch_size": int(ds_bs),
                "max_lr": float(args.max_lr),
                "min_lr": float(args.min_lr),
                "warmup_steps": int(args.warmup_steps),
                "weight_decay": float(args.weight_decay),
                "disc_drop": float(args.disc_drop),
                "disc_hid_layers": int(args.disc_hid_layers),
                "disc_hid_size": int(args.disc_hid_size),
                "gen_hid_layers": int(args.gen_hid_layers),
                "gen_hid_size": int(args.gen_hid_size),
                "rtd_loss_weight": int(args.rtd_loss_weight),
                "rmd_loss_weight": int(args.rmd_loss_weight),
                "mlm_loss_weight": int(args.mlm_loss_weight),
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
                "seq_len": int(args.seq_len),
                "num_train_epochs": int(ds_epochs),
                "train_batch_size": int(ds_bs),
                "max_lr": float(args.max_lr),
                "min_lr": float(args.min_lr),
                "warmup_steps": int(args.warmup_steps),
                "weight_decay": float(args.weight_decay),
                "disc_drop": float(args.disc_drop),
                "disc_hid_layers": int(args.disc_hid_layers),
                "disc_hid_size": int(args.disc_hid_size),
                "gen_hid_layers": int(args.gen_hid_layers),
                "gen_hid_size": int(args.gen_hid_size),
                "rtd_loss_weight": int(args.rtd_loss_weight),
                "rmd_loss_weight": int(args.rmd_loss_weight),
                "mlm_loss_weight": int(args.mlm_loss_weight),
                "device": str(args.device),
            }
            out_key = f"{ds}_{seeds_str}" if seeds_str else ds
            _save_one_dataset_excel(out_dir=out_dir, dataset=out_key, row=row)
            _save_one_dataset_excel_auprc(out_dir=out_dir, dataset=out_key, row=row_auprc)
            print(
                f"[DATE] dataset={ds} seeds={','.join(str(s) for s in seeds_ok)} "
                f"AUROC_mean={auroc_mean:.6f} AUROC_std={auroc_std:.6f} AUPRC_mean={auprc_mean:.6f} AUPRC_std={auprc_std:.6f}"
            )
    if failures:
        print("[DATE] finished with failures:")
        for ds, seed, err in failures:
            print(f"[DATE] dataset={ds} seed={seed} error={err}")
if __name__ == "__main__":
    main()

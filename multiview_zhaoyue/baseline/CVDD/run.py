import os
import sys
import time
from typing import List, Optional
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
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
    from multiview_zhaoyue.baseline.CVDD.dataset import CVDDJsonlDataset
else:
    from .cvdd import CVDD
    from .dataset import CVDDJsonlDataset
def _save_all_datasets_excel_or_csv(out_dir: str, rows: List[dict]):
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, "CVDD_all_datasets.xlsx")
    out_csv = os.path.join(out_dir, "CVDD_all_datasets.csv")
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False)
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
def _save_all_datasets_excel_or_csv_auprc(out_dir: str, rows: List[dict]):
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, "auprc_CVDD_all_datasets.xlsx")
    out_csv = os.path.join(out_dir, "auprc_CVDD_all_datasets.csv")
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False)
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
def _save_one_dataset_excel_or_csv(out_dir: str, dataset: str, row: dict):
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"CVDD_{dataset}.xlsx")
    out_csv = os.path.join(out_dir, f"CVDD_{dataset}.csv")
    rows = [row]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerow(rows[0])
        return out_csv
def _save_one_dataset_excel_or_csv_auprc(out_dir: str, dataset: str, row: dict):
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"auprc_CVDD_{dataset}.xlsx")
    out_csv = os.path.join(out_dir, f"auprc_CVDD_{dataset}.csv")
    rows = [row]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False)
        return out_xlsx
    except Exception:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerow(rows[0])
        return out_csv
def run_one_dataset(
    dataset_name: str,
    data_root: str,
    device: str = "cuda",
    seed: int = 42,
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
):
    dataset = CVDDJsonlDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        min_freq=min_freq,
        max_vocab_size=max_vocab_size,
        max_len=max_len,
    )
    model = CVDD(ad_score="context_dist_mean")
    model.set_network(
        dataset=dataset,
        embedding_size=embedding_size,
        attention_size=attention_size,
        n_attention_heads=n_attention_heads,
        freeze_embedding=False,
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
        desc_prefix=f"[{dataset_name}] ",
    )
    model.test(
        dataset=dataset,
        device=device,
        show_progress=True,
        desc_prefix=f"[{dataset_name}] ",
    )
    t1 = time.time()
    return {
        "dataset": dataset_name,
        "auc": float(model.results.get("test_auc") or 0.0),
        "auprc": float(model.results.get("test_auprc") or 0.0),
        "total_time_sec": float(t1 - t0),
    }
def main(datasets: Optional[List[str]] = None):
    import argparse
    import statistics
    parser = argparse.ArgumentParser(description="Run CVDD baseline on multiview_zhaoyue datasets")
    parser.add_argument("--datasets", type=str, default="olid,covid_fake,liar2,hate_speech,email_spam,smsspam,bbc,movie_review,N24News,agnews")
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
    all_rows = []
    all_rows_auprc = []
    failures = []
    multi_seed_mode = args.seeds is not None
    seeds_str = args.seeds.replace(" ", "") if args.seeds else ""
    for ds in tqdm(datasets, desc="datasets", leave=True):
        if multi_seed_mode:
            aucs = []
            auprcs = []
            times = []
            seeds_ok = []
            for seed in seeds:
                try:
                    row_one = run_one_dataset(
                        dataset_name=ds,
                        data_root=data_root,
                        device=device,
                        seed=int(seed),
                    )
                    aucs.append(float(row_one.get("auc") or 0.0))
                    auprcs.append(float(row_one.get("auprc") or 0.0))
                    times.append(float(row_one.get("total_time_sec") or 0.0))
                    seeds_ok.append(int(seed))
                    print(
                        f"[CVDD] dataset={row_one.get('dataset')} seed={seed} auc={float(row_one.get('auc') or 0.0):.6f} "
                        f"auprc={float(row_one.get('auprc') or 0.0):.6f} "
                        f"total_time_sec={float(row_one.get('total_time_sec') or 0.0):.2f}"
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    failures.append((ds, int(seed), f"{type(e).__name__}: {e}"))
                    print(f"[CVDD] dataset={ds} seed={seed} FAILED: {type(e).__name__}: {e}")
            if aucs:
                auc_mean = float(statistics.mean(aucs))
                auc_std = float(statistics.stdev(aucs)) if len(aucs) > 1 else 0.0
                auprc_mean = float(statistics.mean(auprcs))
                auprc_std = float(statistics.stdev(auprcs)) if len(auprcs) > 1 else 0.0
                time_mean = float(statistics.mean(times))
                time_std = float(statistics.stdev(times)) if len(times) > 1 else 0.0
                row = {
                    "dataset": ds,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                    "total_time_sec_mean": time_mean,
                    "total_time_sec_std": time_std,
                    "seeds": ",".join(str(s) for s in seeds),
                    "n_success": int(len(aucs)),
                    "n_total": int(len(seeds)),
                }
                row_auprc = {
                    "dataset": ds,
                    "auprc_mean": auprc_mean,
                    "auprc_std": auprc_std,
                    "total_time_sec_mean": time_mean,
                    "total_time_sec_std": time_std,
                    "seeds": ",".join(str(s) for s in seeds),
                    "n_success": int(len(auprcs)),
                    "n_total": int(len(seeds)),
                }
                out_key = f"{ds}_{seeds_str}" if seeds_str else ds
                _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=out_key, row=row)
                _save_one_dataset_excel_or_csv_auprc(out_dir=out_dir, dataset=out_key, row=row_auprc)
                print(
                    f"[CVDD] dataset={ds} seeds={','.join(str(s) for s in seeds_ok)} "
                    f"AUC_mean={auc_mean:.6f} AUC_std={auc_std:.6f} "
                    f"AUPRC_mean={auprc_mean:.6f} AUPRC_std={auprc_std:.6f}"
                )
                all_rows.append(row)
                all_rows_auprc.append(row_auprc)
        else:
            result = run_one_dataset(
                dataset_name=ds,
                data_root=data_root,
                device=device,
                seed=seeds[0],
            )
            row = {
                "dataset": result.get("dataset"),
                "auc": result.get("auc"),
                "total_time_sec": result.get("total_time_sec"),
            }
            row_auprc = {
                "dataset": result.get("dataset"),
                "auprc": result.get("auprc"),
                "total_time_sec": result.get("total_time_sec"),
            }
            _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
            _save_one_dataset_excel_or_csv_auprc(out_dir=out_dir, dataset=ds, row=row_auprc)
            print(
                f"[CVDD] dataset={row.get('dataset')} auc={row.get('auc'):.6f} auprc={result.get('auprc'):.6f} total_time_sec={row.get('total_time_sec'):.2f}"
            )
            all_rows.append(row)
            all_rows_auprc.append(row_auprc)
    _save_all_datasets_excel_or_csv(out_dir=out_dir, rows=all_rows)
    _save_all_datasets_excel_or_csv_auprc(out_dir=out_dir, rows=all_rows_auprc)
    if failures:
        print("[CVDD] finished with failures:")
        for ds, seed, err in failures:
            print(f"[CVDD] dataset={ds} seed={seed} error={err}")
if __name__ == "__main__":
    main()

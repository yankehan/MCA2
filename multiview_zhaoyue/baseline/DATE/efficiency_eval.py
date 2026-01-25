import os
import sys
import time
from threading import Event, Thread
from typing import Any, Dict, List, Optional
try:
    import psutil
except Exception:
    psutil = None
import tracemalloc
class _MemoryMonitor:
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = float(interval_s)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._peak_bytes: int = 0
        self._tracemalloc_started: bool = False
    def start(self):
        if psutil is None:
            self._tracemalloc_started = True
            tracemalloc.start()
            return
        proc = psutil.Process(os.getpid())
        def _loop():
            while not self._stop.is_set():
                try:
                    rss = int(proc.memory_info().rss)
                    if rss > self._peak_bytes:
                        self._peak_bytes = rss
                except Exception:
                    pass
                time.sleep(self.interval_s)
        self._stop.clear()
        self._thread = Thread(target=_loop, daemon=True)
        self._thread.start()
    def stop(self) -> Optional[int]:
        if psutil is None:
            if not self._tracemalloc_started:
                return None
            try:
                peak = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()
                self._tracemalloc_started = False
            return int(peak)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return int(self._peak_bytes)
def _save_one_dataset_excel_or_csv(out_dir: str, dataset: str, row: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"{dataset}_efficiency.xlsx")
    out_csv = os.path.join(out_dir, f"{dataset}_efficiency.csv")
    try:
        import pandas as pd
        df = pd.DataFrame([row])
        df.to_excel(out_xlsx, index=False, engine='openpyxl')
        return out_xlsx
    except Exception:
        try:
            import pandas as pd
            df = pd.DataFrame([row])
            df.to_excel(out_xlsx, index=False)
            return out_xlsx
        except Exception:
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
        from multiview_zhaoyue.baseline.DATE.trainer import run_date_one_dataset_efficiency
    else:
        from .trainer import run_date_one_dataset_efficiency
    parser = argparse.ArgumentParser(description="Run DATE baseline efficiency evaluation")
    parser.add_argument("--datasets", type=str, default="olid,hate_speech,movie_review")
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
    for ds in tqdm(datasets, desc="datasets", leave=True):
        ds_epochs = int(EPOCHS_BY_DATASET.get(ds, args.num_train_epochs))
        ds_bs = int(BATCH_SIZE_BY_DATASET.get(ds, args.train_batch_size))
        train_times = []
        test_times = []
        total_times = []
        mems = []
        seeds_ok = []
        errors = []
        for seed in seeds:
            ok = False
            last_err = None
            retries = max(1, int(args.max_retries))
            for attempt in range(1, retries + 1):
                mem_monitor = _MemoryMonitor(interval_s=0.05)
                try:
                    mem_monitor.start()
                    auroc, train_time_sec, test_time_sec, total_time_sec = run_date_one_dataset_efficiency(
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
                    peak_bytes = mem_monitor.stop()
                    mem_mb = None if peak_bytes is None else float(peak_bytes) / (1024.0 * 1024.0)
                    if multi_seed_mode:
                        train_times.append(float(train_time_sec))
                        test_times.append(float(test_time_sec))
                        total_times.append(float(total_time_sec))
                        if mem_mb is not None:
                            mems.append(float(mem_mb))
                        seeds_ok.append(int(seed))
                    else:
                        row = {
                            '数据集名称': ds,
                            '算法名称': 'DATE',
                            '内存使用量(MB)': None if mem_mb is None else round(float(mem_mb), 2),
                            'train_time(s)': round(float(train_time_sec), 4),
                            'test_time(s)': round(float(test_time_sec), 4),
                            'total_time(s)': round(float(total_time_sec), 4),
                        }
                        out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                        print(
                            f"[DATE][EFF] dataset={ds} seed={seed} AUROC={auroc:.6f} "
                            f"memory_usage(MB)={row.get('内存使用量(MB)')} train_time(s)={row.get('train_time(s)')} "
                            f"test_time(s)={row.get('test_time(s)')} total_time(s)={row.get('total_time(s)')} saved={out_file}"
                        )
                    ok = True
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    last_err = e
                    try:
                        mem_monitor.stop()
                    except Exception:
                        pass
                    err = f"{type(e).__name__}: {e}"
                    errors.append(err)
                    print(f"[DATE][EFF] dataset={ds} seed={seed} attempt={attempt}/{retries} FAILED: {err}")
                    if attempt < retries:
                        try:
                            time.sleep(float(args.retry_sleep_sec))
                        except Exception:
                            pass
            if not ok:
                failures.append((ds, int(seed), repr(last_err)))
        if multi_seed_mode:
            if train_times:
                row = {
                    '数据集名称': ds,
                    '算法名称': 'DATE',
                    '内存使用量(MB)': round(float(statistics.mean(mems)), 2) if mems else None,
                    'train_time(s)': round(float(statistics.mean(train_times)), 4),
                    'test_time(s)': round(float(statistics.mean(test_times)), 4),
                    'total_time(s)': round(float(statistics.mean(total_times)), 4),
                }
                if errors:
                    row['Error'] = '; '.join(errors)
                out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                print(
                    f"[DATE][EFF] dataset={ds} seeds={','.join(str(s) for s in seeds_ok)} "
                    f"train_time(s)={row.get('train_time(s)')} test_time(s)={row.get('test_time(s)')} total_time(s)={row.get('total_time(s)')} saved={out_file}"
                )
            else:
                row = {
                    '数据集名称': ds,
                    '算法名称': 'DATE',
                    '内存使用量(MB)': None,
                    'train_time(s)': None,
                    'test_time(s)': None,
                    'total_time(s)': None,
                    'Error': '; '.join(errors) if errors else 'unknown error',
                }
                out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                print(f"[DATE][EFF] dataset={ds} saved={out_file}")
    if failures:
        print("[DATE][EFF] finished with failures:")
        for ds, seed, err in failures:
            print(f"[DATE][EFF] dataset={ds} seed={seed} error={err}")
if __name__ == "__main__":
    main()

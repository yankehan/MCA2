import os
import sys
import time
import warnings
from threading import Event, Thread
from typing import Any, Dict, List, Optional
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
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
        from multiview_zhaoyue.baseline.FATE.trainer import run_fate_one_dataset_efficiency
    else:
        from .trainer import run_fate_one_dataset_efficiency
    parser = argparse.ArgumentParser(description="Run FATE baseline efficiency evaluation")
    parser.add_argument("--datasets", type=str, default="olid,hate_speech,movie_review", help="逗号分隔的数据集列表；为空则跑默认10个")
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
    for ds in tqdm(datasets, desc="datasets", leave=True):
        train_times = []
        test_times = []
        total_times = []
        mems = []
        seeds_ok = []
        errors = []
        for seed in seeds:
            mem_monitor = _MemoryMonitor(interval_s=0.05)
            try:
                mem_monitor.start()
                auroc, train_time_sec, test_time_sec, total_time_sec = run_fate_one_dataset_efficiency(
                    dataset_name=ds,
                    data_root=data_root,
                    out_dir=out_dir,
                    device=args.device,
                    seed=int(seed),
                    model_name=args.model_name,
                    max_seq_len=int(args.max_seq_len),
                    train_batch_size=int(args.train_batch_size),
                    eval_batch_size=int(args.eval_batch_size),
                    num_epochs=int(args.num_epochs),
                    learning_rate=float(args.lr),
                    few_shot_anomalies=int(args.few_shot_anomalies),
                    attention_size=int(args.attention_size),
                    num_heads=int(args.num_heads),
                    topk_ratio=float(args.topk_ratio),
                    include_regularization=(not args.no_regularization),
                    num_workers=0,
                    suppress_internal_output=True,
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
                    train_r = round(float(train_time_sec), 4)
                    test_r = round(float(test_time_sec), 4)
                    total_r = round(float(train_r + test_r), 4)
                    row = {
                        '数据集名称': ds,
                        '算法名称': 'FATE',
                        '内存使用量(MB)': None if mem_mb is None else round(float(mem_mb), 2),
                        'train_time(s)': train_r,
                        'test_time(s)': test_r,
                        'total_time(s)': total_r,
                    }
                    out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                    print(
                        f"[FATE][EFF] dataset={ds} seed={seed} AUROC={auroc:.6f} "
                        f"memory_usage(MB)={row.get('内存使用量(MB)')} train_time(s)={row.get('train_time(s)')} "
                        f"test_time(s)={row.get('test_time(s)')} total_time(s)={row.get('total_time(s)')} saved={out_file}"
                    )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                try:
                    mem_monitor.stop()
                except Exception:
                    pass
                err = f"{type(e).__name__}: {e}"
                errors.append(err)
                failures.append((ds, int(seed), err))
                print(f"[FATE][EFF] dataset={ds} seed={seed} FAILED: {err}")
        if multi_seed_mode:
            if train_times:
                train_mean = float(statistics.mean(train_times))
                test_mean = float(statistics.mean(test_times))
                train_r = round(train_mean, 4)
                test_r = round(test_mean, 4)
                total_r = round(float(train_r + test_r), 4)
                row = {
                    '数据集名称': ds,
                    '算法名称': 'FATE',
                    '内存使用量(MB)': round(float(statistics.mean(mems)), 2) if mems else None,
                    'train_time(s)': train_r,
                    'test_time(s)': test_r,
                    'total_time(s)': total_r,
                }
                if errors:
                    row['Error'] = '; '.join(errors)
                out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                print(
                    f"[FATE][EFF] dataset={ds} seeds={','.join(str(s) for s in seeds_ok)} "
                    f"train_time(s)={row.get('train_time(s)')} test_time(s)={row.get('test_time(s)')} "
                    f"total_time(s)={row.get('total_time(s)')} saved={out_file}"
                )
            else:
                row = {
                    '数据集名称': ds,
                    '算法名称': 'FATE',
                    '内存使用量(MB)': None,
                    'train_time(s)': None,
                    'test_time(s)': None,
                    'total_time(s)': None,
                    'Error': '; '.join(errors) if errors else 'unknown error',
                }
                out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row)
                print(f"[FATE][EFF] dataset={ds} saved={out_file}")
    if failures:
        print("[FATE][EFF] finished with failures:")
        for ds, seed, err in failures:
            print(f"[FATE][EFF] dataset={ds} seed={seed} error={err}")
if __name__ == "__main__":
    main()

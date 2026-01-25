import os
import sys
import time
from threading import Event, Thread
from typing import Any, Dict, List, Optional
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
def _save_one_dataset_excel_or_csv(out_dir: str, dataset: str, row: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_xlsx = os.path.join(out_dir, f"{dataset}_efficiency.xlsx")
    out_csv = os.path.join(out_dir, f"{dataset}_efficiency.csv")
    rows = [row]
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False, engine='openpyxl')
        return out_xlsx
    except Exception:
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
def run_one_dataset_efficiency(
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
) -> Dict[str, Any]:
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
    mem_monitor = _MemoryMonitor(interval_s=0.05)
    mem_monitor.start()
    t_total0 = time.time()
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
    train_time = float(time.time() - t0)
    t0 = time.time()
    model.test(
        dataset=dataset,
        device=device,
        show_progress=True,
        desc_prefix=f"[{dataset_name}] ",
    )
    test_time = float(time.time() - t0)
    total_time = float(time.time() - t_total0)
    peak_bytes = mem_monitor.stop()
    mem_mb = None if peak_bytes is None else float(peak_bytes) / (1024.0 * 1024.0)
    return {
        '数据集名称': dataset_name,
        '算法名称': 'CVDD',
        '内存使用量(MB)': None if mem_mb is None else round(mem_mb, 2),
        'train_time(s)': round(train_time, 4),
        'test_time(s)': round(test_time, 4),
        'total_time(s)': round(total_time, 4),
    }
def main(datasets: Optional[List[str]] = None):
    import argparse
    parser = argparse.ArgumentParser(description="CVDD baseline efficiency evaluation")
    parser.add_argument("--datasets", type=str, default="olid,hate_speech,movie_review")
    parser.add_argument("--device", type=str, default=os.environ.get("CVDD_DEVICE", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="逗号分隔的seed列表，例如 41,42,43；为空则使用 --seed")
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--attention_size", type=int, default=150)
    parser.add_argument("--n_attention_heads", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_p", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab_size", type=int, default=50000)
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
        rows_ok: List[Dict[str, Any]] = []
        errors: List[str] = []
        for seed in seeds:
            try:
                row = run_one_dataset_efficiency(
                    dataset_name=ds,
                    data_root=data_root,
                    device=device,
                    seed=int(seed),
                    embedding_size=int(args.embedding_size),
                    attention_size=int(args.attention_size),
                    n_attention_heads=int(args.n_attention_heads),
                    n_epochs=int(args.n_epochs),
                    batch_size=int(args.batch_size),
                    lr=float(args.lr),
                    lambda_p=float(args.lambda_p),
                    max_len=int(args.max_len),
                    min_freq=int(args.min_freq),
                    max_vocab_size=int(args.max_vocab_size),
                )
                rows_ok.append(row)
                print(
                    f"[CVDD][EFF] dataset={ds} seed={seed} memory_usage(MB)={row.get('内存使用量(MB)')} "
                    f"train_time(s)={row.get('train_time(s)')} test_time(s)={row.get('test_time(s)')} total_time(s)={row.get('total_time(s)')}"
                )
            except KeyboardInterrupt:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                errors.append(err)
                print(f"[CVDD][EFF] dataset={ds} seed={seed} FAILED: {err}")
        if len(rows_ok) == 0:
            row_out = {
                '数据集名称': ds,
                '算法名称': 'CVDD',
                '内存使用量(MB)': None,
                'train_time(s)': None,
                'test_time(s)': None,
                'total_time(s)': None,
                'Error': '; '.join(errors) if errors else 'unknown error',
            }
        elif len(rows_ok) == 1:
            row_out = rows_ok[0]
            if errors:
                row_out = dict(row_out)
                row_out['Error'] = '; '.join(errors)
        else:
            mems = [r.get('内存使用量(MB)') for r in rows_ok if r.get('内存使用量(MB)') is not None]
            trains = [r.get('train_time(s)') for r in rows_ok if r.get('train_time(s)') is not None]
            tests = [r.get('test_time(s)') for r in rows_ok if r.get('test_time(s)') is not None]
            totals = [r.get('total_time(s)') for r in rows_ok if r.get('total_time(s)') is not None]
            def _mean(xs: List[float]) -> Optional[float]:
                if not xs:
                    return None
                return float(sum(xs) / len(xs))
            row_out = {
                '数据集名称': ds,
                '算法名称': 'CVDD',
                '内存使用量(MB)': None if _mean(mems) is None else round(float(_mean(mems)), 2),
                'train_time(s)': None if _mean(trains) is None else round(float(_mean(trains)), 4),
                'test_time(s)': None if _mean(tests) is None else round(float(_mean(tests)), 4),
                'total_time(s)': None if _mean(totals) is None else round(float(_mean(totals)), 4),
            }
            if errors:
                row_out['Error'] = '; '.join(errors)
        out_file = _save_one_dataset_excel_or_csv(out_dir=out_dir, dataset=ds, row=row_out)
        print(f"[CVDD][EFF] saved: {out_file}")
if __name__ == "__main__":
    main()

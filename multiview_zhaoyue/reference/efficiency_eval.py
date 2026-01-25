import argparse
import logging
import os
import random
import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from pyod.models.lof import LOF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
try:
    from pyod.models.so_gaal_new import SO_GAAL
except Exception:
    from pyod.models.so_gaal import SO_GAAL
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.lunar import LUNAR
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"随机数种子已设置为: {seed}")
def _save_results_df(output_dir: str, filename_no_ext: str, df: pd.DataFrame) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_xlsx = os.path.join(output_dir, f"{filename_no_ext}.xlsx")
    try:
        df.to_excel(out_xlsx, index=False, engine='openpyxl')
        logging.info(f"\n✓ 结果已保存到: {out_xlsx}")
        return out_xlsx
    except Exception:
        out_csv = os.path.join(output_dir, f"{filename_no_ext}.csv")
        df.to_csv(out_csv, index=False)
        logging.info(f"\n✓ 结果已保存到: {out_csv}")
        return out_csv
try:
    import psutil
except Exception:
    psutil = None
import tracemalloc
@dataclass
class EfficiencyMetrics:
    memory_usage_mb: Optional[float]
    train_time_s: Optional[float]
    test_time_s: Optional[float]
    total_time_s: Optional[float]
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
class UnifiedEfficiencyBenchmark:
    def __init__(self, embeddings_dir='../embeddings', data_dir='../data', output_dir='./', seed: int = 42):
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.random_state = int(seed)
        self.algorithms = {
            'LOF': self._build_lof,
            'DeepSVDD': self._build_deepsvdd,
            'ECOD': self._build_ecod,
            'IForest': self._build_iforest,
            'SO-GAAL': self._build_sogaal,
            'AE': self._build_autoencoder,
            'VAE': self._build_vae,
            'LUNAR': self._build_lunar,
        }
        self.results: List[Dict[str, Any]] = []
    def load_embeddings(self, dataset_name: str, model_name: str, split: str = 'train') -> np.ndarray:
        file_name = f"{model_name}_{dataset_name}_{split}.npy"
        file_path = os.path.join(
            self.embeddings_dir,
            dataset_name,
            f"{dataset_name}-{split}",
            file_name
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"嵌入文件不存在: {file_path}")
        embeddings = np.load(file_path)
        logging.info(f"加载嵌入: {file_path}, shape: {embeddings.shape}")
        return embeddings
    def load_labels(self, dataset_name: str, split: str = 'train') -> np.ndarray:
        file_path = os.path.join(self.data_dir, f"{dataset_name}_{split}_data.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"标签文件不存在: {file_path}")
        labels = pd.read_json(file_path, lines=True)['label'].values
        logging.info(f"加载标签: {file_path}, 样本数: {len(labels)}, 异常数: {labels.sum()}")
        return labels
    def clean_features(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0, posinf=1e38, neginf=-1e38)
        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        X = np.clip(X, min_float32, max_float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.astype(np.float32)
        return X
    def clean_labels(self, y: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y, nan=0.0, posinf=1e38, neginf=-1e38)
        y = y.astype(np.float32)
        return y
    def _build_lof(self, X_train: np.ndarray):
        return LOF()
    def _build_deepsvdd(self, X_train: np.ndarray):
        n_features = int(X_train.shape[1])
        return DeepSVDD(
            n_features=n_features,
            use_ae=False,
            epochs=5,
            contamination=0.1,
            random_state=int(self.random_state)
        )
    def _build_ecod(self, X_train: np.ndarray):
        return ECOD()
    def _build_iforest(self, X_train: np.ndarray):
        return IForest()
    def _build_sogaal(self, X_train: np.ndarray):
        return SO_GAAL(epoch_num=30, contamination=0.1, verbose=0)
    def _build_autoencoder(self, X_train: np.ndarray):
        return AutoEncoder(epoch_num=30, contamination=0.1)
    def _build_vae(self, X_train: np.ndarray):
        return VAE(epoch_num=30, contamination=0.1, beta=0.8, capacity=0.2)
    def _build_lunar(self, X_train: np.ndarray):
        return LUNAR()
    def evaluate_efficiency(self, algorithm_name: str, dataset_name: str, model_name: str) -> dict:
        logging.info(f"\n{'='*80}")
        logging.info(f"效率评测: {algorithm_name} + {model_name} + {dataset_name}")
        logging.info(f"{'='*80}")
        try:
            X_train = self.load_embeddings(dataset_name, model_name, 'train')
            X_test = self.load_embeddings(dataset_name, model_name, 'test')
            y_train = self.load_labels(dataset_name, 'train')
            y_test = self.load_labels(dataset_name, 'test')
            X_train = self.clean_features(X_train)
            X_test = self.clean_features(X_test)
            y_train = self.clean_labels(y_train)
            y_test = self.clean_labels(y_test)
            build_func = self.algorithms[algorithm_name]
            clf = build_func(X_train)
            mem_monitor = _MemoryMonitor(interval_s=0.05)
            mem_monitor.start()
            total_start = time.perf_counter()
            train_time = None
            test_time = None
            if hasattr(clf, 'fit'):
                t0 = time.perf_counter()
                clf.fit(X_train)
                train_time = time.perf_counter() - t0
            if hasattr(clf, 'decision_function'):
                t0 = time.perf_counter()
                _ = clf.decision_function(X_test)
                test_time = time.perf_counter() - t0
            total_time = time.perf_counter() - total_start
            peak_bytes = mem_monitor.stop()
            memory_usage_mb = None
            if peak_bytes is not None:
                memory_usage_mb = peak_bytes / (1024 * 1024)
            metrics = EfficiencyMetrics(
                memory_usage_mb=None if memory_usage_mb is None else round(float(memory_usage_mb), 2),
                train_time_s=None if train_time is None else round(float(train_time), 4),
                test_time_s=None if test_time is None else round(float(test_time), 4),
                total_time_s=round(float(total_time), 4),
            )
            logging.info(
                f"✓ 完成 - 内存峰值(MB): {metrics.memory_usage_mb}, "
                f"train: {metrics.train_time_s}s, test: {metrics.test_time_s}s, total: {metrics.total_time_s}s"
            )
            return {
                'dataset': dataset_name,
                'model': model_name,
                'algorithm': algorithm_name,
                'memory_usage': metrics.memory_usage_mb,
                'train_time': metrics.train_time_s,
                'test_time': metrics.test_time_s,
                'total_time': metrics.total_time_s,
            }
        except Exception as e:
            logging.error(f"✗ 错误: {str(e)}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'algorithm': algorithm_name,
                'memory_usage': None,
                'train_time': None,
                'test_time': None,
                'total_time': None,
                'Error': str(e)
            }
    def run(self, dataset_name: str, model_name: str, algorithms: Optional[List[str]] = None):
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        total_runs = len(algorithms)
        logging.info(f"\n{'#'*80}")
        logging.info(f"开始效率评测")
        logging.info(f"数据集: {dataset_name}")
        logging.info(f"模型: {model_name}")
        logging.info(f"算法: {algorithms}")
        logging.info(f"总运行次数: {total_runs}")
        logging.info(f"{'#'*80}\n")
        for idx, algorithm in enumerate(algorithms, 1):
            logging.info(f"\n进度: [{idx}/{total_runs}]")
            result = self.evaluate_efficiency(algorithm, dataset_name, model_name)
            self.results.append(result)
    def save_results(self, dataset_name: str, model_name: str) -> str:
        df = pd.DataFrame(self.results)
        column_order = ['dataset', 'model', 'algorithm', 'memory_usage', 'train_time', 'test_time', 'total_time']
        if 'Error' in df.columns:
            column_order.append('Error')
        df = df[[col for col in column_order if col in df.columns]]
        df = df.rename(
            columns={
                'memory_usage': 'memory_usage(MB)',
                'train_time': 'train_time(s)',
                'test_time': 'test_time(s)',
                'total_time': 'total_time(s)',
            }
        )
        return _save_results_df(output_dir=self.output_dir, filename_no_ext=f"{dataset_name}_{model_name}_efficiency", df=df)
def main():
    parser = argparse.ArgumentParser(
        description='效率分析实验 - 内存/训练时间/测试时间/总时间',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python efficiency_eval.py --datasets bbc,email_spam --model bert
  python efficiency_eval.py --datasets bbc --model bert --algorithms LOF,IForest
"""
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称')
    parser.add_argument('--datasets', type=str,
                        default="olid,hate_speech,movie_review",
                        help='逗号分隔的数据集列表；若提供则优先使用该参数')
    parser.add_argument('--model', type=str, required=True,
                        help='嵌入模型名称 (bert, llama, minilm, openai_ada, openai_large, openai_small, qwen, stella)')
    parser.add_argument('--algorithms', type=str, default=None,
                        help='逗号分隔的算法列表；为空则评测全部算法')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子 (默认: 42)')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='输出目录 (默认: ./)')
    args = parser.parse_args()
    model_name = args.model
    seed = int(args.seed)
    set_random_seed(seed)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        if args.dataset is None:
            raise ValueError("必须提供 --dataset 或 --datasets")
        datasets = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
    algorithms = None
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    for dataset_name in datasets:
        print("\n" + "=" * 80)
        print("效率分析实验")
        print(f"数据集: {dataset_name}")
        print(f"模型: {model_name}")
        print(f"随机种子: {seed}")
        if algorithms is None:
            print("算法: 全部8种算法 (LOF, DeepSVDD, ECOD, IForest, SO-GAAL, AE, VAE, LUNAR)")
        else:
            print(f"算法: {algorithms}")
        if psutil is None:
            print("注意: 未检测到 psutil，将使用 tracemalloc 统计 memory_usage(建议 pip install psutil 获取RSS峰值)")
        print("=" * 80)
        benchmark = UnifiedEfficiencyBenchmark(
            embeddings_dir='../../embeddings',
            data_dir='../../data',
            output_dir=str(args.output_dir),
            seed=seed
        )
        benchmark.run(
            dataset_name=dataset_name,
            model_name=model_name,
            algorithms=algorithms
        )
        output_file = benchmark.save_results(dataset_name, model_name)
        print("\n" + "=" * 80)
        print("效率评测完成！")
        print(f"结果已保存到: {output_file}")
        print("=" * 80)
if __name__ == '__main__':
    main()

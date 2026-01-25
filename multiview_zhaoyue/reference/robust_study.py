import argparse
import logging
import os
import random
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.lunar import LUNAR
from pyod.models.so_gaal_new import SO_GAAL
from pyod.models.vae import VAE
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_EMBEDDINGS_DIR = os.path.join(PROJECT_ROOT, 'embeddings')
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ROBUST_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
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
class RobustStudyBenchmark:
    def __init__(self, embeddings_dir: str = None, data_dir: str = None, output_dir: str = './', seed: int = 42):
        self.embeddings_dir = DEFAULT_EMBEDDINGS_DIR if embeddings_dir is None else embeddings_dir
        self.data_dir = DEFAULT_DATA_DIR if data_dir is None else data_dir
        self.output_dir = SCRIPT_DIR if output_dir == './' else output_dir
        self.random_state = int(seed)
        self.algorithms = {
            'LOF': self._run_lof,
            'DeepSVDD': self._run_deepsvdd,
            'ECOD': self._run_ecod,
            'IForest': self._run_iforest,
            'SO-GAAL': self._run_sogaal,
            'AE': self._run_autoencoder,
            'VAE': self._run_vae,
            'LUNAR': self._run_lunar,
        }
        self.results = []
    def load_embeddings(self, dataset_name: str, model_name: str, split: str = 'train') -> np.ndarray:
        file_name = f"{model_name}_{dataset_name}_{split}.npy"
        file_path = os.path.join(
            self.embeddings_dir,
            dataset_name,
            f"{dataset_name}-{split}",
            file_name,
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
        logging.info(f"加载标签: {file_path}, 样本数: {len(labels)}, 异常数: {int(labels.sum())}")
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
    def _run_lof(self, X_train, X_test, y_test):
        clf = LOF()
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_deepsvdd(self, X_train, X_test, y_test):
        n_features = X_train.shape[1]
        clf = DeepSVDD(
            n_features=n_features,
            use_ae=False,
            epochs=5,
            contamination=0.1,
            random_state=int(self.random_state),
        )
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_ecod(self, X_train, X_test, y_test):
        clf = ECOD()
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_iforest(self, X_train, X_test, y_test):
        clf = IForest()
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_sogaal(self, X_train, X_test, y_test):
        clf = SO_GAAL(epoch_num=30, contamination=0.1, verbose=0)
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_autoencoder(self, X_train, X_test, y_test):
        clf = AutoEncoder(epoch_num=30, contamination=0.1)
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_vae(self, X_train, X_test, y_test):
        clf = VAE(epoch_num=30, contamination=0.1, beta=0.8, capacity=0.2)
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _run_lunar(self, X_train, X_test, y_test):
        clf = LUNAR()
        clf.fit(X_train)
        return clf.decision_function(X_test)
    def _build_robust_splits(self, dataset_name: str, model_name: str, anomaly_train_ratio: float, seed: int):
        X_train_base = self.load_embeddings(dataset_name, model_name, 'train')
        y_train_base = self.load_labels(dataset_name, 'train')
        X_test_base = self.load_embeddings(dataset_name, model_name, 'test')
        y_test_base = self.load_labels(dataset_name, 'test')
        y_train_base = self.clean_labels(y_train_base)
        y_test_base = self.clean_labels(y_test_base)
        train_norm_idx = np.where(y_train_base == 0)[0]
        test_norm_idx = np.where(y_test_base == 0)[0]
        test_anom_idx = np.where(y_test_base == 1)[0]
        n_test_anom = int(test_anom_idx.shape[0])
        rng = np.random.RandomState(int(seed))
        n_test_anom_fixed = int(0.8 * n_test_anom)
        perm = rng.permutation(n_test_anom) if n_test_anom > 0 else np.array([], dtype=np.int64)
        test_pos = perm[:n_test_anom_fixed]
        rest_pos = perm[n_test_anom_fixed:]
        test_anom_fixed_idx = test_anom_idx[test_pos] if n_test_anom_fixed > 0 else test_anom_idx[:0]
        train_anom_pool_idx = test_anom_idx[rest_pos] if rest_pos.size > 0 else test_anom_idx[:0]
        n_train_anom = int(float(anomaly_train_ratio) * n_test_anom)
        if n_train_anom > int(train_anom_pool_idx.shape[0]):
            n_train_anom = int(train_anom_pool_idx.shape[0])
        if n_train_anom > 0:
            train_anom_perm = rng.permutation(train_anom_pool_idx.shape[0])
            train_anom_idx = train_anom_pool_idx[train_anom_perm[:n_train_anom]]
        else:
            train_anom_idx = train_anom_pool_idx[:0]
        X_train = X_train_base[train_norm_idx]
        if train_anom_idx.shape[0] > 0:
            X_train = np.concatenate([X_train, X_test_base[train_anom_idx]], axis=0)
        X_test = X_test_base[test_norm_idx]
        y_test = y_test_base[test_norm_idx]
        if test_anom_fixed_idx.shape[0] > 0:
            X_test = np.concatenate([X_test, X_test_base[test_anom_fixed_idx]], axis=0)
            y_test = np.concatenate([y_test, y_test_base[test_anom_fixed_idx]], axis=0)
        stats = {
            'train_normal': int(train_norm_idx.shape[0]),
            'train_anomaly': int(train_anom_idx.shape[0]),
            'test_normal': int(test_norm_idx.shape[0]),
            'test_anomaly': int(test_anom_fixed_idx.shape[0]),
            'total_anomaly': int(n_test_anom),
        }
        return X_train, X_test, y_test, stats
    def evaluate_algorithm_once(self, algorithm_name: str, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        logging.info(f"\n{'=' * 80}")
        logging.info(f"运行: {algorithm_name}")
        logging.info(f"{'=' * 80}")
        try:
            X_train = self.clean_features(X_train)
            X_test = self.clean_features(X_test)
            y_test = self.clean_labels(y_test)
            start_time = time.time()
            y_test_scores = self.algorithms[algorithm_name](X_train, X_test, y_test)
            elapsed_time = time.time() - start_time
            roc_auc = roc_auc_score(y_test, y_test_scores)
            result = {
                'Algorithm': algorithm_name,
                'AUC': round(float(roc_auc), 4),
                'Time(s)': round(float(elapsed_time), 2),
            }
            logging.info(f"✓ 完成 - AUC: {roc_auc:.4f}, 时间: {elapsed_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"✗ 错误: {str(e)}")
            return {
                'Algorithm': algorithm_name,
                'AUC': None,
                'Time(s)': None,
                'Error': str(e),
            }
    def run_robust_study_for_dataset(self, dataset_name: str, model_name: str, algorithms=None, seed: int = 42):
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        self.results = []
        for contam_ratio in ROBUST_LEVELS:
            set_random_seed(int(seed))
            logging.info("\n" + "#" * 80)
            logging.info(
                f"Robust setting: train 异常={int(contam_ratio * 100)}% (占总异常) | "
                f"test 异常=80% (固定，且与训练异常不重叠)"
            )
            logging.info("#" * 80)
            X_train, X_test, y_test, stats = self._build_robust_splits(
                dataset_name=dataset_name,
                model_name=model_name,
                anomaly_train_ratio=float(contam_ratio),
                seed=int(seed),
            )
            for alg in algorithms:
                one = self.evaluate_algorithm_once(alg, X_train, X_test, y_test)
                one.update(
                    {
                        'Dataset': dataset_name,
                        'Model': model_name,
                        'Seed': int(seed),
                        'Contam_Ratio': float(contam_ratio),
                        'Train_Anom_Pct': int(contam_ratio * 100),
                        'Contam_Tag': f"train_anom_{int(contam_ratio * 100)}%",
                        'Train_Normal': stats['train_normal'],
                        'Train_Anomaly': stats['train_anomaly'],
                        'Test_Normal': stats['test_normal'],
                        'Test_Anomaly': stats['test_anomaly'],
                    }
                )
                self.results.append(one)
        return pd.DataFrame(self.results)
def main():
    parser = argparse.ArgumentParser(
        description='Robust study 实验（reference benchmark / 8 algorithms）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--dataset', type=str, default=None, help='单个数据集名称')
    parser.add_argument(
        '--datasets',
        type=str,
        default="covid_fake,bbc,email_spam",
        help='逗号分隔的数据集列表；若提供则优先使用该参数',
    )
    parser.add_argument('--model', type=str, required=True, help='嵌入模型名称 (bert, llama, minilm, openai_ada, openai_large, openai_small, qwen, stella)')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子 (默认: 42)')
    parser.add_argument(
        '--algorithms',
        type=str,
        default=None,
        help='逗号分隔算法列表（默认全部）：LOF,DeepSVDD,ECOD,IForest,SO-GAAL,AE,VAE,LUNAR',
    )
    args = parser.parse_args()
    model_name = args.model
    seed = int(args.seed)
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    else:
        if args.dataset is None:
            raise ValueError('必须提供 --dataset 或 --datasets')
        datasets = [d.strip() for d in str(args.dataset).split(',') if d.strip()]
    algorithms = None
    if args.algorithms is not None:
        algorithms = [a.strip() for a in args.algorithms.split(',') if a.strip()]
    out_dir = SCRIPT_DIR
    for dataset_name in datasets:
        logging.info("\n" + "=" * 80)
        logging.info(f"Robust study 开始")
        logging.info(f"数据集: {dataset_name}")
        logging.info(f"模型: {model_name}")
        logging.info(f"随机种子: {seed}")
        logging.info("=" * 80)
        bench = RobustStudyBenchmark(
            embeddings_dir=DEFAULT_EMBEDDINGS_DIR,
            data_dir=DEFAULT_DATA_DIR,
            output_dir=out_dir,
            seed=int(seed),
        )
        df = bench.run_robust_study_for_dataset(
            dataset_name=dataset_name,
            model_name=model_name,
            algorithms=algorithms,
            seed=int(seed),
        )
        expected_levels = [int(x * 100) for x in ROBUST_LEVELS]
        summary_df = df.pivot_table(
            index=['Dataset', 'Model', 'Algorithm'],
            columns='Train_Anom_Pct',
            values='AUC',
            aggfunc='first',
        )
        for lvl in expected_levels:
            if lvl not in summary_df.columns:
                summary_df[lvl] = np.nan
        summary_df = summary_df[expected_levels]
        col_rename = {lvl: f"{lvl}%异常AUC" for lvl in expected_levels}
        summary_df = summary_df.rename(columns=col_rename).reset_index()
        summary_df = summary_df.rename(
            columns={
                'Dataset': '数据集名称',
                'Model': '使用的Model',
                'Algorithm': '使用的算法',
            }
        )
        algo_order = list(bench.algorithms.keys())
        summary_df['使用的算法'] = pd.Categorical(summary_df['使用的算法'], categories=algo_order, ordered=True)
        summary_df = summary_df.sort_values(['数据集名称', '使用的Model', '使用的算法']).reset_index(drop=True)
        filename_no_ext = f"{dataset_name}_{model_name}_robust_study_seed_{seed}"
        _save_results_df(output_dir=out_dir, filename_no_ext=filename_no_ext, df=summary_df)
if __name__ == '__main__':
    main()

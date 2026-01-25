import numpy as np
import pandas as pd
import os
import sys
import time
import logging
import argparse
import random
import torch
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')
from pyod.models.lof import LOF
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.so_gaal_new import SO_GAAL
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.lunar import LUNAR
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to: {seed}")
def _save_results_df(output_dir: str, filename_no_ext: str, df: pd.DataFrame) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_xlsx = os.path.join(output_dir, f"{filename_no_ext}.xlsx")
    try:
        df.to_excel(out_xlsx, index=False, engine='openpyxl')
        logging.info(f"\n✓ Results saved to: {out_xlsx}")
        return out_xlsx
    except Exception:
        out_csv = os.path.join(output_dir, f"{filename_no_ext}.csv")
        df.to_csv(out_csv, index=False)
        logging.info(f"\n✓ Results saved to: {out_csv}")
        return out_csv
class UnifiedBenchmark:
    def __init__(self, embeddings_dir='../embeddings', data_dir='../data', output_dir='./', seed: int = 42):
        self.embeddings_dir = embeddings_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.random_state = int(seed)
        self.algorithms = {
            'LOF': self._run_lof,
            'DeepSVDD': self._run_deepsvdd,
            'ECOD': self._run_ecod,
            'IForest': self._run_iforest,
            'SO-GAAL': self._run_sogaal,
            'AE': self._run_autoencoder,
            'VAE': self._run_vae,
            'LUNAR': self._run_lunar
        }
        self.model_name_mapping = {
            'bert': 'bert',
            'llama': 'llama',
            'minilm': 'minilm',
            'openai_ada': 'openai_ada',
            'openai_large': 'openai_large',
            'openai_small': 'openai_small',
            'qwen': 'qwen',
            'stella': 'stella'
        }
        self.results = []
    def load_embeddings(self, dataset_name, model_name, split='train'):
        file_name = f"{model_name}_{dataset_name}_{split}.npy"
        file_path = os.path.join(
            self.embeddings_dir,
            dataset_name,
            f"{dataset_name}-{split}",
            file_name
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        embeddings = np.load(file_path)
        logging.info(f"Loaded embeddings: {file_path}, shape: {embeddings.shape}")
        return embeddings
    def load_labels(self, dataset_name, split='train'):
        file_path = os.path.join(self.data_dir, f"{dataset_name}_{split}_data.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Label file not found: {file_path}")
        labels = pd.read_json(file_path, lines=True)['label'].values
        logging.info(f"Loaded labels: {file_path}, samples: {len(labels)}, anomalies: {labels.sum()}")
        return labels
    def clean_features(self, X):
        X = np.nan_to_num(X, nan=0.0, posinf=1e38, neginf=-1e38)
        max_float32 = np.finfo(np.float32).max
        min_float32 = np.finfo(np.float32).min
        X = np.clip(X, min_float32, max_float32)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.astype(np.float32)
        return X
    def clean_labels(self, y):
        y = np.nan_to_num(y, nan=0.0, posinf=1e38, neginf=-1e38)
        y = y.astype(np.float32)
        return y
    def _run_lof(self, X_train, X_test, y_test):
        clf = LOF()
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_deepsvdd(self, X_train, X_test, y_test):
        n_features = X_train.shape[1]
        clf = DeepSVDD(
            n_features=n_features,
            use_ae=False,
            epochs=5,
            contamination=0.1,
            random_state=int(self.random_state)
        )
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_ecod(self, X_train, X_test, y_test):
        clf = ECOD()
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_iforest(self, X_train, X_test, y_test):
        clf = IForest()
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_sogaal(self, X_train, X_test, y_test):
        clf = SO_GAAL(epoch_num=30, contamination=0.1, verbose=0)
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_autoencoder(self, X_train, X_test, y_test):
        clf = AutoEncoder(epoch_num=30, contamination=0.1)
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_vae(self, X_train, X_test, y_test):
        clf = VAE(epoch_num=30, contamination=0.1, beta=0.8, capacity=0.2)
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def _run_lunar(self, X_train, X_test, y_test):
        clf = LUNAR()
        clf.fit(X_train)
        y_test_scores = clf.decision_function(X_test)
        return y_test_scores
    def evaluate_algorithm(self, algorithm_name, dataset_name, model_name):
        logging.info(f"\n{'='*80}")
        logging.info(f"Running: {algorithm_name} + {model_name} + {dataset_name}")
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
            start_time = time.time()
            algorithm_func = self.algorithms[algorithm_name]
            y_test_scores = algorithm_func(X_train, X_test, y_test)
            elapsed_time = time.time() - start_time
            roc_auc = roc_auc_score(y_test, y_test_scores)
            precision, recall, _ = precision_recall_curve(y_test, y_test_scores)
            auprc = auc(recall, precision)
            result = {
                'Algorithm': algorithm_name,
                'AUC': round(roc_auc, 4),
                'AUPRC': round(auprc, 4),
                'Time(s)': round(elapsed_time, 2)
            }
            logging.info(f"✓ Completed - AUC: {roc_auc:.4f} AUPRC: {auprc:.4f}, Time: {elapsed_time:.2f}s")
            return result
        except Exception as e:
            logging.error(f"✗ Error: {str(e)}")
            return {
                'Algorithm': algorithm_name,
                'AUC': None,
                'AUPRC': None,
                'Time(s)': None,
                'Error': str(e)
            }
    def run_benchmark(self, dataset_name, model_name, algorithms=None):
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        total_runs = len(algorithms)
        logging.info(f"\n{'#'*80}")
        logging.info(f"Starting benchmark test")
        logging.info(f"Dataset: {dataset_name}")
        logging.info(f"Model: {model_name}")
        logging.info(f"Algorithms: {algorithms}")
        logging.info(f"Total runs: {total_runs}")
        logging.info(f"{'#'*80}\n")
        start_time = time.time()
        for idx, algorithm in enumerate(algorithms, 1):
            logging.info(f"\nProgress: [{idx}/{total_runs}]")
            result = self.evaluate_algorithm(algorithm, dataset_name, model_name)
            self.results.append(result)
        total_time = time.time() - start_time
        logging.info(f"\n{'#'*80}")
        logging.info(f"Benchmark test completed!")
        logging.info(f"Total runtime: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        logging.info(f"{'#'*80}\n")
    def save_results(self, dataset_name, model_name):
        filename = f"{dataset_name}_{model_name}.xlsx"
        output_path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(self.results)
        column_order = ['Algorithm', 'AUC', 'Time(s)']
        if 'Error' in df.columns:
            column_order.append('Error')
        df = df[[col for col in column_order if col in df.columns]]
        df.to_excel(output_path, index=False, engine='openpyxl')
        logging.info(f"\n✓ Results saved to: {output_path}")
        logging.info(f"  - Total algorithms: {len(df)}")
        logging.info(f"  - Successful runs: {df['AUC'].notna().sum()}")
        logging.info(f"  - Failed runs: {df['AUC'].isna().sum()}")
        return output_path
    def save_auprc_results(self, dataset_name, model_name):
        filename = f"auprc_{dataset_name}_{model_name}.xlsx"
        output_path = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(self.results)
        column_order = ['Algorithm', 'AUPRC', 'Time(s)']
        if 'Error' in df.columns:
            column_order.append('Error')
        df = df[[col for col in column_order if col in df.columns]]
        df.to_excel(output_path, index=False, engine='openpyxl')
        logging.info(f"\n✓ AUPRC results saved to: {output_path}")
        logging.info(f"  - Total algorithms: {len(df)}")
        logging.info(f"  - Successful runs: {df['AUPRC'].notna().sum()}")
        logging.info(f"  - Failed runs: {df['AUPRC'].isna().sum()}")
        return output_path
def main():
    parser = argparse.ArgumentParser(
        description='Unified benchmark program - 8 anomaly detection algorithms evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_benchmark.py --dataset bbc --model bert
  python unified_benchmark.py --dataset bbc --model openai_large --seed 42
  python unified_benchmark.py --dataset email_spam --model qwen --seed 123
        """
    )
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (bbc, covid_fake, email_spam, hate_speech, liar2, olid, smsspam)')
    parser.add_argument('--datasets', type=str, default="olid,covid_fake,hate_speech,liar2,email_spam,bbc,smsspam,movie_review,N24News,agnews",
                        help='Comma-separated dataset list; if provided, this parameter takes priority')
    parser.add_argument('--model', type=str, required=True,
                        help='Embedding model name (bert, llama, minilm, openai_ada, openai_large, openai_small, qwen, stella)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seed list, e.g. 41,42,43; if empty, use --seed')
    args = parser.parse_args()
    model_name = args.model
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [int(args.seed)]
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        if args.dataset is None:
            raise ValueError("Must provide --dataset or --datasets")
        datasets = [d.strip() for d in str(args.dataset).split(",") if d.strip()]
    multi_seed_mode = args.seeds is not None
    seeds_str = args.seeds.replace(" ", "") if args.seeds else ""
    for dataset_name in datasets:
        if multi_seed_mode:
            print("\n" + "="*80)
            print(f"Unified Benchmark Program")
            print(f"Dataset: {dataset_name}")
            print(f"Model: {model_name}")
            print(f"Random seeds: {seeds_str}")
            print(f"Algorithms: All 8 algorithms (LOF, DeepSVDD, ECOD, IForest, SO-GAAL, AE, VAE, LUNAR)")
            print("="*80)
            auc_by_alg = {}
            auprc_by_alg = {}
            time_by_alg = {}
            for seed in seeds:
                set_random_seed(int(seed))
                benchmark = UnifiedBenchmark(
                    embeddings_dir='../../embeddings',
                    data_dir='../../data',
                    output_dir='./',
                    seed=int(seed)
                )
                benchmark.run_benchmark(
                    dataset_name=dataset_name,
                    model_name=model_name,
                    algorithms=None
                )
                for r in benchmark.results:
                    alg = r.get('Algorithm')
                    auc = r.get('AUC')
                    auprc = r.get('AUPRC')
                    tsec = r.get('Time(s)')
                    if alg is None:
                        continue
                    auc_by_alg.setdefault(alg, [])
                    auprc_by_alg.setdefault(alg, [])
                    time_by_alg.setdefault(alg, [])
                    if auc is not None:
                        auc_by_alg[alg].append(float(auc))
                    if auprc is not None:
                        auprc_by_alg[alg].append(float(auprc))
                    if tsec is not None:
                        time_by_alg[alg].append(float(tsec))
            rows = []
            rows_auprc = []
            for alg in list(UnifiedBenchmark().algorithms.keys()):
                aucs = auc_by_alg.get(alg, [])
                auprcs = auprc_by_alg.get(alg, [])
                times = time_by_alg.get(alg, [])
                if len(aucs) == 0:
                    auc_mean = None
                    auc_std = None
                else:
                    auc_mean = float(statistics.mean(aucs))
                    auc_std = float(statistics.stdev(aucs)) if len(aucs) > 1 else 0.0
                if len(auprcs) == 0:
                    auprc_mean = None
                    auprc_std = None
                else:
                    auprc_mean = float(statistics.mean(auprcs))
                    auprc_std = float(statistics.stdev(auprcs)) if len(auprcs) > 1 else 0.0
                if len(times) == 0:
                    time_mean = None
                    time_std = None
                else:
                    time_mean = float(statistics.mean(times))
                    time_std = float(statistics.stdev(times)) if len(times) > 1 else 0.0
                rows.append(
                    {
                        'Algorithm': alg,
                        'AUC_mean': None if auc_mean is None else round(auc_mean, 4),
                        'AUC_std': None if auc_std is None else round(float(auc_std), 4),
                        'Time(s)_mean': None if time_mean is None else round(time_mean, 2),
                        'Time(s)_std': None if time_std is None else round(float(time_std), 2),
                        'n_success': int(len(aucs)),
                        'n_total': int(len(seeds)),
                        'seeds': ",".join(str(s) for s in seeds),
                    }
                )
                rows_auprc.append(
                    {
                        'Algorithm': alg,
                        'AUPRC_mean': None if auprc_mean is None else round(auprc_mean, 4),
                        'AUPRC_std': None if auprc_std is None else round(float(auprc_std), 4),
                        'Time(s)_mean': None if time_mean is None else round(time_mean, 2),
                        'Time(s)_std': None if time_std is None else round(float(time_std), 2),
                        'n_success': int(len(auprcs)),
                        'n_total': int(len(seeds)),
                        'seeds': ",".join(str(s) for s in seeds),
                    }
                )
            df = pd.DataFrame(rows)
            output_file = _save_results_df(output_dir='./', filename_no_ext=f"{dataset_name}_{model_name}_{seeds_str}", df=df)
            df_auprc = pd.DataFrame(rows_auprc)
            output_file_auprc = _save_results_df(output_dir='./', filename_no_ext=f"auprc_{dataset_name}_{model_name}_{seeds_str}", df=df_auprc)
            print("\n" + "="*80)
            print(f"Benchmark test completed!")
            print(f"AUROC results saved to: {output_file}")
            print(f"AUPRC results saved to: {output_file_auprc}")
            print("="*80)
        else:
            seed = seeds[0]
            set_random_seed(seed)
            print("\n" + "="*80)
            print(f"Unified Benchmark Program")
            print(f"Dataset: {dataset_name}")
            print(f"Model: {model_name}")
            print(f"Random seed: {seed}")
            print(f"Algorithms: All 8 algorithms (LOF, DeepSVDD, ECOD, IForest, SO-GAAL, AE, VAE, LUNAR)")
            print("="*80)
            benchmark = UnifiedBenchmark(
                embeddings_dir='../../embeddings',
                data_dir='../../data',
                output_dir='./',
                seed=int(seed)
            )
            benchmark.run_benchmark(
                dataset_name=dataset_name,
                model_name=model_name,
                algorithms=None
            )
            output_file = benchmark.save_results(dataset_name, model_name)
            output_file_auprc = benchmark.save_auprc_results(dataset_name, model_name)
            print("\n" + "="*80)
            print(f"Benchmark test completed!")
            print(f"AUROC results saved to: {output_file}")
            print(f"AUPRC results saved to: {output_file_auprc}")
            print("="*80)
if __name__ == '__main__':
    main()

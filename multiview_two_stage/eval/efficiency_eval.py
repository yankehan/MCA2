
import argparse
import gc
import os
import random
import sys
import time
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✓ 工作目录已切换到: {os.getcwd()}")
from multiview_two_stage.model import MultiViewContrastiveModel
from multiview_two_stage.dataset import load_all_views_full
from multiview_two_stage.trainer import SimpleMultiViewTrainer
try:
    from multiview_two_stage.eval.dataset_configs import DATASET_CONFIGS, print_dataset_config
    USE_DATASET_CONFIGS = True
except ImportError:
    USE_DATASET_CONFIGS = False
    DATASET_CONFIGS = {}
    print("⚠️  未找到dataset_configs.py，将使用默认配置")
DEFAULT_DATASETS = ','.join(list(DATASET_CONFIGS.keys())) if USE_DATASET_CONFIGS and DATASET_CONFIGS else 'bbc'
DATA_NAME = 'bbc'
VIEW_COMBINATIONS = {
    "O-ada+small+large": ['openai_ada', 'openai_small', 'openai_large'],
    "multi-view": ['openai_large', 'bert', 'qwen', 'llama'],
}
TRAIN_CONFIG = {
    'lambda_recon': 1.0,
    'lambda_contrastive': 1.0,
    'temperature': 0.5,
    'batch_size': None,
}
BASE_TRAIN_CONFIG = TRAIN_CONFIG.copy()
MODEL_CONFIG = {
    'latent_dim': 128,
    'hidden_dims': [512, 256],
    'activation': 'relu',
    'batchnorm': True,
}
FIRST_STAGE_EPOCHS = 50
SECOND_STAGE_EPOCHS = 50
LEARNING_RATE = 0.002
GATE_LEARNING_RATE = 0.001
PRINT_EVERY = 1
BASE_FIRST_STAGE_EPOCHS = FIRST_STAGE_EPOCHS
BASE_SECOND_STAGE_EPOCHS = SECOND_STAGE_EPOCHS
BASE_LEARNING_RATE = LEARNING_RATE
BASE_GATE_LEARNING_RATE = GATE_LEARNING_RATE
VIEW_GATE_HIDDEN_DIMS = None
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
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"随机数种子已设置为: {seed}")
def _parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(x) for x in s.split(',') if x.strip()]
def _set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)
def _train_two_stage_efficiency_once(
    train_views_dict: Dict[str, torch.Tensor],
    test_views_dict: Dict[str, torch.Tensor],
    test_labels: torch.Tensor,
    view_dims: Dict[str, int],
    device: torch.device,
    seed: int,
    first_stage_epochs: int,
    second_stage_epochs: int,
    gate_lr: float,
) -> Tuple[float, float, float]:
    set_random_seed(seed)
    model = MultiViewContrastiveModel(
        view_dims=view_dims,
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        activation=MODEL_CONFIG['activation'],
        batchnorm=MODEL_CONFIG['batchnorm'],
        use_view_gate=True,
        view_gate_hidden_dims=VIEW_GATE_HIDDEN_DIMS,
    ).to(device)
    if hasattr(model, 'set_view_gate_mode'):
        model.set_view_gate_mode('uniform')
    if hasattr(model, 'view_gate') and model.view_gate is not None and hasattr(model.view_gate, 'reset_to_uniform'):
        model.view_gate.reset_to_uniform()
    if hasattr(model, 'view_gate') and model.view_gate is not None:
        _set_requires_grad(model.view_gate, False)
    stage1_params = [p for n, p in model.named_parameters() if not n.startswith('view_gate.')]
    optimizer_stage1 = torch.optim.Adam(stage1_params, lr=LEARNING_RATE)
    trainer_stage1 = SimpleMultiViewTrainer(
        model=model,
        optimizer=optimizer_stage1,
        device=device,
        config=TRAIN_CONFIG,
    )
    train_time = 0.0
    test_time = 0.0
    stage1_start = time.time()
    for epoch in range(int(first_stage_epochs)):
        t0 = time.time()
        _ = trainer_stage1.train_epoch(train_views_dict)
        train_time += time.time() - t0
        if (epoch + 1) % PRINT_EVERY == 0:
            elapsed = time.time() - stage1_start
            print(f"[Stage1] Epoch {epoch + 1}/{first_stage_epochs} | Time: {elapsed:.1f}s")
    if hasattr(model, 'set_view_gate_mode'):
        model.set_view_gate_mode('learned')
    for n, p in model.named_parameters():
        if n.startswith('view_gate.'):
            p.requires_grad = True
        else:
            p.requires_grad = False
    if hasattr(model, 'view_gate') and model.view_gate is not None and hasattr(model.view_gate, 'reset_to_uniform'):
        model.view_gate.reset_to_uniform()
    gate_params = [p for n, p in model.named_parameters() if n.startswith('view_gate.')]
    optimizer_stage2 = torch.optim.Adam(gate_params, lr=float(gate_lr))
    stage2_config = dict(TRAIN_CONFIG)
    stage2_config['train_mode'] = 'gate_only'
    trainer_stage2 = SimpleMultiViewTrainer(
        model=model,
        optimizer=optimizer_stage2,
        device=device,
        config=stage2_config,
    )
    stage2_start = time.time()
    for epoch in range(int(second_stage_epochs)):
        t0 = time.time()
        _ = trainer_stage2.train_epoch(train_views_dict)
        train_time += time.time() - t0
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        t0 = time.time()
        _auc, _ = trainer_stage2.evaluate(test_views_dict, test_labels)
        test_time += time.time() - t0
        elapsed = time.time() - stage2_start
        print(f"[Stage2] Epoch {epoch + 1}/{second_stage_epochs} | Time: {elapsed:.1f}s")
    del trainer_stage1
    del trainer_stage2
    del optimizer_stage1
    del optimizer_stage2
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    total_time = float(train_time + test_time)
    return float(train_time), float(test_time), float(total_time)
def _save_efficiency_xlsx(eval_dir: str, dataset_name: str, rows: List[Dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    df = df.rename(
        columns={
            'dataset': '数据集名称',
            'view_combination': '模型视图组合',
            'memory_usage(MB)': '内存使用量(MB)',
            'train_time(s)': 'train_time(s)',
            'test_time(s)': 'test_time(s)',
            'total_time(s)': 'total_time(s)',
        }
    )
    col_order = ['数据集名称', '模型视图组合', '内存使用量(MB)', 'train_time(s)', 'test_time(s)', 'total_time(s)']
    df = df[[c for c in col_order if c in df.columns]]
    os.makedirs(eval_dir, exist_ok=True)
    out_xlsx = os.path.join(eval_dir, f"{dataset_name}_efficiency.xlsx")
    df.to_excel(out_xlsx, index=False, engine='openpyxl')
    return out_xlsx
def main_efficiency(dataset_name: str, seeds: List[int]):
    global DATA_NAME
    DATA_NAME = dataset_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80)
    print("效率分析实验 (ourmethod_eval 对齐)")
    print("=" * 80)
    print(f"\n数据集: {DATA_NAME}")
    print(f"设备: {device}")
    print(f"种子列表: {seeds}")
    rows: List[Dict[str, Any]] = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print("\n" + "=" * 80)
        print(f"正在评估组合: {combo_name}")
        print(f"视图: {view_names}")
        print("=" * 80)
        mem_monitor = _MemoryMonitor(interval_s=0.05)
        mem_monitor.start()
        total_start = time.time()
        try:
            print("\n加载训练集数据（只包含正常数据）...")
            train_views_dict, _train_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='train',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            print("\n加载测试集数据（包含正常+异常数据）...")
            test_views_dict, test_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='test',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            view_dims = {name: int(emb.shape[1]) for name, emb in train_views_dict.items()}
            print(f"视图维度: {view_dims}")
            train_times: List[float] = []
            test_times: List[float] = []
            total_times: List[float] = []
            for seed in seeds:
                print("\n--- 单次运行 ---")
                print(f"种子: {seed}")
                train_t, test_t, total_t = _train_two_stage_efficiency_once(
                    train_views_dict=train_views_dict,
                    test_views_dict=test_views_dict,
                    test_labels=test_labels,
                    view_dims=view_dims,
                    device=device,
                    seed=int(seed),
                    first_stage_epochs=FIRST_STAGE_EPOCHS,
                    second_stage_epochs=SECOND_STAGE_EPOCHS,
                    gate_lr=GATE_LEARNING_RATE,
                )
                train_times.append(float(train_t))
                test_times.append(float(test_t))
                total_times.append(float(total_t))
            peak_bytes = mem_monitor.stop()
            mem_mb = None if peak_bytes is None else peak_bytes / (1024 * 1024)
            row = {
                'dataset': DATA_NAME,
                'view_combination': combo_name,
                'memory_usage(MB)': None if mem_mb is None else round(float(mem_mb), 2),
                'train_time(s)': round(float(np.mean(train_times)), 4) if len(train_times) > 0 else None,
                'test_time(s)': round(float(np.mean(test_times)), 4) if len(test_times) > 0 else None,
                'total_time(s)': round(float(np.mean(total_times)), 4) if len(total_times) > 0 else None,
            }
            rows.append(row)
            print("\n效率结果:")
            print(f"  memory_usage(MB): {row['memory_usage(MB)']}")
            print(f"  train_time(s): {row['train_time(s)']}")
            print(f"  test_time(s): {row['test_time(s)']}")
            print(f"  total_time(s): {row['total_time(s)']}")
        except Exception as e:
            print(f"\n处理 {combo_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            mem_monitor.stop()
        finally:
            try:
                del train_views_dict
            except Exception:
                pass
            try:
                del test_views_dict
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _ = time.time() - total_start
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = _save_efficiency_xlsx(eval_dir=eval_dir, dataset_name=DATA_NAME, rows=rows)
    print(f"\n✓ 效率结果已保存至 {output_file}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='效率分析实验 (multiview_two_stage)')
    parser.add_argument('--dataset', type=str, default="olid,hate_speech,movie_review",
                        help='数据集名称，支持逗号分隔多个 (例如: bbc,olid)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子 (默认: 42)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='逗号分隔随机数种子列表，例如 41,42,43,44,45；若设置则忽略 --seed')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='重构损失权重 (默认: 1.0)')
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='对比学习损失权重 (默认: 1.0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数 (兼容旧参数；等价于 --first_stage_epochs)')
    parser.add_argument('--first_stage_epochs', type=int, default=None,
                        help='第一阶段训练轮数 (默认: 自动根据数据集选择；优先级高于 --epochs)')
    parser.add_argument('--second_stage_epochs', type=int, default=BASE_SECOND_STAGE_EPOCHS,
                        help='第二阶段训练轮数 (默认: 5)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 自动根据数据集选择)')
    parser.add_argument('--gate_lr', type=float, default=BASE_GATE_LEARNING_RATE,
                        help='第二阶段 gate 学习率 (默认: 0.001)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch大小 (默认: None表示使用全部数据)')
    parser.add_argument('--view_gate_hidden_dims', type=str, default="256,128",
                        help='gate 的隐藏层维度，逗号分隔，例如 256 或 256,128；为空表示线性 gate')
    parser.set_defaults(use_auto_config=True)
    parser.add_argument('--use_auto_config', dest='use_auto_config', action='store_true',
                        help='使用数据集自适应配置（默认: 开启）')
    parser.add_argument('--no_auto_config', dest='use_auto_config', action='store_false',
                        help='禁用数据集自适应配置')
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.dataset.split(',') if d.strip()]
    seeds_list = _parse_int_list(args.seeds) if args.seeds is not None else None
    if seeds_list is None or len(seeds_list) == 0:
        seeds_list = [int(args.seed)]
    for dataset_name in dataset_list:
        DATA_NAME = dataset_name
        VIEW_GATE_HIDDEN_DIMS = _parse_int_list(args.view_gate_hidden_dims)
        TRAIN_CONFIG.clear()
        TRAIN_CONFIG.update(BASE_TRAIN_CONFIG)
        FIRST_STAGE_EPOCHS = BASE_FIRST_STAGE_EPOCHS
        SECOND_STAGE_EPOCHS = BASE_SECOND_STAGE_EPOCHS
        LEARNING_RATE = BASE_LEARNING_RATE
        GATE_LEARNING_RATE = BASE_GATE_LEARNING_RATE
        if args.use_auto_config and USE_DATASET_CONFIGS and dataset_name in DATASET_CONFIGS:
            print("\n" + "🔧 使用数据集自适应配置".center(80, "="))
            dataset_config = print_dataset_config(dataset_name)
            first_stage_epochs_arg = args.first_stage_epochs if args.first_stage_epochs is not None else args.epochs
            FIRST_STAGE_EPOCHS = first_stage_epochs_arg if first_stage_epochs_arg is not None else dataset_config['num_epochs']
            LEARNING_RATE = args.lr if args.lr is not None else dataset_config['learning_rate']
            SECOND_STAGE_EPOCHS = int(args.second_stage_epochs)
            GATE_LEARNING_RATE = float(args.gate_lr)
            TRAIN_CONFIG['lambda_recon'] = args.lambda_recon if args.lambda_recon != 1.0 else dataset_config['lambda_recon']
            TRAIN_CONFIG['lambda_contrastive'] = args.lambda_contrastive if args.lambda_contrastive != 1.0 else dataset_config['lambda_contrastive']
            TRAIN_CONFIG['batch_size'] = args.batch_size if args.batch_size is not None else dataset_config.get('batch_size', None)
        else:
            first_stage_epochs_arg = args.first_stage_epochs if args.first_stage_epochs is not None else args.epochs
            if first_stage_epochs_arg is not None:
                FIRST_STAGE_EPOCHS = int(first_stage_epochs_arg)
            LEARNING_RATE = args.lr if args.lr is not None else LEARNING_RATE
            SECOND_STAGE_EPOCHS = int(args.second_stage_epochs)
            GATE_LEARNING_RATE = float(args.gate_lr)
            TRAIN_CONFIG['lambda_recon'] = args.lambda_recon
            TRAIN_CONFIG['lambda_contrastive'] = args.lambda_contrastive
            TRAIN_CONFIG['batch_size'] = args.batch_size
        main_efficiency(dataset_name=dataset_name, seeds=seeds_list)

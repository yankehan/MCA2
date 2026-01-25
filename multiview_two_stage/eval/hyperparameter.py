
import argparse
import numpy as np
import pandas as pd
import torch
import random
import torch.backends.cudnn as cudnn
import os
import sys
import gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✓ 工作目录已切换到: {os.getcwd()}")
from multiview_two_stage.eval import ourmethod_eval as base_eval
try:
    from multiview_two_stage.eval.dataset_configs import DATASET_CONFIGS, get_dataset_config
    USE_DATASET_CONFIGS = True
except ImportError:
    USE_DATASET_CONFIGS = False
    DATASET_CONFIGS = {}
    get_dataset_config = None
    print("⚠️  未找到dataset_configs.py，将使用默认配置")
DEFAULT_DATASETS = ','.join(list(DATASET_CONFIGS.keys())) if USE_DATASET_CONFIGS and DATASET_CONFIGS else 'bbc'
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"随机数种子已设置为: {seed}")
def _parse_int_list(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [int(x) for x in s.split(',') if x.strip()]
def _apply_dataset_auto_config(dataset_name, args):
    base_eval.TRAIN_CONFIG.clear()
    base_eval.TRAIN_CONFIG.update(base_eval.BASE_TRAIN_CONFIG)
    base_eval.FIRST_STAGE_EPOCHS = base_eval.BASE_FIRST_STAGE_EPOCHS
    base_eval.SECOND_STAGE_EPOCHS = base_eval.BASE_SECOND_STAGE_EPOCHS
    base_eval.LEARNING_RATE = base_eval.BASE_LEARNING_RATE
    base_eval.GATE_LEARNING_RATE = base_eval.BASE_GATE_LEARNING_RATE
    if args.first_stage_epochs is not None:
        base_eval.FIRST_STAGE_EPOCHS = int(args.first_stage_epochs)
    if args.second_stage_epochs is not None:
        base_eval.SECOND_STAGE_EPOCHS = int(args.second_stage_epochs)
    if args.lr is not None:
        base_eval.LEARNING_RATE = float(args.lr)
    if args.gate_lr is not None:
        base_eval.GATE_LEARNING_RATE = float(args.gate_lr)
    if args.use_auto_config and USE_DATASET_CONFIGS and dataset_name in DATASET_CONFIGS:
        dataset_config = get_dataset_config(dataset_name)
        if args.first_stage_epochs is None:
            base_eval.FIRST_STAGE_EPOCHS = int(dataset_config['num_epochs'])
        if args.lr is None:
            base_eval.LEARNING_RATE = float(dataset_config['learning_rate'])
        if args.lambda_recon != 1.0:
            base_eval.TRAIN_CONFIG['lambda_recon'] = float(args.lambda_recon)
        else:
            base_eval.TRAIN_CONFIG['lambda_recon'] = float(dataset_config.get('lambda_recon', args.lambda_recon))
        if args.lambda_contrastive != 1.0:
            base_eval.TRAIN_CONFIG['lambda_contrastive'] = float(args.lambda_contrastive)
        else:
            base_eval.TRAIN_CONFIG['lambda_contrastive'] = float(
                dataset_config.get('lambda_contrastive', args.lambda_contrastive)
            )
        base_eval.TRAIN_CONFIG['batch_size'] = args.batch_size if args.batch_size is not None else dataset_config.get(
            'batch_size', None
        )
        base_eval.TRAIN_CONFIG['score_weight_recon'] = (
            args.score_weight_recon
            if args.score_weight_recon is not None
            else dataset_config.get('score_weight_recon', 0.3)
        )
        base_eval.TRAIN_CONFIG['score_weight_consistency'] = (
            args.score_weight_consistency
            if args.score_weight_consistency is not None
            else dataset_config.get('score_weight_consistency', 0.4)
        )
    else:
        base_eval.TRAIN_CONFIG['lambda_recon'] = float(args.lambda_recon)
        base_eval.TRAIN_CONFIG['lambda_contrastive'] = float(args.lambda_contrastive)
        base_eval.TRAIN_CONFIG['batch_size'] = args.batch_size
        base_eval.TRAIN_CONFIG['score_weight_recon'] = args.score_weight_recon if args.score_weight_recon is not None else 0.3
        base_eval.TRAIN_CONFIG['score_weight_consistency'] = (
            args.score_weight_consistency if args.score_weight_consistency is not None else 0.4
        )
def _combo_to_filename_tag(combo_name: str) -> str:
    if combo_name is None:
        return 'combo'
    name = str(combo_name).strip()
    lower = name.lower()
    if 'multi-view' in lower or 'multiview' in lower:
        return 'multiview'
    if lower.startswith('o-') or 'openai' in lower:
        return 'openai'
    cleaned = []
    for ch in lower:
        if ch.isalnum():
            cleaned.append(ch)
        elif ch in ('-', '_', '+'):
            cleaned.append('_')
    tag = ''.join(cleaned).strip('_')
    return tag or 'combo'
def _run_one_combo_grid(
    dataset_name,
    combo_name,
    view_names,
    device,
    seed,
    score_weight_recon_values,
    score_weight_consistency_values,
):
    print("\n" + "=" * 80)
    print(f"数据集: {dataset_name} | 组合: {combo_name}")
    print(f"视图: {view_names}")
    print("=" * 80)
    print("\n加载训练集数据（只包含正常数据）...")
    train_views_dict, _, _ = base_eval.load_all_views_full(
        data_name=dataset_name,
        view_names=view_names,
        split='train',
        embeddings_dir='embeddings',
        normalize=True,
        device=device,
    )
    print("\n加载测试集数据（包含正常+异常数据）...")
    test_views_dict, test_labels, _ = base_eval.load_all_views_full(
        data_name=dataset_name,
        view_names=view_names,
        split='test',
        embeddings_dir='embeddings',
        normalize=True,
        device=device,
    )
    view_dims = {name: emb.shape[1] for name, emb in train_views_dict.items()}
    print(f"视图维度: {view_dims}")
    auc_rows = []
    for lr in score_weight_recon_values:
        row_aucs = []
        for lc in score_weight_consistency_values:
            base_eval.TRAIN_CONFIG['score_weight_recon'] = float(lr)
            base_eval.TRAIN_CONFIG['score_weight_consistency'] = float(lc)
            print("\n" + "-" * 80)
            print(
                f"开始训练/评估: {dataset_name} | {combo_name} | "
                f"score_weight_recon={base_eval.TRAIN_CONFIG['score_weight_recon']} | "
                f"score_weight_consistency={base_eval.TRAIN_CONFIG['score_weight_consistency']}"
            )
            print("-" * 80)
            auc_on, _, _ = base_eval._train_two_stage_and_eval_once(
                train_views_dict=train_views_dict,
                test_views_dict=test_views_dict,
                test_labels=test_labels,
                view_dims=view_dims,
                device=device,
                seed=seed,
                first_stage_epochs=base_eval.FIRST_STAGE_EPOCHS,
                second_stage_epochs=base_eval.SECOND_STAGE_EPOCHS,
                gate_lr=base_eval.GATE_LEARNING_RATE,
            )
            row_aucs.append(float(auc_on))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        auc_rows.append(row_aucs)
    del train_views_dict
    del test_views_dict
    del test_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return auc_rows
def _save_excel_per_combo(dataset_name, combo_name, score_weight_recon_values, score_weight_consistency_values, auc_rows):
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    tag = _combo_to_filename_tag(combo_name)
    output_file = os.path.join(eval_dir, f"{dataset_name}_{tag}.xlsx")
    columns = (
        ['Dataset', 'Combination', 'score_weight_recon']
        + [f"score_weight_consistency={v:.1f}_AUC" for v in score_weight_consistency_values]
    )
    results_rows = []
    for i, lr in enumerate(score_weight_recon_values):
        row = [dataset_name, combo_name, f"score_weight_recon={int(lr)}"] + list(auc_rows[i])
        results_rows.append(row)
    df = pd.DataFrame(results_rows, columns=columns)
    df.to_excel(output_file, index=False)
    print(f"\n✓ 已保存: {output_file}")
def run_experiment(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    dataset_list = [d.strip() for d in args.dataset.split(',') if d.strip()]
    score_weight_recon_values = list(range(1, 11))
    score_weight_consistency_values = [float(x) for x in np.round(np.arange(0.1, 1.0 + 1e-9, 0.1), 1).tolist()]
    for dataset_name in dataset_list:
        base_eval.DATA_NAME = dataset_name
        base_eval.VIEW_GATE_HIDDEN_DIMS = _parse_int_list(args.view_gate_hidden_dims)
        _apply_dataset_auto_config(dataset_name, args)
        target_view_set = {'openai_ada', 'openai_small', 'openai_large'}
        selected = [
            (combo_name, view_names)
            for combo_name, view_names in base_eval.VIEW_COMBINATIONS.items()
            if set(view_names) == target_view_set
        ]
        if len(selected) > 1:
            selected = [selected[0]]
        if not selected:
            selected = [("O-ada+small+large", ['openai_ada', 'openai_small', 'openai_large'])]
        for combo_name, view_names in selected:
            auc_rows = _run_one_combo_grid(
                dataset_name=dataset_name,
                combo_name=combo_name,
                view_names=view_names,
                device=device,
                seed=int(args.seed),
                score_weight_recon_values=score_weight_recon_values,
                score_weight_consistency_values=score_weight_consistency_values,
            )
            _save_excel_per_combo(
                dataset_name=dataset_name,
                combo_name=combo_name,
                score_weight_recon_values=score_weight_recon_values,
                score_weight_consistency_values=score_weight_consistency_values,
                auc_rows=auc_rows,
            )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='超参实验 (multiview_two_stage)')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASETS,
                        help='数据集名称，支持逗号分隔多个 (例如: bbc,olid)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子 (默认: 42)')
    parser.add_argument('--experiment', type=str, default='all', choices=['grid', 'all', 'recon', 'contrastive'],
                        help='运行超参实验（grid: 10x10；为兼容保留 recon/contrastive，但都会按 grid 执行）')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='重构损失权重 (默认: 1.0)')
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='对比学习损失权重 (默认: 1.0)')
    parser.add_argument('--first_stage_epochs', type=int, default=None,
                        help='第一阶段训练轮数 (默认: 自动根据数据集选择)')
    parser.add_argument('--second_stage_epochs', type=int, default=base_eval.BASE_SECOND_STAGE_EPOCHS,
                        help='第二阶段训练轮数 (默认: 使用 ourmethod_eval.py 的默认值)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 自动根据数据集选择)')
    parser.add_argument('--gate_lr', type=float, default=base_eval.BASE_GATE_LEARNING_RATE,
                        help='第二阶段 gate 学习率 (默认: 0.001)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch大小 (默认: None表示使用全部数据)')
    parser.add_argument('--score_weight_recon', type=float, default=None,
                        help='异常分数中重构误差的权重 (默认: 自动根据数据集选择)')
    parser.add_argument('--score_weight_consistency', type=float, default=None,
                        help='异常分数中一致性分数的权重 (默认: 自动根据数据集选择)')
    parser.set_defaults(use_auto_config=True)
    parser.add_argument('--use_auto_config', dest='use_auto_config', action='store_true',
                        help='使用数据集自适应配置（默认: 开启）')
    parser.add_argument('--no_auto_config', dest='use_auto_config', action='store_false',
                        help='禁用数据集自适应配置')
    parser.add_argument('--view_gate_hidden_dims', type=str, default="256,128",
                        help='gate 的隐藏层维度，逗号分隔，例如 256 或 256,128；为空表示线性 gate')
    args = parser.parse_args()
    set_random_seed(int(args.seed))
    if args.experiment in ('grid', 'all', 'recon', 'contrastive'):
        run_experiment(args)

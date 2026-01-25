
import argparse
import gc
import os
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✓ 工作目录已切换到: {os.getcwd()}")
from multiview_two_stage.dataset import load_all_views_full
from multiview_two_stage.model import MultiViewContrastiveModel
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
ROBUST_STUDY_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
TRAIN_CONFIG = {
    'lambda_recon': 1.0,
    'lambda_contrastive': 1.0,
    'temperature': 0.5,
    'batch_size': None,
}
BASE_TRAIN_CONFIG = TRAIN_CONFIG.copy()
MODEL_CONFIG = {
    'latent_dim': 256,
    'hidden_dims': [1024, 512],
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
def _mean_var(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float('nan'), float('nan')
    return float(values.mean()), float(values.var())
def _format_mean_var(mean, var, mean_fmt, var_fmt=None):
    if var_fmt is None:
        var_fmt = mean_fmt
    if np.isnan(mean) or np.isnan(var):
        return 'nan ± nan'
    return f"{mean:{mean_fmt}} ± {var:{var_fmt}}"
def _set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = bool(requires_grad)
def _build_robust_splits(
    train_views_dict,
    test_views_dict,
    test_labels,
    anomaly_train_ratio: float,
    seed: int,
):
    if test_labels is None:
        raise ValueError("test_labels 不能为空")
    device = test_labels.device
    labels_cpu = test_labels.detach().cpu()
    test_norm_indices_in_test_split = torch.where(labels_cpu == 0)[0]
    anom_indices_in_test_split = torch.where(labels_cpu == 1)[0]
    anom_pool_views = {v: test_views_dict[v][anom_indices_in_test_split] for v in test_views_dict}
    anom_pool_labels = test_labels[anom_indices_in_test_split]
    total_anomalies = len(anom_indices_in_test_split)
    n_anom_for_train = int(anomaly_train_ratio * total_anomalies)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_anomalies)
    train_anom_perm_indices = perm[:n_anom_for_train]
    rest_anom_perm_indices = perm[n_anom_for_train:]
    n_anom_for_test = int(0.8 * len(rest_anom_perm_indices))
    test_anom_perm_indices = rng.choice(rest_anom_perm_indices, n_anom_for_test, replace=False)
    train_anom_indices_in_pool = torch.from_numpy(train_anom_perm_indices).long().to(device)
    test_anom_indices_in_pool = torch.from_numpy(test_anom_perm_indices).long().to(device)
    new_train_views = {}
    for v in train_views_dict:
        if n_anom_for_train > 0:
            anom_to_add = anom_pool_views[v][train_anom_indices_in_pool]
            new_train_views[v] = torch.cat([train_views_dict[v], anom_to_add], dim=0)
        else:
            new_train_views[v] = train_views_dict[v]
    test_norm_views = {v: test_views_dict[v][test_norm_indices_in_test_split] for v in test_views_dict}
    test_norm_labels = test_labels[test_norm_indices_in_test_split]
    new_test_views = {}
    if n_anom_for_test > 0:
        test_anom_views = {v: anom_pool_views[v][test_anom_indices_in_pool] for v in anom_pool_views}
        test_anom_labels = anom_pool_labels[test_anom_indices_in_pool]
        for v in test_views_dict:
            new_test_views[v] = torch.cat([test_norm_views[v], test_anom_views[v]], dim=0)
        new_test_labels = torch.cat([test_norm_labels, test_anom_labels], dim=0)
    else:
        new_test_views = test_norm_views
        new_test_labels = test_norm_labels
    stats = {
        'train_normal': list(train_views_dict.values())[0].shape[0],
        'train_anomaly': len(train_anom_indices_in_pool),
        'test_normal': len(test_norm_indices_in_test_split),
        'test_anomaly': len(test_anom_indices_in_pool),
        'total_anomaly': total_anomalies,
    }
    return new_train_views, new_test_views, new_test_labels, stats
def _train_two_stage_and_eval_once(
    train_views_dict,
    test_views_dict,
    test_labels,
    view_dims,
    device,
    seed,
    first_stage_epochs,
    second_stage_epochs,
    gate_lr,
):
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
    print("\n=== 模型参数统计 ===")
    total_params = sum(p.numel() for p in model.parameters())
    if hasattr(model, 'view_gate') and model.view_gate is not None:
        gate_params_num = sum(p.numel() for p in model.view_gate.parameters())
        print(f"总参数数: {total_params}, 门控参数数: {gate_params_num}, 主要参数数: {total_params - gate_params_num}")
    else:
        print(f"总参数数: {total_params}, 门控参数数: 0")
    print("=== 参数统计结束 ===\n")
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
    print("\n" + "Stage 1: 训练 AE + 对比学习（Gate 固定为均匀，不更新）".center(80, "-"))
    print(f"first_stage_epochs={first_stage_epochs}")
    stage1_start = time.time()
    for epoch in range(int(first_stage_epochs)):
        loss_dict = trainer_stage1.train_epoch(train_views_dict)
        if (epoch + 1) % PRINT_EVERY == 0:
            elapsed = time.time() - stage1_start
            print(
                f"[Stage1] Epoch {epoch + 1}/{first_stage_epochs} | "
                f"Loss: {loss_dict['total']:.4f} | Recon: {loss_dict['recon']:.4f} | Contr: {loss_dict['contrastive']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )
    stage1_time = time.time() - stage1_start
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
    print("\n" + "Stage 2: 冻结 AE+CL，仅训练 Gate（每 epoch 评估 AUC）".center(80, "-"))
    print(f"second_stage_epochs={second_stage_epochs}, gate_lr={float(gate_lr)}")
    best_auc = -1.0
    best_state = None
    stage2_start = time.time()
    for epoch in range(int(second_stage_epochs)):
        loss_dict = trainer_stage2.train_epoch(train_views_dict)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        auc, _ = trainer_stage2.evaluate(test_views_dict, test_labels)
        elapsed = time.time() - stage2_start
        print(
            f"[Stage2] Epoch {epoch + 1}/{second_stage_epochs} | "
            f"Loss: {loss_dict['total']:.4f} | Recon: {loss_dict['recon']:.4f} | Contr: {loss_dict['contrastive']:.4f} | "
            f"Test AUC: {auc:.4f} | Time: {elapsed:.1f}s"
        )
        if auc > best_auc:
            best_auc = float(auc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stage2_time = time.time() - stage2_start
    if best_state is not None:
        model.load_state_dict(best_state)
    auc_on = float(best_auc)
    print(f"\n[Stage2] gate_on best AUC: {auc_on:.4f}")
    del trainer_stage1
    del trainer_stage2
    del optimizer_stage1
    del optimizer_stage2
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return auc_on, stage1_time, stage2_time
def _run_robust_study_for_dataset(dataset_name: str, seeds):
    global DATA_NAME
    global FIRST_STAGE_EPOCHS
    global SECOND_STAGE_EPOCHS
    global LEARNING_RATE
    global GATE_LEARNING_RATE
    DATA_NAME = dataset_name
    TRAIN_CONFIG.clear()
    TRAIN_CONFIG.update(BASE_TRAIN_CONFIG)
    FIRST_STAGE_EPOCHS = BASE_FIRST_STAGE_EPOCHS
    SECOND_STAGE_EPOCHS = BASE_SECOND_STAGE_EPOCHS
    LEARNING_RATE = BASE_LEARNING_RATE
    GATE_LEARNING_RATE = BASE_GATE_LEARNING_RATE
    if USE_DATASET_CONFIGS and dataset_name in DATASET_CONFIGS:
        print("\n" + "🔧 使用数据集自适应配置".center(80, "="))
        dataset_config = print_dataset_config(dataset_name)
        FIRST_STAGE_EPOCHS = dataset_config['num_epochs']
        LEARNING_RATE = dataset_config['learning_rate']
        TRAIN_CONFIG['lambda_recon'] = dataset_config['lambda_recon']
        TRAIN_CONFIG['lambda_contrastive'] = dataset_config['lambda_contrastive']
        if TRAIN_CONFIG.get('batch_size', None) is None:
            TRAIN_CONFIG['batch_size'] = dataset_config.get('batch_size', None)
        TRAIN_CONFIG['score_weight_recon'] = dataset_config.get('score_weight_recon', 0.3)
        TRAIN_CONFIG['score_weight_consistency'] = dataset_config.get('score_weight_consistency', 0.4)
    print("=" * 80)
    print("Robust Study (two-stage gate training)")
    print("=" * 80)
    print(f"\n数据集: {DATA_NAME}")
    print(f"种子列表: {seeds}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    results_list = []
    single_seed = len(seeds) == 1
    for contam_ratio in ROBUST_STUDY_LEVELS:
        print("\n" + "=" * 80)
        print(f"污染设置: train 异常={int(contam_ratio * 100)}% (占总异常) | test 异常=80% (占剩余异常)")
        print("=" * 80)
        for combo_name, view_names in VIEW_COMBINATIONS.items():
            print("\n" + "-" * 80)
            print(f"组合: {combo_name}")
            print(f"视图: {view_names}")
            print("-" * 80)
            try:
                print("\n加载基础训练集（只包含正常数据）...")
                base_train_views, base_train_labels, _ = load_all_views_full(
                    data_name=DATA_NAME,
                    view_names=view_names,
                    split='train',
                    embeddings_dir='embeddings',
                    normalize=True,
                    device=device,
                )
                print("\n加载基础测试集（包含正常+异常数据）...")
                base_test_views, base_test_labels, _ = load_all_views_full(
                    data_name=DATA_NAME,
                    view_names=view_names,
                    split='test',
                    embeddings_dir='embeddings',
                    normalize=True,
                    device=device,
                )
                view_dims = {name: emb.shape[1] for name, emb in base_train_views.items()}
                print(f"视图维度: {view_dims}")
                auc_list = []
                stage1_time_list = []
                stage2_time_list = []
                stats_ref = None
                for seed in seeds:
                    print("\n--- 单次运行 ---")
                    print(f"种子: {seed}")
                    train_views, test_views, test_labels, stats = _build_robust_splits(
                        train_views_dict=base_train_views,
                        test_views_dict=base_test_views,
                        test_labels=base_test_labels,
                        anomaly_train_ratio=float(contam_ratio),
                        seed=int(seed),
                    )
                    stats_ref = stats
                    print("\n--- 数据划分统计 ---")
                    print(
                        f"train: normal={stats['train_normal']} + anomaly={stats['train_anomaly']} | "
                        f"test: normal={stats['test_normal']} + anomaly={stats['test_anomaly']} | "
                        f"total_anomaly={stats['total_anomaly']}"
                    )
                    auc_on, stage1_time, stage2_time = _train_two_stage_and_eval_once(
                        train_views_dict=train_views,
                        test_views_dict=test_views,
                        test_labels=test_labels,
                        view_dims=view_dims,
                        device=device,
                        seed=int(seed),
                        first_stage_epochs=FIRST_STAGE_EPOCHS,
                        second_stage_epochs=SECOND_STAGE_EPOCHS,
                        gate_lr=GATE_LEARNING_RATE,
                    )
                    auc_list.append(auc_on)
                    stage1_time_list.append(stage1_time)
                    stage2_time_list.append(stage2_time)
                total_time_list = [t1 + t2 for t1, t2 in zip(stage1_time_list, stage2_time_list)]
                if single_seed:
                    gate_on_auc_value = float(round(float(auc_list[0]), 4))
                    stage1_time_value = float(round(float(stage1_time_list[0]), 1))
                    stage2_time_value = float(round(float(stage2_time_list[0]), 1))
                    total_time_value = float(round(float(total_time_list[0]), 1))
                else:
                    auc_mean, auc_var = _mean_var(auc_list)
                    stage1_mean, stage1_var = _mean_var(stage1_time_list)
                    stage2_mean, stage2_var = _mean_var(stage2_time_list)
                    total_mean, total_var = _mean_var(total_time_list)
                    gate_on_auc_value = _format_mean_var(auc_mean, auc_var, '.4f', '.4f')
                    stage1_time_value = _format_mean_var(stage1_mean, stage1_var, '.1f', '.1f')
                    stage2_time_value = _format_mean_var(stage2_mean, stage2_var, '.1f', '.1f')
                    total_time_value = _format_mean_var(total_mean, total_var, '.1f', '.1f')
                row = {
                    'Dataset': DATA_NAME,
                    'Contam_Ratio': float(contam_ratio),
                    'Train_Anom_Pct': int(contam_ratio * 100),
                    'Contam_Tag': f"train_anom_{int(contam_ratio * 100)}%",
                    'Combination': combo_name,
                    'Num_Views': len(view_names),
                    'Total_Dim': sum(view_dims.values()),
                    'Train_Normal': stats_ref['train_normal'] if stats_ref is not None else None,
                    'Train_Anomaly': stats_ref['train_anomaly'] if stats_ref is not None else None,
                    'Test_Normal': stats_ref['test_normal'] if stats_ref is not None else None,
                    'Test_Anomaly': stats_ref['test_anomaly'] if stats_ref is not None else None,
                    'Gate_ON_AUC': gate_on_auc_value,
                    'Stage1_Time': stage1_time_value,
                    'Stage2_Time': stage2_time_value,
                    'Total_Time': total_time_value,
                    'Seeds': ','.join(str(s) for s in seeds),
                }
                results_list.append(row)
            except Exception as e:
                print(f"\n处理 {combo_name} (contam={contam_ratio}) 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    results_df = pd.DataFrame(results_list)
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    if not results_df.empty:
        expected_levels = [int(x * 100) for x in ROBUST_STUDY_LEVELS]
        summary_df = results_df.pivot_table(
            index=['Dataset', 'Combination'],
            columns='Train_Anom_Pct',
            values='Gate_ON_AUC',
            aggfunc='first',
        )
        for lvl in expected_levels:
            if lvl not in summary_df.columns:
                summary_df[lvl] = np.nan
        summary_df = summary_df[expected_levels]
        col_rename = {lvl: f"{lvl}%异常AUC" for lvl in expected_levels}
        summary_df = summary_df.rename(columns=col_rename).reset_index()
        summary_df = summary_df.rename(columns={'Dataset': '数据集名称', 'Combination': '视图组合'})
        combo_order = list(VIEW_COMBINATIONS.keys())
        summary_df['视图组合'] = pd.Categorical(summary_df['视图组合'], categories=combo_order, ordered=True)
        summary_df = summary_df.sort_values(['数据集名称', '视图组合']).reset_index(drop=True)
        print("\n" + summary_df.to_string(index=False))
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        seeds_tag = '_'.join(str(s) for s in seeds)
        output_file = os.path.join(eval_dir, f"{DATA_NAME}_robust_study_seeds_{seeds_tag}.xlsx")
        summary_df.to_excel(output_file, index=False)
        print(f"\n✓ 结果已保存至 {output_file}")
    print("\n" + "=" * 80)
    print("数据集评估完成！")
    print("=" * 80)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Robust Study (multiview_two_stage)')
    parser.add_argument('--dataset', type=str, default="covid_fake,bbc,email_spam",
                        help='数据集名称，支持逗号分隔多个 (例如: bbc,olid)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机数种子 (默认: 42)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='逗号分隔随机数种子列表，例如 41,42,43；若设置则忽略 --seed')
    parser.add_argument('--second_stage_epochs', type=int, default=BASE_SECOND_STAGE_EPOCHS,
                        help='第二阶段训练轮数 (默认: 50)')
    parser.add_argument('--gate_lr', type=float, default=BASE_GATE_LEARNING_RATE,
                        help='第二阶段 gate 学习率 (默认: 0.001)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch大小 (默认: None表示使用全部数据；若 dataset_configs 有配置则优先使用其值)')
    parser.add_argument('--view_gate_hidden_dims', type=str, default="256,128",
                        help='gate 的隐藏层维度，逗号分隔，例如 256 或 256,128；为空表示线性 gate')
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.dataset.split(',') if d.strip()]
    seeds_list = _parse_int_list(args.seeds) if args.seeds is not None else None
    if seeds_list is None or len(seeds_list) == 0:
        seeds_list = [int(args.seed)]
    VIEW_GATE_HIDDEN_DIMS = _parse_int_list(args.view_gate_hidden_dims)
    BASE_SECOND_STAGE_EPOCHS = int(args.second_stage_epochs)
    BASE_GATE_LEARNING_RATE = float(args.gate_lr)
    if args.batch_size is not None:
        BASE_TRAIN_CONFIG['batch_size'] = int(args.batch_size)
    for dataset_name in dataset_list:
        _run_robust_study_for_dataset(dataset_name=dataset_name, seeds=seeds_list)

import argparse
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import random
import time
import os
import sys
import gc
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✓ 工作目录已切换到: {os.getcwd()}")
from multiview_rcpmod.model import MultiViewContrastiveModel
from multiview_rcpmod.dataset import load_all_views_full, get_view_dimensions
from multiview_rcpmod.trainer import SimpleMultiViewTrainer
try:
    from multiview_rcpmod.eval.dataset_configs import DATASET_CONFIGS, get_dataset_config, print_dataset_config
    USE_DATASET_CONFIGS = True
except ImportError:
    try:
        from multiview_ourmethod.eval.dataset_configs import DATASET_CONFIGS, get_dataset_config, print_dataset_config
        USE_DATASET_CONFIGS = True
    except ImportError:
        USE_DATASET_CONFIGS = False
        DATASET_CONFIGS = {}
DEFAULT_DATASETS = ','.join(list(DATASET_CONFIGS.keys())) if USE_DATASET_CONFIGS and DATASET_CONFIGS else 'bbc'
DATA_NAME = 'bbc'
VIEW_COMBINATIONS = {
    "O-ada+small+large": ['openai_ada', 'openai_small', 'openai_large'],
    "multi-view": ['openai_large', 'bert', 'qwen', 'llama'],
}
TRAIN_CONFIG = {
    'use_contrastive': True,
    'use_memory_bank': True,
    'use_knn_contrastive': True,
    'use_triplet': True,
    'use_uniform': True,
    'lambda_recon': 1.0,
    'lambda_contrastive': 1.0,
    'lambda_knn': 0.1,
    'lambda_triplet': 0.03,
    'lambda_uniform': 0.03,
    'k_neighbors': 6,
    'update_knn_every': 10,
    'temperature': 0.5,
    'memory_ratio': 0.05,
    'memory_size': 10,
    'start_uniform_epoch': 0,
    'score_weight_knn': 0.03,
    'batch_size': None,
}
BASE_TRAIN_CONFIG = TRAIN_CONFIG.copy()
MODEL_CONFIG = {
    'latent_dim': 128,
    'hidden_dims': [512, 256],
    'activation': 'relu',
    'batchnorm': True,
}
NUM_EPOCHS = 200
LEARNING_RATE = 0.002
PRINT_EVERY = 1
BASE_NUM_EPOCHS = NUM_EPOCHS
BASE_LEARNING_RATE = LEARNING_RATE
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
def _train_and_eval_once(
    train_views_dict,
    test_views_dict,
    test_labels,
    view_dims,
    device,
    seed,
):
    set_random_seed(seed)
    model = MultiViewContrastiveModel(
        view_dims=view_dims,
        latent_dim=MODEL_CONFIG['latent_dim'],
        hidden_dims=MODEL_CONFIG['hidden_dims'],
        activation=MODEL_CONFIG['activation'],
        batchnorm=MODEL_CONFIG['batchnorm'],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = SimpleMultiViewTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        config=TRAIN_CONFIG,
    )
    start_time = time.time()
    best_auc, best_auprc, _ = trainer.train(
        train_views_dict=train_views_dict,
        test_views_dict=test_views_dict,
        test_labels=test_labels,
        num_epochs=NUM_EPOCHS,
        print_every=PRINT_EVERY,
    )
    train_time = time.time() - start_time
    del trainer
    del optimizer
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(best_auc), float(best_auprc), float(train_time)
def main(seed=42):
    set_random_seed(seed)
    print("=" * 80)
    print("自定义多视图方法 - 仅重构 + 对比学习 (训练集/测试集分离)")
    print("=" * 80)
    print(f"\n数据集: {DATA_NAME}")
    print(f"配置: lambda_recon={TRAIN_CONFIG['lambda_recon']}, "
          f"lambda_contrastive={TRAIN_CONFIG['lambda_contrastive']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"种子: {seed}")
    results_list = []
    failed_combos = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print("\n" + "=" * 80)
        print(f"正在评估: {combo_name}")
        print(f"视图: {view_names}")
        print("=" * 80)
        try:
            print("\n加载训练集数据（只包含正常数据）...")
            train_views_dict, train_labels, train_norm_stats = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='train',
                embeddings_dir='embeddings',
                normalize=True,
                device=device
            )
            print("\n加载测试集数据（包含正常+异常数据）...")
            test_views_dict, test_labels, test_norm_stats = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='test',
                embeddings_dir='embeddings',
                normalize=True,
                device=device
            )
            print(f"\n训练集标签分布: {torch.bincount(train_labels)}")
            print(f"测试集标签分布: {torch.bincount(test_labels)}")
            print(f"测试集异常比例: {torch.sum(test_labels).item() / len(test_labels):.2%}")
            view_dims = {name: emb.shape[1] for name, emb in train_views_dict.items()}
            print(f"视图维度: {view_dims}")
            result_row = {
                'Combination': combo_name,
                'Views': '+'.join(view_names),
                'Num_Views': len(view_names),
                'Total_Dim': sum(view_dims.values()),
            }
            if len(view_names) > 1:
                print("\n--- 自定义方法训练（仅重构 + 对比学习）---")
                print(f"学习率: lr={LEARNING_RATE}")
                print(f"损失权重: λ_recon={TRAIN_CONFIG['lambda_recon']}, λ_contrastive={TRAIN_CONFIG['lambda_contrastive']}")
                print(f"分数权重: recon={TRAIN_CONFIG.get('score_weight_recon', 0.3)}, "
                      f"consistency={TRAIN_CONFIG.get('score_weight_consistency', 0.4)}")
                best_auc, best_auprc, train_time = _train_and_eval_once(
                    train_views_dict=train_views_dict,
                    test_views_dict=test_views_dict,
                    test_labels=test_labels,
                    view_dims=view_dims,
                    device=device,
                    seed=seed,
                )
                result_row['OurMethod_AUC'] = best_auc
                result_row['OurMethod_AUPRC'] = best_auprc
                result_row['Train_Time'] = train_time
                print(f"\n自定义方法 测试集AUC: {best_auc:.4f} AUPRC: {best_auprc:.4f}")
                print(f"训练时间: {train_time:.2f}秒")
            else:
                result_row['OurMethod_AUC'] = None
                result_row['OurMethod_AUPRC'] = None
                result_row['Train_Time'] = None
            results_list.append(result_row)
        except Exception as e:
            print(f"\n处理 {combo_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_combos.append((combo_name, str(e)))
            continue
    if failed_combos:
        print(f"\nWARNING: {len(failed_combos)}/{len(VIEW_COMBINATIONS)} combinations failed:")
        for name, err in failed_combos:
            print(f"  - {name}: {err}")
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    display_cols = ['Combination', 'Num_Views', 'Total_Dim', 'OurMethod_AUC', 'OurMethod_AUPRC', 'Train_Time']
    print("\n" + results_df[display_cols].to_string(index=False))
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    auroc_cols = ['Combination', 'Views', 'Num_Views', 'Total_Dim', 'OurMethod_AUC', 'Train_Time']
    auroc_df = results_df[auroc_cols].copy()
    output_file = os.path.join(eval_dir, f"{DATA_NAME}_ourmethod_results.xlsx")
    try:
        auroc_df.to_excel(output_file, index=False)
        print(f"\n✓ 完整结果已保存至 {output_file}")
    except Exception as e:
        print(f"\nWARNING: Failed to save {output_file}: {type(e).__name__}: {e}")
    auprc_cols = ['Combination', 'Views', 'Num_Views', 'Total_Dim', 'OurMethod_AUPRC', 'Train_Time']
    auprc_df = results_df[auprc_cols].copy()
    auprc_output_file = os.path.join(eval_dir, f"auprc_{DATA_NAME}_ourmethod_results.xlsx")
    try:
        auprc_df.to_excel(auprc_output_file, index=False)
        print(f"✓ AUPRC结果已保存至 {auprc_output_file}")
    except Exception as e:
        print(f"WARNING: Failed to save {auprc_output_file}: {type(e).__name__}: {e}")
    print("\n" + "=" * 80)
    print("分析")
    print("=" * 80)
    valid_results = results_df[results_df['OurMethod_AUC'].notna()]
    if not valid_results.empty:
        best_result = valid_results.loc[valid_results['OurMethod_AUC'].idxmax()]
        print(f"\n最佳视图组合: {best_result['Combination']}")
        print(f"  测试集AUC: {best_result['OurMethod_AUC']:.4f} AUPRC: {best_result['OurMethod_AUPRC']:.4f}")
        print(f"  视图: {best_result['Views']}")
        print(f"  潜在维度: {MODEL_CONFIG['latent_dim']} (对比拼接维度: {best_result['Total_Dim']:.0f})")
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)
def main_multi(seeds):
    seeds = [int(s) for s in seeds]
    print("=" * 80)
    print("自定义多视图方法 - 多随机种子")
    print("=" * 80)
    print(f"\n数据集: {DATA_NAME}")
    print(f"种子列表: {seeds}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    results_list = []
    failed_combos = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print("\n" + "=" * 80)
        print(f"正在评估: {combo_name}")
        print(f"视图: {view_names}")
        print("=" * 80)
        try:
            print("\n加载训练集数据（只包含正常数据）...")
            train_views_dict, train_labels, train_norm_stats = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='train',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            print("\n加载测试集数据（包含正常+异常数据）...")
            test_views_dict, test_labels, test_norm_stats = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='test',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            view_dims = {name: emb.shape[1] for name, emb in train_views_dict.items()}
            print(f"视图维度: {view_dims}")
            result_row = {
                'Combination': combo_name,
                'Views': '+'.join(view_names),
                'Num_Views': len(view_names),
                'Total_Dim': sum(view_dims.values()),
                'Seeds': ','.join(str(s) for s in seeds),
            }
            if len(view_names) > 1:
                auc_list = []
                auprc_list = []
                time_list = []
                for seed in seeds:
                    print("\n--- 单次运行 ---")
                    print(f"种子: {seed}")
                    auc, auprc, train_time = _train_and_eval_once(
                        train_views_dict=train_views_dict,
                        test_views_dict=test_views_dict,
                        test_labels=test_labels,
                        view_dims=view_dims,
                        device=device,
                        seed=seed,
                    )
                    auc_list.append(auc)
                    auprc_list.append(auprc)
                    time_list.append(train_time)
                    print(f"\n自定义方法 测试集AUC: {auc:.4f} AUPRC: {auprc:.4f}")
                    print(f"训练时间: {train_time:.2f}秒")
                auc_mean, auc_var = _mean_var(auc_list)
                auprc_mean, auprc_var = _mean_var(auprc_list)
                time_mean, time_var = _mean_var(time_list)
                result_row['OurMethod_AUC_Mean'] = auc_mean
                result_row['OurMethod_AUC_Var'] = auc_var
                result_row['OurMethod_AUPRC_Mean'] = auprc_mean
                result_row['OurMethod_AUPRC_Var'] = auprc_var
                result_row['Train_Time_Mean'] = time_mean
                result_row['Train_Time_Var'] = time_var
            else:
                result_row['OurMethod_AUC_Mean'] = None
                result_row['OurMethod_AUC_Var'] = None
                result_row['OurMethod_AUPRC_Mean'] = None
                result_row['OurMethod_AUPRC_Var'] = None
                result_row['Train_Time_Mean'] = None
                result_row['Train_Time_Var'] = None
            results_list.append(result_row)
        except Exception as e:
            print(f"\n处理 {combo_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_combos.append((combo_name, str(e)))
            continue
    if failed_combos:
        print(f"\nWARNING: {len(failed_combos)}/{len(VIEW_COMBINATIONS)} combinations failed:")
        for name, err in failed_combos:
            print(f"  - {name}: {err}")
    print("\n" + "=" * 80)
    print("汇总结果 (均值 / 方差分列)")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    display_cols = [
        'Combination', 'Views', 'Num_Views', 'Total_Dim',
        'OurMethod_AUC_Mean', 'OurMethod_AUC_Var',
        'OurMethod_AUPRC_Mean', 'OurMethod_AUPRC_Var',
        'Train_Time_Mean', 'Train_Time_Var',
        'Seeds'
    ]
    if not results_df.empty:
        print("\n" + results_df[display_cols].to_string(index=False))
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        seeds_tag = '_'.join(str(s) for s in seeds)
        auroc_cols = [
            'Combination', 'Views', 'Num_Views', 'Total_Dim',
            'OurMethod_AUC_Mean', 'OurMethod_AUC_Var',
            'Train_Time_Mean', 'Train_Time_Var',
            'Seeds'
        ]
        output_file = os.path.join(eval_dir, f"{DATA_NAME}_ourmethod_results_seeds_{seeds_tag}.xlsx")
        try:
            results_df[auroc_cols].to_excel(output_file, index=False)
            print(f"\n✓ 完整结果已保存至 {output_file}")
        except Exception as e:
            print(f"\nWARNING: Failed to save {output_file}: {type(e).__name__}: {e}")
        auprc_cols = [
            'Combination', 'Views', 'Num_Views', 'Total_Dim',
            'OurMethod_AUPRC_Mean', 'OurMethod_AUPRC_Var',
            'Train_Time_Mean', 'Train_Time_Var',
            'Seeds'
        ]
        auprc_df = results_df[auprc_cols].copy()
        auprc_output_file = os.path.join(eval_dir, f"auprc_{DATA_NAME}_ourmethod_results_seeds_{seeds_tag}.xlsx")
        try:
            auprc_df.to_excel(auprc_output_file, index=False)
            print(f"✓ AUPRC结果已保存至 {auprc_output_file}")
        except Exception as e:
            print(f"WARNING: Failed to save {auprc_output_file}: {type(e).__name__}: {e}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='自定义多视图方法评估 - 训练集/测试集分离')
    parser.add_argument('--dataset', type=str, default="olid,covid_fake,liar2,hate_speech,email_spam,smsspam,bbc,movie_review,N24News,agnews",
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
                        help='训练轮数 (默认: 自动根据数据集选择)')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率 (默认: 自动根据数据集选择)')
    parser.add_argument('--score_weight_recon', type=float, default=None,
                        help='异常分数中重构误差的权重 (默认: 自动根据数据集选择)')
    parser.add_argument('--score_weight_consistency', type=float, default=None,
                        help='异常分数中一致性分数的权重 (默认: 自动根据数据集选择)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch大小 (默认: None表示使用全部数据，推荐256/512/1024以避免OOM)')
    parser.set_defaults(use_auto_config=True)
    parser.add_argument('--use_auto_config', dest='use_auto_config', action='store_true',
                        help='使用数据集自适应配置（默认: 开启）')
    parser.add_argument('--no_auto_config', dest='use_auto_config', action='store_false',
                        help='禁用数据集自适应配置')
    args = parser.parse_args()
    seeds_list = _parse_int_list(args.seeds) if args.seeds is not None else None
    if seeds_list is None or len(seeds_list) == 0:
        seeds_list = [args.seed]
    dataset_list = [d.strip() for d in args.dataset.split(',') if d.strip()]
    for dataset_name in dataset_list:
        DATA_NAME = dataset_name
        TRAIN_CONFIG.clear()
        TRAIN_CONFIG.update(BASE_TRAIN_CONFIG)
        NUM_EPOCHS = BASE_NUM_EPOCHS
        LEARNING_RATE = BASE_LEARNING_RATE
        if args.use_auto_config and USE_DATASET_CONFIGS and dataset_name in DATASET_CONFIGS:
            dataset_config = print_dataset_config(dataset_name)
            NUM_EPOCHS = args.epochs if args.epochs is not None else dataset_config['num_epochs']
            LEARNING_RATE = args.lr if args.lr is not None else dataset_config['learning_rate']
            TRAIN_CONFIG['lambda_recon'] = args.lambda_recon if args.lambda_recon != 1.0 else dataset_config['lambda_recon']
            TRAIN_CONFIG['lambda_contrastive'] = args.lambda_contrastive if args.lambda_contrastive != 1.0 else dataset_config['lambda_contrastive']
            TRAIN_CONFIG['score_weight_recon'] = args.score_weight_recon if args.score_weight_recon is not None else dataset_config.get('score_weight_recon', 0.3)
            TRAIN_CONFIG['score_weight_consistency'] = args.score_weight_consistency if args.score_weight_consistency is not None else dataset_config.get('score_weight_consistency', 0.4)
            TRAIN_CONFIG['score_weight_knn'] = dataset_config.get('score_weight_knn', TRAIN_CONFIG.get('score_weight_knn', 0.0))
            TRAIN_CONFIG['batch_size'] = args.batch_size if args.batch_size is not None else dataset_config.get('batch_size', None)
        else:
            NUM_EPOCHS = args.epochs if args.epochs is not None else NUM_EPOCHS
            LEARNING_RATE = args.lr if args.lr is not None else LEARNING_RATE
            TRAIN_CONFIG['lambda_recon'] = args.lambda_recon
            TRAIN_CONFIG['lambda_contrastive'] = args.lambda_contrastive
            TRAIN_CONFIG['score_weight_recon'] = args.score_weight_recon if args.score_weight_recon is not None else 0.3
            TRAIN_CONFIG['score_weight_consistency'] = args.score_weight_consistency if args.score_weight_consistency is not None else 0.4
            TRAIN_CONFIG['batch_size'] = args.batch_size
        if len(seeds_list) == 1:
            main(seed=seeds_list[0])
        else:
            main_multi(seeds=seeds_list)

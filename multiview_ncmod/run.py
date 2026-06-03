
import argparse
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import random
import time
import os
import sys
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"✓ 工作目录已切换到: {os.getcwd()}")
from multiview_ncmod.dataset import load_all_views
from multiview_ncmod.model import NCMODModel
from multiview_ncmod.trainer import NCMODTrainer, get_new_knn, cal_scores, cal_final_scores
from sklearn.metrics import roc_auc_score, average_precision_score
DATA_NAME = 'bbc'
VIEW_COMBINATIONS = {
    "O-ada+small+large": ['openai_ada', 'openai_small', 'openai_large'],
    "multi-view": ['openai_large', 'bert', 'qwen', 'llama'],
}
DEFAULT_CONFIG = {
    'num_rounds': 10,
    'num_epochs': 16,
    'batch_size': 20,
    'learning_rate': 0.0001,
    'k_neibs': 8,
    'module_weight': 1.0,
}
DATASET_CONFIGS = {
    'olid': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.001,
        'k_neibs': 8,
    },
    'bbc': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'k_neibs': 8,
    },
    'covid_fake': {
        'num_rounds': 10,
        'num_epochs': 10,
        'batch_size': 20,
        'learning_rate': 0.001,
        'k_neibs': 8,
    },
    'liar2': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.01,
        'k_neibs': 8,
    },
    'hate_speech': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.0001,
        'k_neibs': 8,
    },
    'email_spam': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.01,
        'k_neibs': 8,
    },
    'smsspam': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 32,
        'learning_rate': 0.1,
        'k_neibs': 8,
    },
    'movie_review': {
        'num_rounds': 10,
        'num_epochs': 10,
        'batch_size': 256,
        'learning_rate': 0.01,
        'k_neibs': 8,
    },
    'N24News': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 256,
        'learning_rate': 0.01,
        'k_neibs': 8,
    },
    'agnews': {
        'num_rounds': 5,
        'num_epochs': 10,
        'batch_size': 256,
        'learning_rate': 0.01,
        'k_neibs': 8,
    },
}
NCMOD_CONFIG = DEFAULT_CONFIG.copy()
def get_dataset_config(data_name, use_dataset_config=True):
    if use_dataset_config and data_name in DATASET_CONFIGS:
        config = DEFAULT_CONFIG.copy()
        config.update(DATASET_CONFIGS[data_name])
        print(f"\n📋 使用数据集 '{data_name}' 的特定配置:")
        print(f"   num_rounds={config['num_rounds']}, num_epochs={config['num_epochs']}, "
              f"batch_size={config['batch_size']}, lr={config['learning_rate']}, k_neibs={config['k_neibs']}")
        return config
    else:
        if use_dataset_config:
            print(f"\n⚠️  数据集 '{data_name}' 无特定配置，使用默认配置")
        else:
            print(f"\n📋 使用默认配置（已禁用数据集特定配置）")
        return DEFAULT_CONFIG.copy()
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"随机数种子已设置为: {seed}")
def run_ncmod(data_name, view_names, config, device='cpu'):
    print(f"\n加载数据集: {data_name}")
    print(f"视图: {view_names}")
    dataset, views_dict, labels = load_all_views(
        data_name=data_name,
        view_names=view_names,
        split='test',
        embeddings_dir='embeddings',
        normalize=True,
        device=device,
        num_neibs=config['k_neibs']
    )
    view_dims = {name: emb.shape[1] for name, emb in views_dict.items()}
    num_views = len(view_names)
    print(f"视图维度: {view_dims}")
    model = NCMODModel(view_dims=view_dims, latent_dim=32, device=device)
    trainer = NCMODTrainer(
        model=model,
        lr=config['learning_rate'],
        n_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        weight_decay=1e-6,
        device=device,
        module_weight=config['module_weight']
    )
    view_encoded = [None for _ in range(num_views)]
    recon_error = [None for _ in range(num_views)]
    start_time = time.time()
    best_auc = 0
    best_auprc = 0
    best_round = 0
    print(f"\n开始训练 {config['num_rounds']} 轮...")
    for eround in range(config['num_rounds']):
        round_start = time.time()
        for id_view, view_name in enumerate(view_names):
            _, view_encoded[id_view], recon_error[id_view] = trainer.train_single_view(
                eround, id_view, dataset, view_name, pre_train=False
            )
        neibs_global, neibs_local, weights_global = get_new_knn(view_encoded, config['k_neibs'])
        for id_view in range(num_views):
            dataset.set_knn(neibs_global[id_view], neibs_local[id_view], weights_global[id_view])
        recon_scores, knn_scores = cal_scores(view_encoded, config['k_neibs'], recon_error)
        total_scores = cal_final_scores(recon_scores, knn_scores)
        labels_np = labels.cpu().numpy()
        current_auc = roc_auc_score(labels_np, total_scores)
        current_auprc = average_precision_score(labels_np, total_scores)
        round_time = time.time() - round_start
        elapsed_time = time.time() - start_time
        print(f"Round {eround+1}/{config['num_rounds']} | "
              f"AUC: {current_auc:.4f} | "
              f"AUPRC: {current_auprc:.4f} | "
              f"Round Time: {round_time:.1f}s | "
              f"Total Time: {elapsed_time:.1f}s")
        if current_auc > best_auc:
            best_auc = current_auc
            best_auprc = current_auprc
            best_round = eround + 1
    train_time = time.time() - start_time
    print(f"\n训练完成！最佳AUC: {best_auc:.4f} | 最佳AUPRC: {best_auprc:.4f} (Round {best_round})")
    return best_auc, best_auprc, train_time
def run_single_experiment(data_name, seed, device, use_dataset_config=True):
    set_random_seed(seed)
    print(f"\n{'='*60}")
    print(f"数据集: {data_name} | 种子: {seed}")
    print(f"{'='*60}")
    config = get_dataset_config(data_name, use_dataset_config)
    results_list = []
    failed_combos = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print(f"\n正在评估: {combo_name}")
        print(f"视图: {view_names}")
        try:
            auc, auprc, train_time = run_ncmod(
                data_name=data_name,
                view_names=view_names,
                config=config,
                device=device
            )
            print(f"✓ {combo_name} 完成 - AUC: {auc:.4f}, AUPRC: {auprc:.4f}")
            result_row = {
                'Combination': combo_name,
                'NCMOD_AUC': auc,
                'NCMOD_AUPRC': auprc,
                'Train_Time': train_time
            }
            results_list.append(result_row)
        except Exception as e:
            print(f"✗ 处理 {combo_name} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_combos.append((combo_name, str(e)))
            continue
    if failed_combos:
        print(f"\nWARNING: {len(failed_combos)}/{len(VIEW_COMBINATIONS)} combinations failed:")
        for name, err in failed_combos:
            print(f"  - {name}: {err}")
    return results_list
def main(datasets, seeds, use_dataset_config=True):
    print("=" * 80)
    print("NCMOD 基线模型评估 - 多视图组合")
    print("=" * 80)
    print(f"\n数据集: {datasets}")
    print(f"种子: {seeds}")
    print(f"数据集特定配置: {'启用' if use_dataset_config else '禁用'}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    all_results = {}
    for data_name in datasets:
        print(f"\n\n{'='*80}")
        print(f"开始评估数据集: {data_name}")
        print(f"{'='*80}")
        dataset_results = {}
        for seed in seeds:
            results_list = run_single_experiment(data_name, seed, device, use_dataset_config)
            for result in results_list:
                combo_name = result['Combination']
                if combo_name not in dataset_results:
                    dataset_results[combo_name] = []
                dataset_results[combo_name].append(result)
        aggregated_results = []
        aggregated_results_auprc = []
        for combo_name, combo_results in dataset_results.items():
            aucs = [r['NCMOD_AUC'] for r in combo_results]
            auprcs = [r['NCMOD_AUPRC'] for r in combo_results]
            times = [r['Train_Time'] for r in combo_results]
            aggregated_results.append({
                'Combination': combo_name,
                'AUC_Mean': np.mean(aucs),
                'AUC_Std': np.std(aucs, ddof=1) if len(aucs) > 1 else 0,
                'Time_Mean': np.mean(times),
                'Time_Std': np.std(times, ddof=1) if len(times) > 1 else 0,
            })
            aggregated_results_auprc.append({
                'Combination': combo_name,
                'AUPRC_Mean': np.mean(auprcs),
                'AUPRC_Std': np.std(auprcs, ddof=1) if len(auprcs) > 1 else 0,
                'Time_Mean': np.mean(times),
                'Time_Std': np.std(times, ddof=1) if len(times) > 1 else 0,
            })
        all_results[data_name] = aggregated_results
        print(f"\n{'='*80}")
        print(f"数据集 {data_name} 汇总结果 (基于 {len(seeds)} 个种子)")
        print(f"{'='*80}")
        results_df = pd.DataFrame(aggregated_results)
        print("\n" + results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        output_file = f'multiview_ncmod/NCMOD_{data_name}_results.xlsx'
        try:
            results_df.to_excel(output_file, index=False, float_format='%.4f')
            print(f"\n✓ AUROC结果已保存至 {output_file}")
        except Exception as e:
            print(f"\nWARNING: Failed to save {output_file}: {type(e).__name__}: {e}")
        results_df_auprc = pd.DataFrame(aggregated_results_auprc)
        output_file_auprc = f'multiview_ncmod/auprc_NCMOD_{data_name}_results.xlsx'
        try:
            results_df_auprc.to_excel(output_file_auprc, index=False, float_format='%.4f')
            print(f"✓ AUPRC结果已保存至 {output_file_auprc}")
        except Exception as e:
            print(f"WARNING: Failed to save {output_file_auprc}: {type(e).__name__}: {e}")
        if not results_df.empty:
            best_idx = results_df['AUC_Mean'].idxmax()
            best_result = results_df.iloc[best_idx]
            print(f"\n最佳视图组合 (AUROC): {best_result['Combination']}")
            print(f"  AUC: {best_result['AUC_Mean']:.4f} ± {best_result['AUC_Std']:.4f}")
            print(f"  训练时间: {best_result['Time_Mean']:.2f} ± {best_result['Time_Std']:.2f}秒")
        if not results_df_auprc.empty:
            best_idx_auprc = results_df_auprc['AUPRC_Mean'].idxmax()
            best_result_auprc = results_df_auprc.iloc[best_idx_auprc]
            print(f"\n最佳视图组合 (AUPRC): {best_result_auprc['Combination']}")
            print(f"  AUPRC: {best_result_auprc['AUPRC_Mean']:.4f} ± {best_result_auprc['AUPRC_Std']:.4f}")
            print(f"  训练时间: {best_result_auprc['Time_Mean']:.2f} ± {best_result_auprc['Time_Std']:.2f}秒")
    print("\n" + "=" * 80)
    print("所有评估完成！")
    print("=" * 80)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NCMOD基线模型评估 - 多视图组合')
    parser.add_argument('--datasets', type=str,
                        default=",".join(DATASET_CONFIGS.keys()),
                        help='数据集名称，支持逗号分隔多个 (例如: bbc,olid,agnews)')
    parser.add_argument('--seeds', type=str, default="42",
                        help='随机数种子，支持逗号分隔多个 (例如: 41,42,43,44,45)')
    parser.add_argument('--num_rounds', type=int, default=10,
                        help='训练轮数 (默认: 10)')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='每轮epoch数 (默认: 16)')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='批次大小 (默认: 20)')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率 (默认: 0.0001)')
    parser.add_argument('--k_neibs', type=int, default=8,
                        help='邻居数量 (默认: 8)')
    parser.add_argument('--use_dataset_config', action='store_true', default=True,
                        help='使用数据集特定配置 (默认: True)')
    parser.add_argument('--no_dataset_config', dest='use_dataset_config', action='store_false',
                        help='禁用数据集特定配置，使用全局配置')
    args = parser.parse_args()
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    if not args.use_dataset_config:
        if args.num_rounds != 10:
            DEFAULT_CONFIG['num_rounds'] = args.num_rounds
        if args.num_epochs != 16:
            DEFAULT_CONFIG['num_epochs'] = args.num_epochs
        if args.batch_size != 20:
            DEFAULT_CONFIG['batch_size'] = args.batch_size
        if args.learning_rate != 0.0001:
            DEFAULT_CONFIG['learning_rate'] = args.learning_rate
        if args.k_neibs != 8:
            DEFAULT_CONFIG['k_neibs'] = args.k_neibs
        print(f"\n⚙️  使用命令行全局配置: num_rounds={DEFAULT_CONFIG['num_rounds']}, "
              f"num_epochs={DEFAULT_CONFIG['num_epochs']}, batch_size={DEFAULT_CONFIG['batch_size']}, "
              f"lr={DEFAULT_CONFIG['learning_rate']}, k_neibs={DEFAULT_CONFIG['k_neibs']}")
    main(datasets=datasets, seeds=seeds, use_dataset_config=args.use_dataset_config)

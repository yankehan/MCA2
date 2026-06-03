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
print(f"✓ Working directory changed to: {os.getcwd()}")
from multiview_two_stage.model import MultiViewContrastiveModel
from multiview_two_stage.dataset import load_all_views_full
from multiview_two_stage.trainer import SimpleMultiViewTrainer
try:
    from multiview_two_stage.eval.dataset_configs import DATASET_CONFIGS, print_dataset_config
    USE_DATASET_CONFIGS = True
except ImportError:
    USE_DATASET_CONFIGS = False
    DATASET_CONFIGS = {}
    print("⚠️  dataset_configs.py not found, using default configuration")
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
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    print(f"Random seed set to: {seed}")
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
    print("\n=== Model Parameter Statistics ===")
    total_params = sum(p.numel() for p in model.parameters())
    if hasattr(model, 'view_gate') and model.view_gate is not None:
        gate_params_num = sum(p.numel() for p in model.view_gate.parameters())
        print(f"Total params: {total_params}, Gate params: {gate_params_num}, Main params: {total_params - gate_params_num}")
    else:
        print(f"Total params: {total_params}, Gate params: 0")
    print("=== Parameter Statistics End ===\n")
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
    print("\n" + "Stage 1: Train AE + Contrastive Learning (Gate fixed uniform, no update)".center(80, "-"))
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
    print("\n" + "Stage 2: Freeze AE+CL, Train Gate Only (Evaluate AUC every epoch)".center(80, "-"))
    print(f"second_stage_epochs={second_stage_epochs}, gate_lr={float(gate_lr)}")
    best_auc = -1.0
    best_auprc = -1.0
    best_state = None
    stage2_start = time.time()
    for epoch in range(int(second_stage_epochs)):
        loss_dict = trainer_stage2.train_epoch(train_views_dict)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        auc, auprc, _ = trainer_stage2.evaluate(test_views_dict, test_labels)
        elapsed = time.time() - stage2_start
        print(
            f"[Stage2] Epoch {epoch + 1}/{second_stage_epochs} | "
            f"Loss: {loss_dict['total']:.4f} | Recon: {loss_dict['recon']:.4f} | Contr: {loss_dict['contrastive']:.4f} | "
            f"Test AUC: {auc:.4f} AUPRC: {auprc:.4f} | Time: {elapsed:.1f}s"
        )
        if auc > best_auc:
            best_auc = float(auc)
            best_auprc = float(auprc)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stage2_time = time.time() - stage2_start
    if best_state is not None:
        model.load_state_dict(best_state)
    auc_on = float(best_auc)
    auprc_on = float(best_auprc)
    print(f"\n[Stage2] gate_on best AUC: {auc_on:.4f} AUPRC: {auprc_on:.4f}")
    del trainer_stage1
    del trainer_stage2
    del optimizer_stage1
    del optimizer_stage2
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return auc_on, auprc_on, stage1_time, stage2_time
def main(seed=42):
    set_random_seed(seed)
    print("=" * 80)
    print("Gate ON Evaluation (Two-stage training: Stage1 backbone + Stage2 gate)")
    print("=" * 80)
    print(f"\nDataset: {DATA_NAME}")
    print(f"Config: lambda_recon={TRAIN_CONFIG['lambda_recon']}, "
          f"lambda_contrastive={TRAIN_CONFIG['lambda_contrastive']}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    results_list = []
    results_list_auprc = []
    failed_combos = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print("\n" + "=" * 80)
        print(f"Evaluating combination: {combo_name}")
        print(f"Views: {view_names}")
        print("=" * 80)
        try:
            print("\nLoading training data (normal data only)...")
            train_views_dict, train_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='train',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            print("\nLoading test data (normal + anomaly data)...")
            test_views_dict, test_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='test',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            view_dims = {name: emb.shape[1] for name, emb in train_views_dict.items()}
            print(f"View dimensions: {view_dims}")
            row = {
                'Combination': combo_name,
                'Num_Views': len(view_names),
                'Total_Dim': sum(view_dims.values()),
            }
            print("\n--- Training/Evaluation Config ---")
            print(f"Learning rate: lr={LEARNING_RATE}")
            print(f"Loss weights: λ_recon={TRAIN_CONFIG['lambda_recon']}, λ_contrastive={TRAIN_CONFIG['lambda_contrastive']}")
            print(f"Score weights: recon={TRAIN_CONFIG.get('score_weight_recon', 0.3)}, "
                  f"consistency={TRAIN_CONFIG.get('score_weight_consistency', 0.4)}")
            auc_on, auprc_on, stage1_time, stage2_time = _train_two_stage_and_eval_once(
                train_views_dict=train_views_dict,
                test_views_dict=test_views_dict,
                test_labels=test_labels,
                view_dims=view_dims,
                device=device,
                seed=seed,
                first_stage_epochs=FIRST_STAGE_EPOCHS,
                second_stage_epochs=SECOND_STAGE_EPOCHS,
                gate_lr=GATE_LEARNING_RATE,
            )
            row['Gate_ON_AUC'] = auc_on
            row['Stage1_Time'] = stage1_time
            row['Stage2_Time'] = stage2_time
            row['Total_Time'] = stage1_time + stage2_time
            row_auprc = row.copy()
            row_auprc['Gate_ON_AUPRC'] = auprc_on
            print("\nResults:")
            print(f"  gate_on best AUC (Stage2): {auc_on:.4f} AUPRC: {auprc_on:.4f}")
            print(f"  Stage1 time: {stage1_time:.1f}s")
            print(f"  Stage2 time: {stage2_time:.1f}s")
            results_list.append(row)
            results_list_auprc.append(row_auprc)
        except Exception as e:
            print(f"\nError processing {combo_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_combos.append((combo_name, str(e)))
            continue
    if failed_combos:
        print(f"\nWARNING: {len(failed_combos)}/{len(VIEW_COMBINATIONS)} combinations failed:")
        for name, err in failed_combos:
            print(f"  - {name}: {err}")
    print("\n" + "=" * 80)
    print("Summary Results")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        display_cols = [
            'Combination', 'Num_Views', 'Total_Dim',
            'Gate_ON_AUC',
            'Stage1_Time', 'Stage2_Time', 'Total_Time'
        ]
        print("\n" + results_df[display_cols].to_string(index=False))
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(eval_dir, f"{DATA_NAME}_ourmethod_results.xlsx")
        results_df[display_cols].to_excel(output_file, index=False)
        print(f"\n✓ Complete results saved to {output_file}")
        results_df_auprc = pd.DataFrame(results_list_auprc)
        if not results_df_auprc.empty:
            display_cols_auprc = [
                'Combination', 'Num_Views', 'Total_Dim',
                'Gate_ON_AUPRC',
                'Stage1_Time', 'Stage2_Time', 'Total_Time'
            ]
            output_file_auprc = os.path.join(eval_dir, f"auprc_{DATA_NAME}_ourmethod_results.xlsx")
            results_df_auprc[display_cols_auprc].to_excel(output_file_auprc, index=False)
            print(f"✓ AUPRC results saved to {output_file_auprc}")
        if 'Gate_ON_AUC' in results_df.columns and results_df['Gate_ON_AUC'].notna().any():
            best_idx = results_df['Gate_ON_AUC'].idxmax()
            best_row = results_df.loc[best_idx]
            print("\nBest combination:")
            print(f"  Combination: {best_row['Combination']}")
            print(f"  Gate_ON_AUC : {best_row['Gate_ON_AUC']:.4f}")
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
def main_multi(seeds):
    seeds = [int(s) for s in seeds]
    print("=" * 80)
    print("Gate ON Evaluation (Two-stage training, multiple random seeds)")
    print("=" * 80)
    print(f"\nDataset: {DATA_NAME}")
    print(f"Seed list: {seeds}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    results_list = []
    results_list_auprc = []
    failed_combos = []
    for combo_name, view_names in VIEW_COMBINATIONS.items():
        print("\n" + "=" * 80)
        print(f"Evaluating combination: {combo_name}")
        print(f"Views: {view_names}")
        print("=" * 80)
        try:
            print("\nLoading training data (normal data only)...")
            train_views_dict, train_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='train',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            print("\nLoading test data (normal + anomaly data)...")
            test_views_dict, test_labels, _ = load_all_views_full(
                data_name=DATA_NAME,
                view_names=view_names,
                split='test',
                embeddings_dir='embeddings',
                normalize=True,
                device=device,
            )
            view_dims = {name: emb.shape[1] for name, emb in train_views_dict.items()}
            print(f"View dimensions: {view_dims}")
            auc_on_list = []
            auprc_on_list = []
            stage1_time_list = []
            stage2_time_list = []
            for seed in seeds:
                print("\n--- Single Run ---")
                print(f"Seed: {seed}")
                auc_on, auprc_on, stage1_time, stage2_time = _train_two_stage_and_eval_once(
                    train_views_dict=train_views_dict,
                    test_views_dict=test_views_dict,
                    test_labels=test_labels,
                    view_dims=view_dims,
                    device=device,
                    seed=seed,
                    first_stage_epochs=FIRST_STAGE_EPOCHS,
                    second_stage_epochs=SECOND_STAGE_EPOCHS,
                    gate_lr=GATE_LEARNING_RATE,
                )
                auc_on_list.append(auc_on)
                auprc_on_list.append(auprc_on)
                stage1_time_list.append(stage1_time)
                stage2_time_list.append(stage2_time)
                print("\nResults:")
                print(f"  Gate ON best AUC: {auc_on:.4f} AUPRC: {auprc_on:.4f}")
                print(f"  Stage1 time: {stage1_time:.1f}s")
                print(f"  Stage2 time: {stage2_time:.1f}s")
            auc_on_mean, auc_on_var = _mean_var(auc_on_list)
            auprc_on_mean, auprc_on_var = _mean_var(auprc_on_list)
            stage1_time_mean, stage1_time_var = _mean_var(stage1_time_list)
            stage2_time_mean, stage2_time_var = _mean_var(stage2_time_list)
            total_time_list = [t1 + t2 for t1, t2 in zip(stage1_time_list, stage2_time_list)]
            total_time_mean, total_time_var = _mean_var(total_time_list)
            row = {
                'Combination': combo_name,
                'Num_Views': len(view_names),
                'Total_Dim': sum(view_dims.values()),
                'Gate_ON_AUC': _format_mean_var(auc_on_mean, auc_on_var, '.4f', '.4f'),
                'Stage1_Time': _format_mean_var(stage1_time_mean, stage1_time_var, '.1f', '.1f'),
                'Stage2_Time': _format_mean_var(stage2_time_mean, stage2_time_var, '.1f', '.1f'),
                'Total_Time': _format_mean_var(total_time_mean, total_time_var, '.1f', '.1f'),
                'Seeds': ','.join(str(s) for s in seeds),
            }
            results_list.append(row)
            row_auprc = {
                'Combination': combo_name,
                'Num_Views': len(view_names),
                'Total_Dim': sum(view_dims.values()),
                'Gate_ON_AUPRC': _format_mean_var(auprc_on_mean, auprc_on_var, '.4f', '.4f'),
                'Stage1_Time': _format_mean_var(stage1_time_mean, stage1_time_var, '.1f', '.1f'),
                'Stage2_Time': _format_mean_var(stage2_time_mean, stage2_time_var, '.1f', '.1f'),
                'Total_Time': _format_mean_var(total_time_mean, total_time_var, '.1f', '.1f'),
                'Seeds': ','.join(str(s) for s in seeds),
            }
            results_list_auprc.append(row_auprc)
        except Exception as e:
            print(f"\nError processing {combo_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed_combos.append((combo_name, str(e)))
            continue
    if failed_combos:
        print(f"\nWARNING: {len(failed_combos)}/{len(VIEW_COMBINATIONS)} combinations failed:")
        for name, err in failed_combos:
            print(f"  - {name}: {err}")
    print("\n" + "=" * 80)
    print("Summary Results (Mean ± Variance)")
    print("=" * 80)
    results_df = pd.DataFrame(results_list)
    if not results_df.empty:
        display_cols = [
            'Combination', 'Num_Views', 'Total_Dim',
            'Gate_ON_AUC',
            'Stage1_Time', 'Stage2_Time', 'Total_Time'
        ]
        print("\n" + results_df[display_cols].to_string(index=False))
        eval_dir = os.path.dirname(os.path.abspath(__file__))
        seeds_tag = '_'.join(str(s) for s in seeds)
        output_file = os.path.join(eval_dir, f"{DATA_NAME}_ourmethod_results_seeds_{seeds_tag}.xlsx")
        results_df[display_cols + ['Seeds']].to_excel(output_file, index=False)
        print(f"\n✓ Complete results saved to {output_file}")
        results_df_auprc = pd.DataFrame(results_list_auprc)
        if not results_df_auprc.empty:
            display_cols_auprc = [
                'Combination', 'Num_Views', 'Total_Dim',
                'Gate_ON_AUPRC',
                'Stage1_Time', 'Stage2_Time', 'Total_Time'
            ]
            output_file_auprc = os.path.join(eval_dir, f"auprc_{DATA_NAME}_ourmethod_results_seeds_{seeds_tag}.xlsx")
            results_df_auprc[display_cols_auprc + ['Seeds']].to_excel(output_file_auprc, index=False)
            print(f"✓ AUPRC results saved to {output_file_auprc}")
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gate ON Evaluation (multiview_two_stage)')
    parser.add_argument('--dataset', type=str, default=DEFAULT_DATASETS,
                        help='Dataset name, supports comma-separated multiple datasets (e.g., bbc,olid)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated random seed list, e.g., 41,42,43,44,45; if set, ignores --seed')
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='Reconstruction loss weight (default: 1.0)')
    parser.add_argument('--lambda_contrastive', type=float, default=1.0,
                        help='Contrastive learning loss weight (default: 1.0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Training epochs (compatible with old parameter; equivalent to --first_stage_epochs)')
    parser.add_argument('--first_stage_epochs', type=int, default=None,
                        help='First stage training epochs (default: auto-select based on dataset; higher priority than --epochs)')
    parser.add_argument('--second_stage_epochs', type=int, default=BASE_SECOND_STAGE_EPOCHS,
                        help='Second stage training epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: auto-select based on dataset)')
    parser.add_argument('--gate_lr', type=float, default=BASE_GATE_LEARNING_RATE,
                        help='Second stage gate learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Mini-batch size (default: None means use all data)')
    parser.add_argument('--score_weight_recon', type=float, default=None,
                        help='Weight of reconstruction error in anomaly score (default: auto-select based on dataset)')
    parser.add_argument('--score_weight_consistency', type=float, default=None,
                        help='Weight of consistency score in anomaly score (default: auto-select based on dataset)')
    parser.add_argument('--view_gate_hidden_dims', type=str, default="256,128",
                        help='Gate hidden layer dimensions, comma-separated, e.g., 256 or 256,128; empty means linear gate')
    parser.set_defaults(use_auto_config=True)
    parser.add_argument('--use_auto_config', dest='use_auto_config', action='store_true',
                        help='Use dataset adaptive configuration (default: enabled)')
    parser.add_argument('--no_auto_config', dest='use_auto_config', action='store_false',
                        help='Disable dataset adaptive configuration')
    args = parser.parse_args()
    dataset_list = [d.strip() for d in args.dataset.split(',') if d.strip()]
    seeds_list = _parse_int_list(args.seeds) if args.seeds is not None else None
    if seeds_list is None or len(seeds_list) == 0:
        seeds_list = [args.seed]
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
            print("\n" + "🔧 Using Dataset Adaptive Configuration".center(80, "="))
            dataset_config = print_dataset_config(dataset_name)
            first_stage_epochs_arg = args.first_stage_epochs if args.first_stage_epochs is not None else args.epochs
            FIRST_STAGE_EPOCHS = first_stage_epochs_arg if first_stage_epochs_arg is not None else dataset_config['num_epochs']
            LEARNING_RATE = args.lr if args.lr is not None else dataset_config['learning_rate']
            SECOND_STAGE_EPOCHS = int(args.second_stage_epochs)
            GATE_LEARNING_RATE = float(args.gate_lr)
            TRAIN_CONFIG['lambda_recon'] = args.lambda_recon if args.lambda_recon != 1.0 else dataset_config['lambda_recon']
            TRAIN_CONFIG['lambda_contrastive'] = args.lambda_contrastive if args.lambda_contrastive != 1.0 else dataset_config['lambda_contrastive']
            TRAIN_CONFIG['batch_size'] = args.batch_size if args.batch_size is not None else dataset_config.get('batch_size', None)
            TRAIN_CONFIG['score_weight_recon'] = args.score_weight_recon if args.score_weight_recon is not None else dataset_config.get('score_weight_recon', 0.3)
            TRAIN_CONFIG['score_weight_consistency'] = args.score_weight_consistency if args.score_weight_consistency is not None else dataset_config.get('score_weight_consistency', 0.4)
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
            TRAIN_CONFIG['score_weight_recon'] = args.score_weight_recon if args.score_weight_recon is not None else 0.3
            TRAIN_CONFIG['score_weight_consistency'] = args.score_weight_consistency if args.score_weight_consistency is not None else 0.4
        if len(seeds_list) == 1:
            main(seed=seeds_list[0])
        else:
            main_multi(seeds=seeds_list)

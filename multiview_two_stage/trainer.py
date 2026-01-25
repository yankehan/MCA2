
import time
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from .loss import contrastive_loss, contrastive_loss_per_sample, contrastive_score
import gc
class SimpleMultiViewTrainer:
    def __init__(self, model, optimizer, device='cpu', config=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = {
            'lambda_recon': 1.0,
            'lambda_contrastive': 1.0,
            'temperature': 0.5,
            'score_weight_recon': 0.3,
            'score_weight_consistency': 0.4,
            'batch_size': None,
            'print_gate_weight_each_epoch': True,
        }
        if config is not None:
            self.config.update(config)
        self.train_history = {
            'loss_total': [],
            'loss_recon': [],
            'loss_contrastive': [],
        }
    def compute_loss(self, views_dict, z_dict, recon_dict):
        loss_dict = {}
        view_names = list(views_dict.keys())
        batch_size = list(views_dict.values())[0].shape[0]
        view_weights = None
        view_weights_dict = None
        if hasattr(self.model, 'compute_view_weights'):
            out = self.model.compute_view_weights(views_dict)
            if out is not None:
                view_weights, view_weights_dict = out
        if view_weights_dict is None:
            loss_recon = 0
            for view_name in view_names:
                loss_recon += F.mse_loss(recon_dict[view_name], views_dict[view_name])
            loss_recon /= len(view_names)
            loss_contrastive = 0
            if self.config['lambda_contrastive'] > 0:
                for i in range(len(view_names)):
                    for j in range(i + 1, len(view_names)):
                        z_i = F.normalize(z_dict[view_names[i]], dim=1)
                        z_j = F.normalize(z_dict[view_names[j]], dim=1)
                        loss_contrastive += contrastive_loss(
                            z_i, z_j, batch_size,
                            temperature=self.config['temperature']
                        )
                n_pairs = len(view_names) * (len(view_names) - 1) / 2
                loss_contrastive /= n_pairs
            loss_dict['recon'] = loss_recon.item()
            loss_dict['contrastive'] = loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0.0
            loss_total = (
                self.config['lambda_recon'] * loss_recon +
                self.config['lambda_contrastive'] * loss_contrastive
            )
            loss_dict['total'] = loss_total.item()
            return loss_total, loss_dict
        recon_losses = []
        for view_name in view_names:
            per_sample_recon = F.mse_loss(
                recon_dict[view_name],
                views_dict[view_name],
                reduction='none'
            ).mean(dim=1)
            recon_losses.append(per_sample_recon)
        recon_losses_mat = torch.stack(recon_losses, dim=1)
        weights_mat = torch.stack([view_weights_dict[v] for v in view_names], dim=1)
        loss_recon_per_sample = (weights_mat * recon_losses_mat).sum(dim=1)
        loss_recon = loss_recon_per_sample.mean()
        loss_dict['recon'] = loss_recon.item()
        loss_contrastive = torch.tensor(0.0, device=loss_recon.device)
        if self.config['lambda_contrastive'] > 0:
            per_view_contrastive = {}
            counts = {}
            for v in view_names:
                per_view_contrastive[v] = torch.zeros(batch_size, device=loss_recon.device)
                counts[v] = 0
            for i in range(len(view_names)):
                for j in range(i + 1, len(view_names)):
                    vi = view_names[i]
                    vj = view_names[j]
                    z_i = F.normalize(z_dict[vi], dim=1)
                    z_j = F.normalize(z_dict[vj], dim=1)
                    pair_loss = contrastive_loss_per_sample(
                        z_i, z_j, batch_size,
                        temperature=self.config['temperature']
                    )
                    per_view_contrastive[vi] += pair_loss
                    per_view_contrastive[vj] += pair_loss
                    counts[vi] += 1
                    counts[vj] += 1
            per_view_loss_list = []
            for v in view_names:
                denom = counts[v] if counts[v] > 0 else 1
                per_view_loss_list.append(per_view_contrastive[v] / denom)
            per_view_loss_mat = torch.stack(per_view_loss_list, dim=1)
            loss_contrastive_per_sample = (weights_mat * per_view_loss_mat).sum(dim=1)
            loss_contrastive = loss_contrastive_per_sample.mean()
        loss_dict['contrastive'] = loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0.0
        loss_total = (
            self.config['lambda_recon'] * loss_recon +
            self.config['lambda_contrastive'] * loss_contrastive
        )
        loss_dict['total'] = loss_total.item()
        return loss_total, loss_dict
    def train_epoch(self, views_dict):
        train_mode = self.config.get('train_mode', 'all')
        if train_mode == 'gate_only':
            self.model.eval()
            if hasattr(self.model, 'view_gate') and self.model.view_gate is not None:
                self.model.view_gate.train()
        else:
            self.model.train()
        if self.config.get('print_gate_weight_each_epoch', False):
            if hasattr(self.model, 'use_view_gate') and bool(getattr(self.model, 'use_view_gate', False)):
                if hasattr(self.model, 'view_gate') and self.model.view_gate is not None:
                    self.model.view_gate.debug_print = True
        batch_size = self.config.get('batch_size', None)
        if batch_size is None:
            z_dict, recon_dict = self.model(views_dict)
            loss_total, loss_dict = self.compute_loss(views_dict, z_dict, recon_dict)
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            for key, value in loss_dict.items():
                if key in self.train_history:
                    self.train_history[key].append(value)
            return loss_dict
        n_samples = list(views_dict.values())[0].shape[0]
        indices = torch.randperm(n_samples)
        epoch_losses = {'total': 0.0, 'recon': 0.0, 'contrastive': 0.0}
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_views = {
                view_name: views_dict[view_name][batch_indices]
                for view_name in views_dict.keys()
            }
            z_dict, recon_dict = self.model(batch_views)
            loss_total, loss_dict = self.compute_loss(batch_views, z_dict, recon_dict)
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict[key]
            n_batches += 1
        avg_loss_dict = {key: value / n_batches for key, value in epoch_losses.items()}
        for key, value in avg_loss_dict.items():
            if key in self.train_history:
                self.train_history[key].append(value)
        return avg_loss_dict
    def evaluate(self, views_dict, labels, eval_batch_size=None):
        self.model.eval()
        if eval_batch_size is None:
            eval_batch_size = self.config.get('batch_size', None)
        n_samples = list(views_dict.values())[0].shape[0]
        if eval_batch_size is None or n_samples <= eval_batch_size:
            return self._evaluate_full(views_dict, labels)
        all_scores = []
        with torch.no_grad():
            for i in range(0, n_samples, eval_batch_size):
                end_idx = min(i + eval_batch_size, n_samples)
                batch_views = {
                    view_name: views_dict[view_name][i:end_idx]
                    for view_name in views_dict.keys()
                }
                batch_scores = self._compute_anomaly_scores(batch_views)
                all_scores.append(batch_scores)
        final_score = torch.cat(all_scores, dim=0)
        torch.cuda.empty_cache()
        gc.collect()
        if labels is not None:
            labels_np = labels.cpu().numpy()
            scores_np = final_score.cpu().numpy()
            auc = roc_auc_score(labels_np, scores_np)
            auprc = average_precision_score(labels_np, scores_np)
        else:
            auc = None
            auprc = None
        return auc, auprc, final_score
    def _evaluate_full(self, views_dict, labels):
        self.model.eval()
        with torch.no_grad():
            final_score = self._compute_anomaly_scores(views_dict)
            if labels is not None:
                labels_np = labels.cpu().numpy()
                scores_np = final_score.cpu().numpy()
                auc = roc_auc_score(labels_np, scores_np)
                auprc = average_precision_score(labels_np, scores_np)
            else:
                auc = None
                auprc = None
        return auc, auprc, final_score
    def _compute_anomaly_scores(self, views_dict):
        z_dict, recon_dict = self.model(views_dict)
        if hasattr(self.model, 'view_names'):
            view_names = [v for v in self.model.view_names if v in views_dict]
        else:
            view_names = list(views_dict.keys())
        batch_size = list(views_dict.values())[0].shape[0]
        view_weights_dict = None
        if hasattr(self.model, 'compute_view_weights'):
            out = self.model.compute_view_weights(views_dict)
            if out is not None:
                _, view_weights_dict = out
        recon_errors = []
        for view_name in view_names:
            error = F.mse_loss(
                recon_dict[view_name],
                views_dict[view_name],
                reduction='none'
            ).mean(dim=1)
            recon_errors.append(error)
        recon_errors_mat = torch.stack(recon_errors, dim=1)
        if view_weights_dict is None:
            recon_score = recon_errors_mat.mean(dim=1)
        else:
            weights_mat = torch.stack([view_weights_dict[v] for v in view_names], dim=1)
            recon_score = (weights_mat * recon_errors_mat).sum(dim=1)
        if len(view_names) > 1:
            per_view_consistency = {}
            counts = {}
            for v in view_names:
                per_view_consistency[v] = torch.zeros(batch_size, device=recon_score.device)
                counts[v] = 0
            for i in range(len(view_names)):
                for j in range(i + 1, len(view_names)):
                    vi = view_names[i]
                    vj = view_names[j]
                    z_i = F.normalize(z_dict[vi], dim=1)
                    z_j = F.normalize(z_dict[vj], dim=1)
                    score = contrastive_score(
                        z_i, z_j,
                        temperature=self.config['temperature']
                    )
                    per_view_consistency[vi] += score
                    per_view_consistency[vj] += score
                    counts[vi] += 1
                    counts[vj] += 1
            per_view_scores = []
            for v in view_names:
                denom = counts[v] if counts[v] > 0 else 1
                per_view_scores.append(per_view_consistency[v] / denom)
            per_view_scores_mat = torch.stack(per_view_scores, dim=1)
            if view_weights_dict is None:
                consistency_score = per_view_scores_mat.mean(dim=1)
            else:
                weights_mat = torch.stack([view_weights_dict[v] for v in view_names], dim=1)
                consistency_score = (weights_mat * per_view_scores_mat).sum(dim=1)
        else:
            consistency_score = torch.zeros_like(recon_score)
        w_recon = self.config.get('score_weight_recon', 0.3)
        w_consistency = self.config.get('score_weight_consistency', 0.4)
        final_score = (
            w_recon * recon_score +
            w_consistency * consistency_score
        )
        return final_score
    def train(self, train_views_dict, test_views_dict, test_labels, num_epochs=100, print_every=10):
        print(f"开始训练 {num_epochs} 轮...")
        print(f"视图: {list(train_views_dict.keys())}")
        print(f"训练集样本数量: {list(train_views_dict.values())[0].shape[0]}")
        print(f"测试集样本数量: {list(test_views_dict.values())[0].shape[0]}")
        batch_size = self.config.get('batch_size', None)
        if batch_size is None:
            print(f"训练模式: 全批次训练（使用所有数据）")
        else:
            print(f"训练模式: Mini-batch训练（batch_size={batch_size}）")
        print(f"权重配置: lambda_recon={self.config['lambda_recon']}, "
              f"lambda_contrastive={self.config['lambda_contrastive']}")
        best_auc = 0
        best_scores = None
        best_model_state = None
        start_time = time.time()
        for epoch in range(num_epochs):
            loss_dict = self.train_epoch(train_views_dict)
            if (epoch + 1) % print_every == 0:
                torch.cuda.empty_cache()
                gc.collect()
                auc, auprc, scores = self.evaluate(test_views_dict, test_labels)
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {loss_dict['total']:.4f} | "
                      f"Recon: {loss_dict['recon']:.4f} | "
                      f"Contr: {loss_dict['contrastive']:.4f} | "
                      f"Test AUC: {auc:.4f} | "
                      f"Time: {elapsed:.1f}s")
                if auc > best_auc:
                    best_auc = auc
                    best_scores = scores
                    best_model_state = {
                        'model_state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                        'epoch': epoch + 1
                    }
        print(f"\n训练完成！最佳测试AUC: {best_auc:.4f}")
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"已恢复第 {best_model_state['epoch']} 轮的最佳模型")
        final_auc, final_auprc, final_scores = self.evaluate(test_views_dict, test_labels)
        return best_auc, final_scores
    def get_learned_embeddings(self, views_dict, fusion_method='mean'):
        self.model.eval()
        with torch.no_grad():
            z_dict = self.model.encode(views_dict)
            if fusion_method == 'mean':
                z_fused = torch.stack(list(z_dict.values())).mean(dim=0)
            elif fusion_method == 'concat':
                z_fused = torch.cat(list(z_dict.values()), dim=1)
            elif fusion_method == 'weighted_mean':
                z_list = list(z_dict.values())
                weights = torch.ones(len(z_list)) / len(z_list)
                z_fused = sum(w * z for w, z in zip(weights, z_list))
            else:
                raise ValueError(f"未知的融合方法: {fusion_method}")
            learned_embeddings = z_fused.cpu().numpy()
        return learned_embeddings

import time
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .loss import (
    contrastive_loss,
    contrastive_loss_oa,
    contrastive_score,
    triplet_loss,
    uniform_loss,
    compute_knn_indices,
    update_memory_bank,
)
import gc

class SimpleMultiViewTrainer:
    def __init__(self, model, optimizer, device='cpu', config=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = {
            'use_contrastive': True,
            'use_knn_contrastive': False,
            'use_triplet': False,
            'use_uniform': False,
            'use_memory_bank': False,
            'lambda_recon': 1.0,
            'lambda_contrastive': 1.0,
            'lambda_knn': 0.0,
            'lambda_triplet': 0.0,
            'lambda_uniform': 0.0,
            'k_neighbors': 6,
            'memory_size': 10,
            'memory_ratio': 0.05,
            'temperature': 0.5,
            'update_knn_every': 10,
            'start_uniform_epoch': 0,
            'score_weight_recon': 0.3,
            'score_weight_consistency': 0.4,
            'score_weight_knn': 0.0,
            'batch_size': None,
        }
        if config is not None:
            self.config.update(config)
        self.memory_bank = {}
        self.train_history = {
            'total': [],
            'recon': [],
            'contrastive': [],
            'knn': [],
            'triplet': [],
            'uniform': [],
        }

    def compute_loss(self, views_dict, z_dict, recon_dict, k_indices=None, epoch=0):
        loss_dict = {}
        view_names = list(views_dict.keys())
        batch_size = list(views_dict.values())[0].shape[0]
        loss_recon = 0
        for view_name in view_names:
            loss_recon += F.mse_loss(recon_dict[view_name], views_dict[view_name])
        loss_recon /= len(view_names)
        loss_dict['recon'] = loss_recon.item()
        loss_contrastive = 0
        if self.config.get('use_contrastive', True) and self.config['lambda_contrastive'] > 0:
            for i in range(len(view_names)):
                for j in range(i + 1, len(view_names)):
                    z_i = F.normalize(z_dict[view_names[i]], dim=1)
                    z_j = F.normalize(z_dict[view_names[j]], dim=1)
                    if self.config.get('use_memory_bank', False) and view_names[i] in self.memory_bank:
                        memory_i = torch.cat(self.memory_bank[view_names[i]], dim=0)
                        memory_j = torch.cat(self.memory_bank[view_names[j]], dim=0)
                        memoryh_i = self.model.encoders[view_names[i]](memory_i)
                        memoryh_j = self.model.encoders[view_names[j]](memory_j)
                        memoryh_i = F.normalize(memoryh_i, dim=1)
                        memoryh_j = F.normalize(memoryh_j, dim=1)
                        loss_contrastive += contrastive_loss_oa(
                            z_i,
                            z_j,
                            memoryh_i,
                            memoryh_j,
                            batch_size,
                            temperature=self.config['temperature'],
                        )
                    else:
                        loss_contrastive += contrastive_loss(
                            z_i,
                            z_j,
                            batch_size,
                            temperature=self.config['temperature'],
                        )
            n_pairs = len(view_names) * (len(view_names) - 1) / 2
            loss_contrastive /= n_pairs
        loss_dict['contrastive'] = loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0.0
        loss_knn = 0
        if self.config.get('use_knn_contrastive', False) and self.config.get('lambda_knn', 0.0) > 0 and k_indices is not None:
            k_indices_tensor = torch.tensor(k_indices, dtype=torch.long, device=self.device)
            for view_name in view_names:
                z = F.normalize(z_dict[view_name], dim=1)
                for k in range(k_indices_tensor.shape[1]):
                    neighbor_indices = k_indices_tensor[:, k]
                    z_neighbor = z[neighbor_indices]
                    loss_knn += contrastive_loss(
                        z,
                        z_neighbor,
                        batch_size,
                        temperature=self.config['temperature'],
                    )
            loss_knn /= (len(view_names) * k_indices_tensor.shape[1])
        loss_dict['knn'] = loss_knn.item() if isinstance(loss_knn, torch.Tensor) else 0.0
        loss_triplet = 0
        if self.config.get('use_triplet', False) and self.config.get('lambda_triplet', 0.0) > 0 and k_indices is not None:
            k_indices_tensor = torch.tensor(k_indices, dtype=torch.long, device=self.device)
            for view_name in view_names:
                z = z_dict[view_name]
                positive = z[k_indices_tensor[:, 0]]
                negative = z[k_indices_tensor[:, -1]]
                loss_triplet += triplet_loss(z, positive, negative)
            loss_triplet /= len(view_names)
        loss_dict['triplet'] = loss_triplet.item() if isinstance(loss_triplet, torch.Tensor) else 0.0
        loss_uniform = 0
        if self.config.get('use_uniform', False) and self.config.get('lambda_uniform', 0.0) > 0 and epoch >= self.config.get('start_uniform_epoch', 0):
            for view_name in view_names:
                loss_uniform += uniform_loss(z_dict[view_name])
            loss_uniform /= len(view_names)
        loss_dict['uniform'] = loss_uniform.item() if isinstance(loss_uniform, torch.Tensor) else 0.0
        loss_total = (
            self.config['lambda_recon'] * loss_recon +
            self.config['lambda_contrastive'] * loss_contrastive +
            self.config.get('lambda_knn', 0.0) * loss_knn +
            self.config.get('lambda_triplet', 0.0) * loss_triplet +
            self.config.get('lambda_uniform', 0.0) * loss_uniform
        )
        loss_dict['total'] = loss_total.item()
        return loss_total, loss_dict

    def train_epoch(self, views_dict, k_indices=None, epoch=0):
        self.model.train()
        batch_size = self.config.get('batch_size', None)
        if batch_size is None:
            z_dict, recon_dict = self.model(views_dict)
            loss_total, loss_dict = self.compute_loss(views_dict, z_dict, recon_dict, k_indices=k_indices, epoch=epoch)
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            if self.config.get('use_memory_bank', False):
                with torch.no_grad():
                    update_memory_bank(
                        z_dict,
                        self.memory_bank,
                        views_dict,
                        top_ratio=self.config.get('memory_ratio', 0.05),
                        max_size=self.config.get('memory_size', 10),
                    )
            for key, value in loss_dict.items():
                if key in self.train_history:
                    self.train_history[key].append(value)
            return loss_dict
        n_samples = list(views_dict.values())[0].shape[0]
        indices = torch.randperm(n_samples)
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'contrastive': 0.0,
            'knn': 0.0,
            'triplet': 0.0,
            'uniform': 0.0,
        }
        n_batches = 0
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_views = {
                view_name: views_dict[view_name][batch_indices]
                for view_name in views_dict.keys()
            }
            z_dict, recon_dict = self.model(batch_views)
            batch_k_indices = None
            if (self.config.get('use_knn_contrastive', False) or self.config.get('use_triplet', False)) and (
                self.config.get('lambda_knn', 0.0) > 0 or self.config.get('lambda_triplet', 0.0) > 0
            ):
                with torch.no_grad():
                    z_fused = torch.stack(list(z_dict.values())).mean(dim=0)
                    batch_k_indices = compute_knn_indices(
                        z_fused,
                        k=self.config.get('k_neighbors', 6),
                    )
            loss_total, loss_dict = self.compute_loss(
                batch_views,
                z_dict,
                recon_dict,
                k_indices=batch_k_indices,
                epoch=epoch,
            )
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            if self.config.get('use_memory_bank', False):
                with torch.no_grad():
                    update_memory_bank(
                        z_dict,
                        self.memory_bank,
                        batch_views,
                        top_ratio=self.config.get('memory_ratio', 0.05),
                        max_size=self.config.get('memory_size', 10),
                    )
            for key in epoch_losses.keys():
                epoch_losses[key] += loss_dict.get(key, 0.0)
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
            auroc = roc_auc_score(labels_np, scores_np)
            precision, recall, _ = precision_recall_curve(labels_np, scores_np)
            auprc = auc(recall, precision)
        else:
            auroc = None
            auprc = None
        return auroc, auprc, final_score

    def _evaluate_full(self, views_dict, labels):
        self.model.eval()
        with torch.no_grad():
            final_score = self._compute_anomaly_scores(views_dict)
            if labels is not None:
                labels_np = labels.cpu().numpy()
                scores_np = final_score.cpu().numpy()
                auroc = roc_auc_score(labels_np, scores_np)
                precision, recall, _ = precision_recall_curve(labels_np, scores_np)
                auprc = auc(recall, precision)
            else:
                auroc = None
                auprc = None
        return auroc, auprc, final_score

    def _compute_anomaly_scores(self, views_dict):
        z_dict, recon_dict = self.model(views_dict)
        recon_errors = []
        for view_name in views_dict.keys():
            error = F.mse_loss(
                recon_dict[view_name],
                views_dict[view_name],
                reduction='none'
            ).mean(dim=1)
            recon_errors.append(error)
        recon_score = torch.stack(recon_errors).mean(dim=0)
        consistency_scores = []
        view_names = list(z_dict.keys())
        for i in range(len(view_names)):
            for j in range(i + 1, len(view_names)):
                z_i = F.normalize(z_dict[view_names[i]], dim=1)
                z_j = F.normalize(z_dict[view_names[j]], dim=1)
                score = contrastive_score(
                    z_i, z_j,
                    temperature=self.config['temperature']
                )
                consistency_scores.append(score)
        if consistency_scores:
            consistency_score = torch.stack(consistency_scores).mean(dim=0)
        else:
            consistency_score = torch.zeros_like(recon_score)
        w_recon = self.config.get('score_weight_recon', 0.3)
        w_consistency = self.config.get('score_weight_consistency', 0.4)
        w_knn = self.config.get('score_weight_knn', 0.0)
        knn_score = torch.zeros_like(recon_score)
        if w_knn > 0:
            z_fused = torch.stack(list(z_dict.values())).mean(dim=0)
            dist_matrix = torch.cdist(z_fused, z_fused, p=2)
            dist_matrix.fill_diagonal_(float('inf'))
            k = min(5, max(1, z_fused.shape[0] - 1))
            knn_dists, _ = torch.topk(dist_matrix, k, largest=False, dim=1)
            knn_score = knn_dists.mean(dim=1)
        final_score = (
            w_recon * recon_score +
            w_consistency * consistency_score +
            w_knn * knn_score
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
            print(f"训练模式: Mini-batch训练（batch_size={batch_size})")
        print(f"权重配置: lambda_recon={self.config['lambda_recon']}, "
              f"lambda_contrastive={self.config['lambda_contrastive']}, "
              f"lambda_knn={self.config.get('lambda_knn', 0.0)}, "
              f"lambda_triplet={self.config.get('lambda_triplet', 0.0)}, "
              f"lambda_uniform={self.config.get('lambda_uniform', 0.0)}")
        best_auc = 0
        best_auprc = 0
        best_scores = None
        best_model_state = None
        start_time = time.time()
        k_indices = None
        for epoch in range(num_epochs):
            if (
                (self.config.get('use_knn_contrastive', False) and self.config.get('lambda_knn', 0.0) > 0)
                or (self.config.get('use_triplet', False) and self.config.get('lambda_triplet', 0.0) > 0)
            ) and self.config.get('batch_size', None) is None:
                if epoch % self.config.get('update_knn_every', 10) == 0:
                    with torch.no_grad():
                        z_dict = self.model.encode(train_views_dict)
                        z_fused = torch.stack(list(z_dict.values())).mean(dim=0)
                        k_indices = compute_knn_indices(
                            z_fused,
                            k=self.config.get('k_neighbors', 6),
                        )
            loss_dict = self.train_epoch(train_views_dict, k_indices=k_indices, epoch=epoch)
            if (epoch + 1) % print_every == 0:
                torch.cuda.empty_cache()
                gc.collect()
                auc, auprc, scores = self.evaluate(test_views_dict, test_labels)
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {loss_dict['total']:.4f} | "
                      f"Recon: {loss_dict['recon']:.4f} | "
                      f"Contr: {loss_dict['contrastive']:.4f} | "
                      f"KNN: {loss_dict.get('knn', 0.0):.4f} | "
                      f"Trip: {loss_dict.get('triplet', 0.0):.4f} | "
                      f"Unif: {loss_dict.get('uniform', 0.0):.4f} | "
                      f"Test AUC: {auc:.4f} AUPRC: {auprc:.4f} | "
                      f"Time: {elapsed:.1f}s")
                if auc > best_auc:
                    best_auc = auc
                    best_auprc = auprc
                    best_scores = scores
                    best_model_state = {
                        'model_state_dict': {k: v.clone() for k, v in self.model.state_dict().items()},
                        'epoch': epoch + 1
                    }
        print(f"\n训练完成！最佳测试AUC: {best_auc:.4f} AUPRC: {best_auprc:.4f}")
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state['model_state_dict'])
            print(f"已恢复第 {best_model_state['epoch']} 轮的最佳模型")
        final_auc, final_auprc, final_scores = self.evaluate(test_views_dict, test_labels)
        return best_auc, best_auprc, final_scores

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

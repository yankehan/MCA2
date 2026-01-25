import torch
import torch.optim as optim
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
class NCMODTrainer:
    def __init__(self, model, lr=0.001, n_epochs=16, batch_size=20,
                 weight_decay=1e-6, device='cpu', module_weight=1.0):
        self.model = model
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.module_weight = module_weight
        self.optimizers = {}
        for view_name in model.view_names:
            self.optimizers[view_name] = optim.Adam(
                model.get_view_net(view_name).parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
    def train_single_view(self, id_round, id_view, dataset, view_name, pre_train=False):
        view_net = self.model.get_view_net(view_name)
        view_net.train()
        optimizer = self.optimizers[view_name]
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        num_epochs = self.n_epochs
        for epoch in range(num_epochs):
            for data in train_loader:
                views_dict, neighbor_global, neighbor_local, weights_global, labels, index = data
                inputs = views_dict[view_name].to(self.device)
                neighbor_g = neighbor_global[view_name].to(self.device)
                neighbor_l = neighbor_local[view_name].to(self.device)
                weights_g = weights_global.to(self.device)
                optimizer.zero_grad()
                encoded, outputs = view_net(inputs)
                encoded_global_neighbor, _ = view_net(neighbor_g)
                encoded_local_neighbor, _ = view_net(neighbor_l)
                recon_error = torch.sum((outputs - inputs) ** 2, dim=1)
                dis_global = torch.sum((encoded - encoded_global_neighbor) ** 2, dim=1)
                if pre_train and id_round == 0:
                    scores = recon_error
                else:
                    scores = recon_error + self.module_weight * weights_g * dis_global
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()
        view_encoded, recon_error = self.test_single_view(dataset, view_name)
        return view_net, view_encoded, recon_error
    def test_single_view(self, dataset, view_name):
        view_net = self.model.get_view_net(view_name)
        view_net.eval()
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        encoded_data = None
        all_scores = []
        with torch.no_grad():
            for data in test_loader:
                views_dict, _, _, _, labels, idx = data
                inputs = views_dict[view_name].to(self.device)
                encoded, outputs = view_net(inputs)
                recon_error = torch.sum((outputs - inputs) ** 2, dim=1)
                if encoded_data is None:
                    encoded_data = encoded.cpu().data.numpy()
                else:
                    encoded_data = np.concatenate((encoded_data, encoded.cpu().numpy()), axis=0)
                all_scores.extend(recon_error.cpu().data.numpy().tolist())
        scores = np.array(all_scores)
        return encoded_data, scores
def get_new_knn(view_encoded, num_neibs):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]
    neibs_local = [[] for _ in range(num_views)]
    kth_neib_local = [[] for _ in range(num_views)]
    for i in range(num_views):
        nbrs = NearestNeighbors(n_neighbors=num_neibs + 1, algorithm='ball_tree').fit(view_encoded[i])
        distances, indices = nbrs.kneighbors(view_encoded[i])
        for j in range(num_obj):
            indice_j = list(indices[j])
            if j in indice_j:
                indice_j.remove(j)
            else:
                indice_j = indice_j[1:]
            neibs_local[i].append(indice_j)
            kth_neib_local[i].append(indices[j, num_neibs])
    weights_global = [[] for _ in range(num_views)]
    neibs_global = [[] for _ in range(num_views)]
    for i in range(num_views):
        for j in range(num_obj):
            tmp = []
            tmp_weights = []
            for k in range(num_views):
                if k != i:
                    tmp = tmp + neibs_local[k][j]
            for k in range(len(tmp)):
                dis_jk = cal_dis(view_encoded[i], j, tmp[k])
                if dis_jk > 1e-8:
                    w = cal_dis(view_encoded[i], j, kth_neib_local[i][j]) / dis_jk
                else:
                    w = 2.0
                tmp_weights.append(w)
            neibs_global[i].append(tmp)
            weights_global[i].append(tmp_weights)
    return neibs_global, neibs_local, weights_global
def cal_dis(X, a, b):
    return np.sum((X[a] - X[b]) ** 2)
def cal_scores(view_encoded, num_neibs, recon_error):
    num_views = len(view_encoded)
    num_obj = view_encoded[0].shape[0]
    recon_scores = np.zeros(num_obj)
    for i in range(num_views):
        s1 = (recon_error[i] - np.min(recon_error[i])) / (np.max(recon_error[i]) - np.min(recon_error[i]) + 1e-8)
        recon_scores += s1
    dim_concen = sum(view_encoded[i].shape[1] for i in range(num_views))
    encoded_concen = np.zeros((num_obj, dim_concen))
    tmp_dim = 0
    for i in range(num_views):
        encoded_concen[:, tmp_dim: tmp_dim + view_encoded[i].shape[1]] = view_encoded[i]
        tmp_dim += view_encoded[i].shape[1]
    nbrs = NearestNeighbors(n_neighbors=num_neibs+1, algorithm='ball_tree').fit(encoded_concen)
    distances, indices = nbrs.kneighbors(encoded_concen)
    knn_scores = np.sum(distances, axis=1)
    return recon_scores, knn_scores
def cal_final_scores(recon_scores, knn_scores):
    s1 = (recon_scores - np.min(recon_scores)) / (np.max(recon_scores) - np.min(recon_scores) + 1e-8)
    s2 = (knn_scores - np.min(knn_scores)) / (np.max(knn_scores) - np.min(knn_scores) + 1e-8)
    total_scores = 0.5 * s1 + s2
    return total_scores

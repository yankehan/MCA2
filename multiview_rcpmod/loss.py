import torch
import torch.nn as nn
import torch.nn.functional as F
def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N // 2):
        mask[i, N // 2 + i] = 0
        mask[N // 2 + i, i] = 0
    mask = mask.bool()
    return mask
def contrastive_loss(h_i, h_j, batch_size=None, contr_weights=None, temperature=0.5):
    if batch_size is None:
        batch_size = h_i.shape[0]
    if contr_weights is not None:
        h_i = h_i * contr_weights.view(-1, 1)
        h_j = h_j * contr_weights.view(-1, 1)
    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)
    sim = torch.matmul(h, h.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N).to(h.device)
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    return loss
def contrastive_loss_oa(h_i, h_j, memoryh_i, memoryh_j, batch_size,
                        contr_weights=None, temperature=0.5):
    if contr_weights is not None:
        h_i = h_i * contr_weights.view(-1, 1)
        h_j = h_j * contr_weights.view(-1, 1)
    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)
    memoryh = torch.cat((memoryh_i, memoryh_j), dim=0)
    negsims = torch.matmul(h, memoryh.T) / temperature
    sim = torch.matmul(h, h.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N).to(h.device)
    negative_samples = sim[mask].reshape(N, -1)
    negative_samples = torch.cat((negative_samples, negsims), dim=1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss = criterion(logits, labels)
    loss /= N
    return loss
def contrastive_score(h_i, h_j, batch_size=None, temperature=0.5):
    if batch_size is None:
        batch_size = h_i.shape[0]
    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)
    sim = torch.matmul(h, h.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N).to(h.device)
    negative_samples = sim[mask].reshape(N, -1)
    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    criterion = nn.CrossEntropyLoss(reduction="none")
    scores = criterion(logits, labels)
    scores = (scores[:batch_size] + scores[batch_size:]) / 2
    return scores
def triplet_loss(anchor, positive, negative, margin=1.0):
    pdist = nn.PairwiseDistance(2)
    per_point_loss = pdist(anchor, positive) - pdist(anchor, negative) + margin
    per_point_loss = F.relu(per_point_loss)
    loss = per_point_loss.mean()
    return loss
def pairwise_NNs_inner(x):
    dots = torch.mm(x, x.t())
    n = x.shape[0]
    dots.view(-1)[::(n + 1)].fill_(-1)
    _, I = torch.max(dots, 1)
    return I
def uniform_loss(x):
    pdist = nn.PairwiseDistance(2)
    I = pairwise_NNs_inner(x.data)
    distances = pdist(x, x[I])
    loss = -torch.log(x.shape[0] * distances + 1e-8).mean()
    return loss
def compute_knn_indices(embeddings, k=6, metric='cosine'):
    from sklearn.neighbors import NearestNeighbors
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    neigh = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    neigh.fit(embeddings)
    _, indices = neigh.kneighbors(embeddings)
    indices = indices[:, 1:]
    return indices
def update_memory_bank(z_dict, memory_bank, views_dict, top_ratio=0.05, max_size=10):
    view_names = list(z_dict.keys())
    similarities = []
    for i in range(len(view_names)):
        for j in range(i + 1, len(view_names)):
            z_i = F.normalize(z_dict[view_names[i]], dim=1)
            z_j = F.normalize(z_dict[view_names[j]], dim=1)
            sim = (z_i * z_j).sum(dim=1)
            similarities.append(sim)
    if not similarities:
        return
    avg_similarity = torch.stack(similarities).mean(dim=0)
    topnum = max(int(avg_similarity.numel() * top_ratio), 1)
    _, hard_indices = torch.topk(-avg_similarity, topnum)
    for view_name in view_names:
        if view_name not in memory_bank:
            memory_bank[view_name] = []
        hard_samples = views_dict[view_name][hard_indices]
        memory_bank[view_name].append(hard_samples)
        if len(memory_bank[view_name]) > max_size:
            memory_bank[view_name].pop(0)

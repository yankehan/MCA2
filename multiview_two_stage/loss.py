import torch
import torch.nn as nn
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
def mask_correlated_samples(N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N // 2):
        mask[i, N // 2 + i] = 0
        mask[N // 2 + i, i] = 0
    mask = mask.bool()
    return mask
def contrastive_loss_per_sample(h_i, h_j, batch_size=None, temperature=0.5):
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
    loss_vec = criterion(logits, labels)
    loss_per_sample = (loss_vec[:batch_size] + loss_vec[batch_size:]) / 2
    return loss_per_sample
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

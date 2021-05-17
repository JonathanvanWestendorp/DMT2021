import torch
import numpy as np


def pairwise_loss(scores, labels):
    sig = 1

    # if there's only one rating
    if labels.size(0) < 2:
        return None

    S = labels[:, None] - labels
    scores = scores - scores.squeeze()
    S[S > 0] = 1
    S[S < 0] = -1
    C = (1 / 2) * (1 - S) * sig * scores + torch.log(1 + torch.exp(-scores))
    return torch.mean(C)


def compute_lambda_i(scores, labels):
    S = labels[:, None] - labels
    scores = scores - scores.squeeze()
    S[S > 0] = 1
    S[S < 0] = -1
    lambda_total = ((1 - S) / 2) - (1 / (1 + torch.exp(scores)))
    return torch.sum(lambda_total, 1, keepdim=True)


def listwise_loss(scores, labels, device):
    lamda = compute_lambda_i(scores, labels)

    ls = torch.from_numpy(np.arange(1, len(scores) + 1)).type(torch.FloatTensor)[:, None]
    ls = ls.to(device)
    DCG = torch.sum(scores / torch.log2(ls + 1), dim=1)
    scor = scores
    not_rel = torch.where(labels == 0)[0]
    rel_docs = None
    for i in not_rel:
        rel_docs = torch.cat([scor[0:i], scor[i + 1:]])

    if rel_docs is None:
        IDCG = 1
    else:
        ls = torch.from_numpy(np.arange(1, len(rel_docs) + 1)).type(torch.FloatTensor)
        ls = ls.to(device)
        IDCG = torch.ones(DCG.size()).to(device) * torch.sum((2 ** rel_docs) - 1 / torch.log2(ls + 1))
    NDCG = DCG / IDCG

    delta_IRM = NDCG - NDCG[:, None]
    grad = lamda * torch.sum(torch.abs(delta_IRM), dim=1, keepdim=True)
    return grad
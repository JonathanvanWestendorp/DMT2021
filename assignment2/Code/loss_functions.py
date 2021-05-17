import torch


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

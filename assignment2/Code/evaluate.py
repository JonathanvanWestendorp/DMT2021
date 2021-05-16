import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score

from dataset import PropRanking
from model import NeuralModule


def evaluate_model(model, valid_data_loader, k, device):
    scores = []
    for inputs, targets in valid_data_loader:
        out = model(inputs.to(device))
        scores.append(ndcg_score(targets[0].T.numpy(), out[0].T.cpu().numpy(), k=k))

    return np.mean(scores)


if __name__ == '__main__':
    dataset = PropRanking(0.8, "../Data/training_set_VU_DM.pkl")
    valid_data = dataset.get_validation()

    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

    model = NeuralModule(19, 1)
    model.load_state_dict(torch.load("../Models/pointwise_epochs3_seed0_nan_cols_removed"))
    model.eval()

    with torch.no_grad():
        evaluate_model(model, valid_data_loader)

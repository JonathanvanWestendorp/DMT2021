import torch
from torch.utils.data import DataLoader

from dataset import PropRanking
from model import NeuralModule


def evaluate_model(model, valid_data):
    return None


if __name__ == '__main__':
    dataset = PropRanking(0.8, "../Data/training_set_VU_DM.pkl")
    train_data = dataset.get_train()
    valid_data = dataset.get_validation()

    train_data_loader = DataLoader(train_data, batch_size=512, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=512, shuffle=True)

    model = NeuralModule(19, 1)
    model.load_state_dict(torch.load("../Models"))
    model.eval()

    with torch.no_grad:
        evaluate_model(model, train_data_loader)

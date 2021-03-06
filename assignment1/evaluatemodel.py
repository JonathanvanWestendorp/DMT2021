import torch
from torch.utils.data import DataLoader
import dataloader
import pandas as pd

from lstm import MoodPredictionModel

import argparse

# Variables to set to 0 to evaluate it's influence on the prediction
INFLUENCE = ['appCat.communication', 'appCat.entertainment']
def accuracy(config):
    data = pd.read_csv(config.data_path)
    inputs, targets = dataloader.process(data, window_size=5)

    device = torch.device(config.device)

    n = len(targets)
    train_inputs, train_targets = inputs[:round(.8 * n)].to(device), targets[:round(.8 * n)].to(device)
    test_inputs, test_targets = inputs[round(.8 * n):].to(device), targets[round(.8 * n):].to(device)

    print("Initializing LSTM model...")
    model = MoodPredictionModel(config.input_length,
                                config.input_dim,
                                config.num_hidden,
                                config.num_layers).to(device)

    print("Loading model parameters from trained model")
    model.load_state_dict(torch.load(config.saved_model))
    model.eval()
    print(model)

    train_out = model(train_inputs).squeeze()
    test_out = model(test_inputs).squeeze()

    correct_train = (abs(train_out - train_targets) < config.corr_thres).sum().item()
    train_accuracy = correct_train / train_out.size(0)

    correct_test = (abs(test_out - test_targets) < config.corr_thres).sum().item()
    test_accuracy = correct_test / test_out.size(0)

    print(f"Accuracy on training set: {train_accuracy*100}%")
    print(f"Accuracy on testing set: {test_accuracy*100}%")

def evaluate(config):
    data_loader = DataLoader(dataloader.MoodDataSet(config.data_path, INFLUENCE),
                             batch_size=config.batch_size, shuffle=True)

    device = torch.device(config.device)
    print(device)

    print("Initializing LSTM model...")
    model = MoodPredictionModel(config.input_length,
                                config.input_dim,
                                config.num_hidden,
                                config.num_layers).to(device)

    print("Loading model parameters from trained model")
    model.load_state_dict(torch.load(config.saved_model))
    model.eval()

    for _, (batch_inputs, _) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)

        # Print batch as dataframe if needed
        print(pd.DataFrame(batch_inputs.cpu().numpy()[0]))

        out = model(batch_inputs)
        print(out)
        exit(1)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=7,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers in the model')
    parser.add_argument('--saved_model', type=str, default='./trained_model_seed_546_80perc',
                        help='Path of model to load')
    # Dataloader params
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Number of examples to process in a batch')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--data_path', type=str, default="./dataset_mood_smartphone.csv",
                        help='Data path')
    parser.add_argument('--corr_thres', type=float, default=.75,
                        help='Threshold for correctness of output')
    config = parser.parse_args()

    #evaluate(config)
    accuracy(config)
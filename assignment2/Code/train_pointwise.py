from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PropRanking
from model import NeuralModule
from evaluate import evaluate_model

import numpy as np
import matplotlib.pyplot as plt


def train(config):
    print(f"Random Seed: {config.seed}")
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("Loading data...")
    dataset = PropRanking(config.split_ratio, config.train_path)
    train_data = dataset.get_train()
    valid_data = dataset.get_validation()

    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=1, shuffle=True)

    device = torch.device(config.device)

    print(f"Initializing neural model on {device}...")
    model = NeuralModule(train_data.num_features, 1).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Keep track of accuracy and loss
    loss_list = []
    for e in range(config.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(train_data_loader):
            # Only for time measurement of step through network
            t1 = time.time()

            # Move to GPU
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.unsqueeze(1).to(device)

            # Reset for next iteration
            model.zero_grad()

            # Forward pass
            out = model(batch_inputs)
            # print(out[0].item(), batch_targets[0].item())
            # Compute the loss, gradients and update network parameters
            loss = loss_function(out, batch_targets)
            loss.backward()

            optimizer.step()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1) if t2 != t1 else np.inf

            if step % 100 == 0:
                loss_list.append(float(loss))
                print("[{}] Epoch {}, Train Step {:04d}, Batch Size = {}, Examples/Sec = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), e + 1, step,
                        config.batch_size, examples_per_second, loss
                        ))
                if step % 3500 == 0:
                    print("Evaluating model...")
                    with torch.no_grad():
                        print(f"nDCG@5: {evaluate_model(model, valid_data_loader, config.k, device)}\n")
    with torch.no_grad():
        print(f"nDCG@5: {evaluate_model(model, valid_data_loader, config.k, device)}\n")

    print('Done training.')

    # Save trained model
    torch.save(model.state_dict(), f"../Models/pointwise_epochs{config.epochs}_seed{config.seed}_nan_cols_removed")

    plt.plot(loss_list)
    plt.show()


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--split_ratio', type=float, default=.8,
                        help='Ratio between training set and validation set')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--summary_path', type=str, default="../summaries/",
                        help='Output path for summaries')
    parser.add_argument('--train_path', type=str, default="../Data/training_set_VU_DM.pkl",
                        help='Train data path')
    parser.add_argument('--k', type=int, default=5,
                        help='Truncate nDCG calculation')

    # Change to random for random seed
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for reproducibility')

    args = parser.parse_args()

    # Train the model
    train(args)

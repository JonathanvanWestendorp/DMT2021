from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import dataloader

from lstm import MoodPredictionModel

import numpy as np
import matplotlib.pyplot as plt


def train(config):
    accuracy_lists, loss_lists = [], []
    seeds = [0, 123, 546]
    print(f"Random Seeds: {seeds}")
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_loader = DataLoader(dataloader.MoodDataSet(config.data_path, []),
                                 batch_size=config.batch_size, shuffle=True)

        # Initialize the device which to run the model on
        device = torch.device(config.device)
        print(device)

        print("Initializing LSTM model...")
        model = MoodPredictionModel(config.input_length,
                                    config.input_dim,
                                    config.num_hidden,
                                    config.num_layers).to(device)

        # Setup the loss and optimizer
        loss_function = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Keep track of accuracy and loss
        accuracy_list, loss_list = [], []
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Only for time measurement of step through network
            t1 = time.time()

            # Move to GPU
            batch_inputs = batch_inputs.to(device)                         # [batch_size, seq_length, n_features]
            batch_targets = torch.unsqueeze(batch_targets, 1).to(device)   # [batch_size]

            # Reset for next iteration
            model.zero_grad()

            # Forward pass
            out = model(batch_inputs)
            # Compute the loss, gradients and update network parameters
            loss = loss_function(out, batch_targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config.max_norm)

            optimizer.step()

            correct = (abs(out - batch_targets) < config.corr_thres).sum().item()
            accuracy = correct / out.size(0)

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1) if t2 != t1 else np.inf

            accuracy_list.append(accuracy)
            loss_list.append(float(loss))

            if step % 50 == 0:

                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                       Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                        config.train_steps, config.batch_size, examples_per_second,
                        accuracy, loss
                        ))

            # Check if training is finished
            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report
                # https://github.com/pytorch/pytorch/pull/9655
                break

        print('Done training.')
        accuracy_lists.append(accuracy_list)
        loss_lists.append(loss_list)

        # Save trained model
        torch.save(model.state_dict(), f"./trained_model_seed_{seed}")

    accuracy_lists = np.array(accuracy_lists)
    loss_lists = np.array(loss_lists)

    acc_mean, acc_std = np.mean(accuracy_lists, axis=0), np.std(accuracy_lists, axis=0)
    loss_mean, loss_std = np.mean(loss_lists, axis=0), np.std(loss_lists, axis=0)

    # Plotting
    x = np.arange(len(accuracy_lists[0]))

    fig, axs = plt.subplots(1, 2)

    #fig.suptitle(f"Accuracy and Loss for with T = {config.input_length}", fontsize=16)

    axs[0].fill_between(x, acc_mean - acc_std, acc_mean + acc_std, alpha=.4, label="Acc std.")
    axs[0].plot(x, acc_mean, label="Acc mean")
    axs[0].set_xlabel("Steps")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].fill_between(x, loss_mean - loss_std, loss_mean + loss_std, alpha=.4, label="Loss std.")
    axs[1].plot(x, loss_mean, label="Loss mean")
    axs[1].set_xlabel("Steps")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss")
    axs[1].legend()

    plt.savefig("accandloss", dpi=500)
    plt.show()


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

    # Training params
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=2000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--corr_thres', type=float, default=.75,
                        help='Threshold for correctness of output')

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--data_path', type=str, default="./dataset_mood_smartphone.csv",
                        help='Data path')

    config = parser.parse_args()

    # Train the model
    train(config)

import pandas as pd
import torch

import dataloader


if __name__ == "__main__":
    data = pd.read_csv("./dataset_mood_smartphone.csv")
    _, targets = dataloader.process(data)

    targets = torch.round(targets)

    n = len(targets)
    train_targets, test_targets = targets[:round(.8*n)], targets[round(.8*n):]

    train_diff = train_targets[:-1] == train_targets[1:]
    test_diff = test_targets[:-1] == test_targets[1:]

    train_accuracy = sum(train_diff) / len(train_diff)
    test_accuracy = sum(test_diff) / len(test_diff)

    print(f"Accuracy on training set: {train_accuracy}")
    print(f"Accuracy on testing set: {test_accuracy}")

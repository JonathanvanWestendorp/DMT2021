import torch
import pandas as pd
from sklearn import svm

import dataloader


if __name__ == "__main__":
    data = pd.read_csv("./dataset_mood_smartphone.csv")
    inputs, targets = dataloader.process(data, window_size=2)

    inputs = inputs.mean(dim=1)
    targets = torch.round(targets)

    n = len(targets)
    train_inputs, train_targets = inputs[:round(.8*n)], targets[:round(.8*n)]
    test_inputs, test_targets = inputs[round(.8*n):], targets[round(.8*n):]

    clf = svm.SVC()
    clf.fit(train_inputs, train_targets)

    train_out = clf.predict(train_inputs)
    test_out = clf.predict(test_inputs)

    train_diff = torch.tensor(train_out) == train_targets
    test_diff = torch.tensor(test_out) == test_targets

    train_accuracy = sum(train_diff) / len(train_diff)
    test_accuracy = sum(test_diff) / len(test_diff)

    print(f"Accuracy on training set: {train_accuracy}")
    print(f"Accuracy on testing set: {test_accuracy}")

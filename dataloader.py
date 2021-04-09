import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Outermost dates in dataset
BASE_TIME = pd.Timestamp("2014-02-15")
END_TIME = pd.Timestamp("2014-06-09")


def window_split(x, window_size=5):
    length = x.size(0)
    splits = []

    for slice_start in range(0, length - window_size + 1):
        slice_end = slice_start + window_size
        splits.append(x[slice_start:slice_end])

    return torch.stack(splits[:-1])


def process(data):
    inputs, targets = [], []
    data.time = pd.to_datetime(data.time)
    for patient_id in data.id.unique():
        patient_inputs = []
        patient_data = data[data.id == patient_id]
        for feature in data.variable.unique():
            outer_rows = pd.DataFrame([[patient_id, BASE_TIME, feature, np.nan],
                                       [patient_id, END_TIME, feature, np.nan]],
                                      columns=["id", "time", "variable", "value"])

            patient_feature_data = patient_data[patient_data.variable == feature].append(outer_rows)

            if feature == "mood":
                # Discard first 5 moods because there's not enough prior feature information for these
                patient_mood_grouped = patient_feature_data.resample('D', on='time')['value'].mean()[5:]
                targets.append(torch.tensor(patient_mood_grouped.tolist()))
            else:
                patient_feature_grouped = patient_feature_data.resample('D', on='time')['value'].mean()
                patient_inputs.append(patient_feature_grouped.tolist())

        inputs.append(window_split(torch.tensor(patient_inputs).T))

    inputs = torch.cat(inputs)
    targets = torch.cat(targets)

    # TODO remove NaN values!

    return inputs, targets


class MoodDataSet(Dataset):
    def __init__(self, csv_file, input_length):
        self.input_length = input_length
        self.raw = pd.read_csv(csv_file)
        self.data = process(self.raw)

    def __len__(self):
        return self.raw.id.nunique()

    def __getitem__(self, idx):
        window_start = np.random.randint(low=0, high=self.data[0].shape[-1] - self.input_length)
        window_end = window_start + self.input_length
        return self.data[0][idx, :, window_start:window_end], self.data[1][idx, window_start:window_end]


if __name__ == "__main__":
    data = pd.read_csv("dataset_mood_smartphone.csv")
    print(process(data))

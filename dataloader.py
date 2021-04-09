import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Outermost dates in dataset
BASE_TIME = pd.Timestamp("2014-02-15")
END_TIME = pd.Timestamp("2014-06-09")


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
                patient_mood_grouped = patient_feature_data.resample('D', on='time')['value'].mean()
                targets.append(patient_mood_grouped.iloc[1::5].tolist())
            else:
                patient_feature_grouped = patient_feature_data.resample('5D', on='time')['value'].mean()
                patient_inputs.append(patient_feature_grouped.tolist())

        inputs.append(patient_inputs)
    print(torch.tensor(targets))
    print(torch.tensor(inputs).shape, torch.tensor(targets).shape)


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
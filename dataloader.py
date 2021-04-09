import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Outermost dates in dataset
BASE_TIME = pd.Timestamp("2014-02-15")
END_TIME = pd.Timestamp("2014-06-09")

# Features to skip. Decided based on significance and lack of data
SKIP = ["call", "sms", "appCat.weather", "appCat.utilities", "appCat.unknown", "appCat.travel", "appCat.other",
        "appCat.builtin", "appCat.finance", "appCat.game", "appCat.office"]


# TODO Mood interpolation? Missing finance, game, office attributes?

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
        for i, feature in enumerate(data[~data.variable.isin(SKIP)].variable.unique()):
            outer_rows = pd.DataFrame([[patient_id, BASE_TIME, feature, np.nan],
                                       [patient_id, END_TIME, feature, np.nan]],
                                      columns=["id", "time", "variable", "value"])

            patient_feature_data = patient_data[patient_data.variable == feature].append(outer_rows)

            if feature == "mood":
                # Discard first 5 moods because there's not enough prior feature information for these
                patient_mood_grouped = patient_feature_data.resample('D', on='time').value.mean()[5:]
                targets.append(torch.tensor(patient_mood_grouped.tolist()))
            else:
                # Linear interpolation for missing data
                patient_feature_grouped = patient_feature_data.resample('D', on='time').value.mean().interpolate(method='linear', limit_direction='both')
                patient_inputs.append(patient_feature_grouped.tolist())

        inputs.append(window_split(torch.tensor(patient_inputs).T))

    inputs = torch.cat(inputs)
    targets = torch.cat(targets)

    # Remove missing days in target moods
    missing_idx = targets.isnan()
    inputs = inputs[~missing_idx, :, :]
    targets = targets[~missing_idx]

    return inputs, targets


class MoodDataSet(Dataset):
    def __init__(self, csv_file):
        self.data = process(pd.read_csv(csv_file))

    def __len__(self):
        return len(self.data[1])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

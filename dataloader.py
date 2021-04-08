import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def process(data):
    # new_data = pd.DataFrame(columns=['id', 'time_start', 'time_end' 'circumplex.arousal', 'circumplex.valence',
    #                                  'activity', 'screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication'
    #                                  'appCat.entertainment', 'appCat.finance', 'appCat.game', 'appCat.office'
    #                                  'appCat.other', 'appCat.social', 'appCat.travel', 'appCat.unknown'
    #                                  'appCat.utilities', 'appCat.weather', 'mood'])
    data['time'] = pd.to_datetime(data.time)
    print(data['id'].nunique())
    ids = data['id'].unique()
    features = data[data['variable'] != "mood"]['variable'].unique()
    for patient_id in ids:
        patient_data = data[data['id'] == patient_id]
        for feature in features:
            patient_feature_data = patient_data[patient_data['variable'] == feature]
            ## HIER MOETEN WE WAT MEE!
            print(patient_feature_data.resample('D', on='time', origin=pd.Timestamp('2014-01-01'))['value'].mean())

        break


    targets, inputs = [], []

    # Get targets
    moods = data[data['variable'] == "mood"]
    targets = [moods[moods['id'] == patient_id]['value'].tolist() for patient_id in moods['id'].unique()]

    min_target_length = min([len(ls) for ls in targets])

    for i in range(len(targets)):
        targets[i] = targets[i][:min_target_length]

    # Get inputs
    features = data[data['variable'] != "mood"]['variable'].unique()
    for feature in features:
        intermediate_inputs = []
        data_per_feature = data[data['variable'] == feature]
        for patient_id in data['id'].unique():
            values = data_per_feature[data_per_feature['id'] == patient_id]['value'].tolist()
            intermediate_inputs.append(values)

        min_length = min([len(ls) for ls in intermediate_inputs])
        if min_length < min_target_length:
            continue

        for i in range(len(intermediate_inputs)):
            intermediate_inputs[i] = intermediate_inputs[i][:min_target_length]

        inputs.append(intermediate_inputs)

    return torch.tensor(inputs).permute(1, 0, 2), torch.tensor(targets)


class MoodDataSet(Dataset):
    def __init__(self, csv_file, input_length):
        self.input_length = input_length
        self.raw = pd.read_csv(csv_file)
        self.data = process(self.raw)

    def __len__(self):
        return self.raw['id'].nunique()

    def __getitem__(self, idx):
        window_start = np.random.randint(low=0, high=self.data[0].shape[-1] - self.input_length)
        window_end = window_start + self.input_length
        return self.data[0][idx, :, window_start:window_end], self.data[1][idx, window_start:window_end]

from os.path import isfile
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset

# Threshold for too large NaN percentage
NAN_THRESH = .8


def _process(raw, split_ratio):
    if isfile('../Data/train_features.pkl') and isfile('../Data/train_targets.pkl') and isfile('../Data/valid_features.pkl') and isfile('../Data/valid_targets.pkl'):
        with open('../Data/train_features.pkl', 'rb') as f:
            train_features = pickle.load(f)
        with open('../Data/train_targets.pkl', 'rb') as f:
            train_targets = pickle.load(f)
        with open('../Data/valid_features.pkl', 'rb') as f:
            valid_features = pickle.load(f)
        with open('../Data/valid_targets.pkl', 'rb') as f:
            valid_targets = pickle.load(f)

        return (train_features, train_targets), (valid_features, valid_targets)

    split = round(split_ratio * raw.srch_id.nunique())

    # Remove if to many NaN values
    raw = raw.dropna(thresh=len(raw) * NAN_THRESH, axis=1)

    # Else fill with zeros
    raw = raw.fillna(0)

    # Increase booking weight
    raw.booking_bool = raw.booking_bool * 4

    # Irrelevant features
    to_remove = ['srch_id', 'date_time', 'position', 'click_bool', 'booking_bool']

    features, targets = [], []
    for _, group in raw.groupby('srch_id'):
        feature = torch.tensor(group.drop(to_remove, axis=1).values, dtype=torch.float)
        target = torch.tensor((group.click_bool + group.booking_bool).values, dtype=torch.float)
        features.append(feature)
        targets.append(target)

    train_features, valid_features = features[:split], features[split:]
    train_targets, valid_targets = targets[:split], targets[split:]

    with open('../Data/train_features.pkl', 'wb') as f:
        pickle.dump(train_features, f)
    with open('../Data/train_targets.pkl', 'wb') as f:
        pickle.dump(train_targets, f)
    with open('../Data/valid_features.pkl', 'wb') as f:
        pickle.dump(valid_features, f)
    with open('../Data/valid_targets.pkl', 'wb') as f:
        pickle.dump(valid_targets, f)

    return (train_features, train_targets), (valid_features, valid_targets)


def _process_test(raw):
    raw = raw.dropna(thresh=len(raw) * NAN_THRESH, axis=1)
    return raw.drop(['date_time'], axis=1).fillna(0)


class PropRankingSplit(Dataset):
    def __init__(self, data, split):
        self.split = split
        self.features, self.targets = data[0], data[1]
        self.num_features = self.features[0].shape[1] if len(self.features) > 0 else 0

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PropRanking(object):
    def __init__(self, split_ratio=.8, train_path=None, test_path=None):
        if train_path:
            self.train, self.valid = _process(pd.read_pickle(train_path), split_ratio)

        if test_path:
            self.test = _process_test(pd.read_pickle(test_path))

    def get_train(self):
        return PropRankingSplit(self.train, 'train')

    def get_validation(self):
        return PropRankingSplit(self.valid, 'valid')

    def get_test(self):
        return self.test


if __name__ == '__main__':
    rawie = pd.read_pickle("../Data/training_set_VU_DM.pkl")
    (traindata, _), (valid_data, valid_targets) = _process(rawie, .8)
    print(valid_data[0].shape, valid_targets[0].shape)

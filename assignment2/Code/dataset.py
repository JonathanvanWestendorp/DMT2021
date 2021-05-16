import pandas as pd
import torch
from torch.utils.data import Dataset

# Threshold for too large NaN percentage
NAN_THRESH = .8


def _process(raw, split_ratio):
    split = round(split_ratio * len(raw))

    # Remove if to many NaN values
    raw = raw.dropna(thresh=len(raw) * NAN_THRESH, axis=1)

    # Else fill with zeros
    raw = raw.fillna(0)

    # Increase booking weight
    raw.booking_bool = raw.booking_bool * 4

    raw_train, raw_valid = raw.iloc[:split], raw.iloc[split:]

    return _process_train(raw_train), _process_valid(raw_valid)


def _process_train(raw):
    # Calculate relevance labels
    targets = torch.tensor(raw.click_bool + raw.booking_bool, dtype=torch.float)

    # Irrelevant for learning ranking or used for target calculation or not present in test set
    to_remove = ['srch_id', 'date_time', 'position', 'click_bool', 'booking_bool']
    features = torch.tensor(raw.drop(to_remove, axis=1).values, dtype=torch.float)

    return features, targets


def _process_valid(raw):
    # Calculate relevance labels
    grouped_targets = pd.concat([raw.srch_id, raw.click_bool + raw.booking_bool], axis=1).groupby('srch_id')
    targets = [t.drop('srch_id', axis=1).values for _, t in grouped_targets]

    # The same features are dropped for the valid set except for srch_id because we need that for grouping
    to_remove = ['date_time', 'position', 'click_bool', 'booking_bool']

    grouped_features = raw.drop(to_remove, axis=1).groupby('srch_id')
    features = [torch.tensor(q.drop('srch_id', axis=1).values, dtype=torch.float) for _, q in grouped_features]

    return features, targets


def _process_test(raw):
    raw = raw.dropna(thresh=len(raw) * NAN_THRESH, axis=1)
    return raw.drop(['date_time'], axis=1).fillna(0)


class PropRankingSplit(Dataset):
    def __init__(self, data, split):
        self.split = split
        self.features, self.targets = data[0], data[1]
        self.num_features = self.features.shape[1] if split == 'train' else self.features[0].shape[1]

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

import pandas as pd
import torch
from torch.utils.data import Dataset


def _process(raw, split_ratio):
    split = round(split_ratio * len(raw))

    raw = raw.fillna(0)

    # Calculate relevance labels TODO: Improve this... Maybe classification instead of regression
    targets = torch.tensor(1 / raw.position + raw.click_bool + raw.booking_bool, dtype=torch.float)

    # Irrelevant for learning ranking or used for target calculation or not present in test set
    to_remove = ['srch_id', 'date_time', 'position', 'click_bool', 'booking_bool', 'gross_bookings_usd']
    features = torch.tensor(raw.drop(to_remove, axis=1).values, dtype=torch.float)

    train_targets, valid_targets = targets[:split], targets[split:]
    train_features, valid_features = features[:split], features[split:]

    return (train_features, train_targets), (valid_features, valid_targets)


def _process_test(raw):
    return torch.tensor(raw.drop(['srch_id', 'date_time'], axis=1).values)


class PropRankingSplit(Dataset):
    def __init__(self, data, split):
        self.split = split
        self.features, self.targets = data[0], data[1]
        self.num_features = self.features.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class PropRanking(object):
    def __init__(self, split_ratio, train_path, test_path):
        self.train, self.valid = _process(pd.read_pickle(train_path), split_ratio)
        self.test = _process_test(pd.read_pickle(test_path))

    def get_train(self):
        return PropRankingSplit(self.train, 'train')

    def get_validation(self):
        return PropRankingSplit(self.valid, 'valid')

    def get_test(self):
        return self.test


if __name__ == '__main__':
    _process(pd.read_pickle('../Data/training_set_VU_DM.pkl'), .8)

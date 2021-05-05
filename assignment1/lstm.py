from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MoodPredictionModel(nn.Module):
    def __init__(self, seq_length, n_features, n_hidden, n_layers):
        super(MoodPredictionModel, self).__init__()

        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True)

        self.l_linear = nn.Linear(n_hidden * seq_length, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out.contiguous().view(x.size()[0], -1)
        return self.l_linear(x)

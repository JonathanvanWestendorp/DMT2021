from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

# TODO Make Model
class MoodPredictionModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(MoodPredictionModel, self).__init__()

        embedding_dim = round(.5 * lstm_num_hidden)

        self.encode = nn.Embedding(vocabulary_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, lstm_num_hidden, lstm_num_layers)
        self.decode = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x):
        x = self.encode(x)
        h, state = self.lstm(x)
        p = self.decode(h).permute(0, 2, 1)
        return p

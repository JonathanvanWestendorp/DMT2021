"""
This module implements a bidirectional LSTM in PyTorch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


class biLSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(biLSTM, self).__init__()

        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.decode = nn.Linear(2 * hidden_dim, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

        self.fwLayer = LSTMCell(seq_length, input_dim, hidden_dim)
        self.bwLayer = LSTMCell(seq_length, input_dim, hidden_dim)

        nn.init.kaiming_normal_(self.decode.weight, mode='fan_in')

    def forward(self, x):
        c = torch.zeros((self.batch_size, self.hidden_dim), device=self.device)
        h = torch.zeros((self.batch_size, self.hidden_dim), device=self.device)

        h_fw = self.fwLayer(x, c, h)
        h_bw = self.bwLayer(torch.fliplr(x), c, h)

        # Concatenate forward and backward h
        h = torch.cat((h_fw, h_bw), 1)

        p = self.decode(h)

        out = self.logSoftmax(p)
        return out


class LSTMCell(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim):

        super(LSTMCell, self).__init__()

        self.encode = nn.Embedding(seq_length, input_dim)

        k = np.sqrt(2 / hidden_dim)
        self.Wih = nn.Parameter(torch.randn(4 * hidden_dim, input_dim) * k)
        self.Whh = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim) * k)
        self.bih = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.bhh = nn.Parameter(torch.zeros(4 * hidden_dim))

    def forward(self, x, c, h):
        x = self.encode(x.long()).permute(1, 0, 2)

        for inp in x:
            lstm_gates = inp.mm(self.Wih.T) + self.bih + h.mm(self.Whh.T) + self.bhh
            i, f, g, o = lstm_gates.chunk(4, 1)

            i = i.sigmoid()
            f = f.sigmoid()
            g = g.tanh()
            o = o.sigmoid()

            c = g * i + c * f
            h = c.tanh() * o

        return h

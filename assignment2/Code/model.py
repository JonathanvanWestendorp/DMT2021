from torch import nn


class NeuralModule(nn.Module):
    def __init__(self, num_features, output_dim):
        super(NeuralModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, output_dim))

    def forward(self, x):
        return self.net(x)

from torch import nn


class Regression(nn.Module):
    def __init__(self, num_features):
        super(Regression, self).__init__()

        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 10)
        self.out = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.out(x)
        return x

import torch
import torch.nn as nn


class RankNetNetwork(nn.Module):
    def __init__(self, config, item_feature_size):
        super(RankNetNetwork, self).__init__()
        self.item_feature_size = item_feature_size

        self.prepare = nn.Sequential(
            nn.Linear(item_feature_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
        )

        self.model = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
        )

        self.output_sig = nn.Sigmoid()

    def forward(self, input_1, input_2):
        input_1 = self.prepare(input_1)
        input_2 = self.prepare(input_2)

        cat = torch.cat((input_1, input_2), 1)

        return self.model(cat)

    def predict(self, input):
        input = self.prepare(input)
        return self.output_sig(self.model(input))

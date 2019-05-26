import torch
import torch.nn as nn

from utility.noise import GaussianNoise


class RankNetNetwork(nn.Module):
    def __init__(self, config, device, item_feature_size):
        super(RankNetNetwork, self).__init__()
        self.item_feature_size = item_feature_size

        self.prepare = nn.Sequential(
            nn.Dropout(0.5),
            GaussianNoise(device=device, sigma=1.0),
            nn.BatchNorm1d(item_feature_size),
            nn.Linear(item_feature_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
        )

        self.model = nn.Sequential(
            nn.Linear(16 * 2, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
        )

        self.output_sig = nn.Sigmoid()

    def forward(self, input_1, input_2):
        input_1 = input_1.float()
        input_2 = input_2.float()

        input_1 = self.prepare(input_1)
        input_2 = self.prepare(input_2)

        cat = torch.cat((input_1, input_2), 1)

        return self.model(cat)

    def predict(self, input_1, input_2):
        #input_1 = (input_1 > 0.5).float()
        #input_2 = (input_2 > 0.5).float()
        #input_1 = self.output_sig(input_1)
        #input_2 = self.output_sig(input_2)

        result = self.forward(input_1=input_1, input_2=input_2)
        return self.output_sig(result)


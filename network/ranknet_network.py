import torch.nn as nn


class RankNetNetwork(nn.Module):
    def __init__(self, config, target_item_size):
        super(RankNetNetwork, self).__init__()
        self.target_item_size = target_item_size

        self.model = nn.Sequential(
            nn.Linear(target_item_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_1, input_2):
        s1 = self.model(input_1)
        s2 = self.model(input_2)

        return self.output_sig(s1 - s2)

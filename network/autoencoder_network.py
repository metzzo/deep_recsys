import torch.nn as nn
import torch


class AutoEncoderNetwork(nn.Module):
    def __init__(self, device):
        super(AutoEncoderNetwork, self).__init__()
        self.item_size = 256 # TODO: change

        self.encoder = nn.Sequential(
            nn.Linear(self.item_size, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, self.item_size),
            nn.BatchNorm1d(num_features=self.item_size),
            nn.ReLU(),
        )

        self.device = device

        for name, param in self.encoder.named_parameters():
            nn.init.normal_(param)
        for name, param in self.decoder.named_parameters():
            nn.init.normal_(param)

    def forward(self, items: torch.Tensor):

        encoded = self.encoder(items)
        decoded = self.decoder(encoded)

        return encoded, decoded


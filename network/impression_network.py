import torch.nn as nn
import torch


class ImpressionRankNetwork(nn.Module):
    def __init__(self, config, item_size, device):
        super(ImpressionRankNetwork, self).__init__()
        self.device = device
        self.rank = nn.Sequential(
            nn.Linear(202 * 2, 202),
            nn.BatchNorm1d(num_features=202),
            nn.ReLU(),
            nn.Linear(202, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(16 * 25 + 25, 25),
            nn.BatchNorm1d(num_features=25),
            nn.ReLU(),
            nn.Linear(25, 25),
        )

    def rank_impression(self, impression, wants):
        concat = torch.cat([impression, wants], dim=1)
        return self.rank(concat) #  self.rank((impression + wants)/2.0).float() #

    def forward(self, impressions: [torch.Tensor], wants: torch.Tensor, prices: [torch.Tensor]):
        prices = torch.stack(prices).to(device=self.device).float()

        wants = wants.detach()
        wants = (wants > 0.5).float()

        impressions = [
            impression.to(self.device).float() for impression in impressions
        ]
        padded_sequence = torch.nn.utils.rnn.pad_sequence(impressions, batch_first=True)

        ranked = [self.rank_impression(padded_sequence[:, i, :], wants) for i in range(0, padded_sequence.size(1))]
        ranked = torch.cat(ranked, dim=1)
        ranked = torch.cat([ranked, prices], dim=1)
        ranked = self.combine(ranked) # torch.rand(ranked.shape).cuda()
        return ranked

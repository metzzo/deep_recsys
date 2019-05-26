import torch.nn as nn
import torch


class ImpressionRankNetwork(nn.Module):
    def __init__(self, config, item_size, device):
        super(ImpressionRankNetwork, self).__init__()
        self.hidden_dim = 512
        self.gru = nn.GRU(item_size * 2, self.hidden_dim, batch_first=True, num_layers=1, dropout=0.0)
        self.item_size = item_size
        self.target_item_size = item_size

        fcn_size = 100
        self.hidden2tag = nn.Sequential(
            #nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, fcn_size),
            nn.BatchNorm1d(num_features=fcn_size),
            nn.ReLU(),
            nn.Linear(fcn_size, item_size),
            nn.Sigmoid(),
        )
        self.device = device

        self.sigmoid = nn.Sigmoid()

        for name, param in self.hidden2tag.named_parameters():
            nn.init.normal_(param)

    def forward(self, impressions: [torch.Tensor], wants: torch.Tensor):
        wants = wants.detach()
        wants = (wants > 0.5).float()
        #wants = self.sigmoid(wants)

        impressions = [
            impression.to(self.device).float() for impression in impressions
        ]
        padded_sequence = torch.nn.utils.rnn.pad_sequence(impressions, batch_first=True)
        wants = wants.reshape(wants.size(0), 1, wants.size(1)).repeat(1, padded_sequence.size(1), 1)
        padded_sequence = torch.cat([padded_sequence, wants], dim=2)

        padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            padded_sequence,
            torch.ones(padded_sequence.size(0)).to(self.device) * 25.0,
            batch_first=True
        )

        _, hidden = self.gru(padded_sequence)
        hidden = hidden[-1].view(hidden.size(1), -1)

        item_space = self.hidden2tag(hidden)

        return item_space


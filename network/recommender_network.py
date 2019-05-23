import torch.nn as nn
import torch


class RecommenderNetwork(nn.Module):
    def __init__(self, config, item_size, target_item_size):
        super(RecommenderNetwork, self).__init__()
        self.hidden_dim = config.get('hidden_dim')
        self.gru = nn.GRU(item_size, self.hidden_dim, batch_first=True, num_layers=3, dropout=0.1)
        self.item_size = item_size
        self.target_item_size = target_item_size

        self.hidden2tag = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(),
            nn.Linear(200, target_item_size),
            nn.Sigmoid(),
        )

        for name, param in self.hidden2tag.named_parameters():
            nn.init.normal_(param)

    def forward(self, sessions: torch.Tensor, session_lengths: torch.Tensor):

        sessions = torch.nn.utils.rnn.pack_padded_sequence(sessions, session_lengths, batch_first=True)

        _, hidden = self.gru(sessions)
        hidden = hidden[-1].view(hidden.size(1), -1)

        item_space = self.hidden2tag(hidden)

        return item_space


import torch.nn as nn
import torch


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim, item_size, target_item_size, device):
        super(LSTMNetwork, self).__init__()
        from main import BATCH_SIZE
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(item_size, hidden_dim, batch_first=True, num_layers=1)
        self.item_size = item_size
        self.target_item_size = target_item_size
        self.batch_size = BATCH_SIZE

        self.hidden2tag = nn.Sequential(
            nn.Linear(hidden_dim, 200),
            nn.BatchNorm1d(num_features=200),
            nn.ReLU(),
            nn.Linear(200, target_item_size),
            nn.Sigmoid(),
        )

        self.device = device

        for name, param in self.hidden2tag.named_parameters():
            nn.init.normal_(param)

    def forward(self, sessions: torch.Tensor, session_lengths: torch.Tensor):

        sessions = torch.nn.utils.rnn.pack_padded_sequence(sessions, session_lengths, batch_first=True)

        _, hidden = self.gru(sessions)
        hidden = hidden[-1].view(hidden.size(1), -1)

        item_space = self.hidden2tag(hidden)

        return item_space


import torch.nn as nn
import torch


class RecommenderNetwork(nn.Module):
    def __init__(self, config, item_size, target_item_size):
        super(RecommenderNetwork, self).__init__()
        self.hidden_dim = config.get('hidden_dim')

        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=item_size, kernel_size=(1, item_size), stride=1),
            nn.BatchNorm2d(num_features=item_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, item_size)),
        )

        self.gru = nn.GRU(item_size, self.hidden_dim, batch_first=True, num_layers=config.get('num_gru_layers'), dropout=0.0)
        self.target_item_size = target_item_size

        fcn_size = config.get('fc_layer_size')
        self.hidden2tag = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, fcn_size),
            nn.BatchNorm1d(num_features=fcn_size),
            nn.ReLU(),
            nn.Linear(fcn_size, target_item_size),
            nn.Sigmoid(),
        )

        for name, param in self.hidden2tag.named_parameters():
            nn.init.normal_(param)

    def forward(self, sessions: torch.Tensor, session_lengths: torch.Tensor):
        sessions = sessions.reshape(sessions.size(0), 1, sessions.size(1), sessions.size(2))
        sessions = self.embedding(sessions)
        sessions = sessions.permute([0, 2, 1, 3])
        sessions = sessions.reshape(sessions.size(0), sessions.size(1), -1)
        sessions = torch.nn.utils.rnn.pack_padded_sequence(sessions, session_lengths, batch_first=True)

        _, hidden = self.gru(sessions)
        hidden = hidden[-1].view(hidden.size(1), -1)

        item_space = self.hidden2tag(hidden)

        return item_space


import torch.nn as nn
import torch
"""

val  Score:  0.54
New best \o/

"""

class RecommenderNetwork(nn.Module):
    def __init__(self, config, item_size, target_item_size):
        super(RecommenderNetwork, self).__init__()
        self.hidden_dim = config.get('hidden_dim')
        self.item_size = item_size
        """
        self.embedding = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=item_size, kernel_size=(2, item_size), stride=1, padding=1),
            #nn.BatchNorm2d(num_features=item_size),
            #nn.ReLU(),
            #nn.AvgPool2d(kernel_size=(1, item_size), padding=0),
        )
        """

        self.gru = nn.GRU(
            item_size,
            self.hidden_dim,
            batch_first=True,
            num_layers=config.get('num_gru_layers'),
            dropout=0.0,
            bidirectional=True,
        )
        self.target_item_size = target_item_size

        fcn_size = config.get('fc_layer_size')
        self.hidden2tag = nn.Sequential(
            nn.Linear(self.hidden_dim, target_item_size),
            nn.BatchNorm1d(num_features=target_item_size),
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


import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim, item_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(item_size, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, item_size)

    def forward(self, input: torch.Tensor):
        lstm_out, _ = self.lstm(input)
        item_space = self.hidden2tag(lstm_out.view(input.size(0), -1))
        return item_space

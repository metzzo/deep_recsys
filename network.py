import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTMNetwork(nn.Module):
    def __init__(self, hidden_dim, item_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(item_size, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, item_size)
        self.item_size = item_size

    def forward(self, input: torch.Tensor):
        input = input.permute(1, 0, 2)

        lstm_out, _ = self.lstm(input)
        lstm_out = lstm_out.view(lstm_out.size(0), self.hidden_dim, -1)
        last_step = lstm_out[-1].view(lstm_out.size(2), -1)

        item_space = self.hidden2tag(last_step)
        return item_space

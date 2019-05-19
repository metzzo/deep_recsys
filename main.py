import os
import random

from torch import nn, optim
from torch.utils.data import DataLoader

from network import LSTMNetwork
from recsys_dataset import RecSysDataset

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
DEBUG = True

HIDDEN_DIM = 6


if __name__ == '__main__':
    random.seed(42)

    dataset = RecSysDataset()


    model = LSTMNetwork(
        hidden_dim=HIDDEN_DIM,
        item_size=dataset.item_size
    )
    loss_function = nn.KLDivLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=dataset.collator)

    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for item_sequence, targets in train_loader:
            model.zero_grad()

            tag_scores = model(item_sequence)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()



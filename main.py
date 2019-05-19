import os
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from network import LSTMNetwork
from recsys_dataset import RecSysDataset

import torch.nn.functional as F

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
DEBUG = True

HIDDEN_DIM = 6

NUM_WORKERS = 0


if __name__ == '__main__':
    random.seed(42)

    train_dataset = RecSysDataset(split=0.8, before=True)
    val_dataset = RecSysDataset(split=0.8, before=True)

    network = LSTMNetwork(
        hidden_dim=HIDDEN_DIM,
        item_size=train_dataset.item_size
    )
    loss_function = nn.KLDivLoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01)

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS, collate_fn=val_dataset.collator),
    }

    for epoch in range(100):
        print('-'*30)
        print("Epoch ", epoch)
        for phase in ['train', 'val']:
            for item_properties, item_impressions, impression_ids, targets in data_loaders[phase]:
                if phase == 'train':
                    network.train()
                else:
                    network.eval()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    item_scores = network(item_properties).double()
                    loss = loss_function(item_scores, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        network.eval()

                    for impression_id, item_impression, item_score in zip(impression_ids, item_impressions, item_scores):
                        item_score_repeated = item_score.repeat(len(item_impression), 1)
                        sim = F.cosine_similarity(item_score_repeated, item_impression)
                        sorted = torch.argsort(sim, descending=True)
                        print(impression_id)
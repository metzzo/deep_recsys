import os
import random

import progressbar
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from functions import score_submissions, get_reciprocal_ranks
from network import LSTMNetwork
from recsys_dataset import RecSysDataset

import torch.nn.functional as F
import pandas as pd
import numpy as np

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
DEBUG = True

HIDDEN_DIM = 128

NUM_WORKERS = 0

BATCH_SIZE = 32

if __name__ == '__main__':
    random.seed(42)

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_dataset = RecSysDataset(split=0.8, before=True)
    val_dataset = RecSysDataset(split=0.8, before=True)

    network = LSTMNetwork(
        hidden_dim=HIDDEN_DIM,
        item_size=train_dataset.item_size
    )
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=0.01)

    datasets = {
        "train": train_dataset,
        "val": val_dataset,
    }

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=val_dataset.collator),
    }

    predictions = {
        "train": pd.Series([''] * len(train_dataset)),
        "val": pd.Series([''] * len(val_dataset))
    }

    losses = np.zeros(int(len(train_dataset) / BATCH_SIZE + 1.0))

    network = network.to(device)
    for epoch in range(100):
        print('-'*15, "Epoch: ", epoch, '\t', '-'*15)
        for phase in ['train', 'val']:
            cur_dataset = datasets[phase]
            cur_predictions = predictions[phase]

            if phase == 'train':
                network.train()
            else:
                network.eval()

            predictions_ptr = 0
            losses.fill(0)

            for idx, (item_properties, item_impressions, impression_ids, targets) in progressbar.progressbar(enumerate(data_loaders[phase])):
                item_properties = item_properties.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    item_scores = network(item_properties).double()
                    loss = loss_function(item_scores, targets)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        losses[idx] = loss.item()

                    for impression_id, item_impression, item_score in zip(impression_ids, item_impressions, item_scores):
                        item_score_repeated = item_score.repeat(len(item_impression), 1)
                        sim = F.cosine_similarity(item_score_repeated.cpu(), item_impression)
                        sorted = torch.argsort(sim, descending=True)
                        sorted_impressions = ' '.join(torch.gather(impression_id, 0, sorted).detach().cpu().numpy().astype(str))
                        cur_predictions.iloc[predictions_ptr] = sorted_impressions
                        predictions_ptr += 1

            ground_truth_df = cur_dataset.get_submission()
            predicted_df = cur_dataset.get_submission(cur_predictions)

            score = score_submissions(df_subm=predicted_df, df_gt=ground_truth_df, objective_function=get_reciprocal_ranks)
            print(phase, " Score: ", score)
            if phase == 'train':
                print(phase, " Loss: ", losses.mean())

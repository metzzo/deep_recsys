import datetime
import os
import random
import shutil

import progressbar
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utility.functions import score_submissions, get_reciprocal_ranks, SUBM_INDICES
from network import LSTMNetwork
from dataset.recsys_dataset import RecSysDataset

import torch.nn.functional as F
import pandas as pd
import numpy as np

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')

MODEL_NAME = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
MODEL_BASE_PATH = os.path.join(DATA_PATH, 'ae_model')
MODEL_PATH = os.path.join(DATA_PATH, 'ae_model', MODEL_NAME + ".model")
DEBUG = False

HIDDEN_DIM = 128

PATIENCE = 100

BATCH_SIZE = 128

CALC_BASELINE = True

NUM_EPOCHS = 1000

if __name__ == '__main__':
    random.seed(42)

    use_cuda = torch.cuda.is_available()
    print("Uses CUDA: {0}".format(use_cuda))
    device = torch.device("cuda:0" if use_cuda else "cpu")

    train_dataset = RecSysDataset(split=0.7, before=True, include_impressions=False, train_mode=True)
    train_val_dataset = RecSysDataset(split=0.7, before=True, include_impressions=True)
    val_dataset = RecSysDataset(split=0.7, before=False, include_impressions=True)

    network = LSTMNetwork(
        hidden_dim=HIDDEN_DIM,
        item_size=train_dataset.item_size,
        target_item_size=train_dataset.target_item_size,
        device=device
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.01, weight_decay=0.00001)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=75)


    datasets = {
        "train": train_dataset,
        "train_val": train_dataset,
        "val": val_dataset,
    }

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_dataset.collator),
        "train_val": DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=train_val_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=val_dataset.collator),
    }

    sizes = {
        "train": int(len(train_dataset) / BATCH_SIZE) + 1,
        "train_val": int(len(train_dataset) / BATCH_SIZE) + 1,
        "val": int(len(val_dataset) / BATCH_SIZE) + 1,
    }

    losses = np.zeros(int(len(train_dataset) / BATCH_SIZE + 1.0))

    network = network.to(device)

    best_score_so_far = None

    cur_patience = 0
    for epoch in range(NUM_EPOCHS):
        print('-'*15, "Epoch: ", epoch + 1, '\t', '-'*15)
        for phase in ['train',  'val']: # 'train_val',
            cur_dataset = datasets[phase]
            cur_predictions = pd.DataFrame.from_dict({
                'user_id': [''] * len(cur_dataset),
                'session_id': [''] * len(cur_dataset),
                'timestamp': [''] * len(cur_dataset),
                'step': [''] * len(cur_dataset),
                'item_recommendations': [''] * len(cur_dataset),
            })
            prediction_ptr = 0
            if phase == 'train':
                network.train()
            else:
                network.eval()

            do_validation = phase != 'train'
            losses.fill(0)
            with progressbar.ProgressBar(max_value=sizes[phase], redirect_stdout=True) as bar:
                for idx, data in enumerate(data_loaders[phase]):
                    if do_validation:
                        sessions, session_lengths, session_targets, item_impressions, impression_ids, ids = data
                    else:
                        sessions, session_lengths, session_targets = data
                        impression_ids = None
                        item_impressions = None
                        ids = None

                    sessions = sessions.to(device)
                    session_targets = session_targets.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        item_scores = network(sessions, session_lengths).float()

                        if phase == 'train':
                            loss = loss_function(item_scores, session_targets)
                            loss.backward()
                            optimizer.step()
                            losses[idx] = loss.item()

                    if do_validation:
                        item_scores = item_scores.to(device)
                        for id, impression_id, item_impression, item_score in zip(ids, impression_ids, item_impressions, item_scores):
                            impression_id = impression_id.to(device)
                            item_impression = item_impression.to(device).float()

                            item_score_repeated = item_score.repeat(len(item_impression), 1)
                            sim = F.cosine_similarity(item_score_repeated, item_impression)
                            sorted = torch.argsort(sim, descending=True)
                            sorted_impressions = ' '.join(torch.gather(impression_id, 0, sorted).cpu().numpy().astype(str))
                            cur_pred = cur_predictions.iloc[prediction_ptr]
                            cur_pred.at['item_recommendations'] = sorted_impressions
                            cur_pred.at['user_id'] = id['user_id']
                            cur_pred.at['session_id'] = id['session_id']
                            cur_pred.at['timestamp'] = id['timestamp']
                            cur_pred.at['step'] = id['step']
                            prediction_ptr += 1
                    bar.update(idx)
            if do_validation:
                cur_predictions.set_index(SUBM_INDICES, inplace=True)

                ground_truth_df = cur_dataset.get_submission()
                predicted_df = cur_predictions

                score = score_submissions(df_subm=predicted_df, df_gt=ground_truth_df, objective_function=get_reciprocal_ranks)
                print(phase, " Score: ", score)
                lr_scheduler.step(score)
                if phase == 'val':
                    if best_score_so_far is None or score > best_score_so_far:
                        torch.save(network.state_dict(), MODEL_PATH)
                        best_score_so_far = score
                        cur_patience = 0
                        print("New best \\o/")
                    else:
                        cur_patience += 1
                        if cur_patience > PATIENCE:
                            print("Not patient anymore => Quit")
                            break
            if phase == 'train':
                print(phase, " Loss: ", losses.mean())
        if cur_patience > PATIENCE:
            break

    print("Final best model: ", best_score_so_far)
    shutil.move(
        MODEL_PATH,
        os.path.join(
            MODEL_BASE_PATH,
            '{}_{}.model'.format(MODEL_NAME, round(best_score_so_far, 2))
        )
    )

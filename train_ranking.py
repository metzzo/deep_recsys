import os
import random
import shutil

import numpy as np
import pandas as pd
import progressbar
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset.ranknet_dataset import RankNetDataset, RankNetData
from dataset.recsys_dataset import RecSysDataset

from network.impression_network import ImpressionRankNetwork
from network.recommender_network import RecommenderNetwork
from ranking_configs import ranking_configs, prepare_config
from train_recommender import get_rec_sys_data, DATA_PATH
from utility.helpers import get_string_timestamp
from utility.prediction import Prediction
from utility.split_utility import AllSamplesExceptStrategy, RandomSampleStrategy, AllSamplesIncludeStrategy


def train(config, state=None, model=None):
    MODEL_NAME = get_string_timestamp()
    MODEL_BASE_PATH = os.path.join(DATA_PATH, 'ranking_model')
    MODEL_PATH = os.path.join(DATA_PATH, 'ranking_model', MODEL_NAME + ".pth")

    random.seed(42)

    config = prepare_config(config, model=model)

    batch_size = config.get('batch_size')
    patience = config.get('patience')
    num_epochs = config.get('num_epochs')
    learning_rate = config.get('learning_rate')
    phases = config.get('phases')
    reduce_factor = config.get('reduce_factor')
    reduce_patience = config.get('reduce_patience')
    weight_decay = config.get('weight_decay')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data = get_rec_sys_data(
        size=config.get('dataset_size')
    )

    train_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=RandomSampleStrategy(split=0.7),
        include_impressions=True,
        train_mode=False
    )
    train_val_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=AllSamplesIncludeStrategy(include=train_dataset.session_ids),
        include_impressions=True
    )
    val_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=AllSamplesExceptStrategy(exclude=train_dataset.session_ids),
        include_impressions=True
    )

    recommender_state = torch.load(config.get('recommender_model'))

    recommend_network = RecommenderNetwork(
        config=recommender_state.get('config'),
        item_size=train_dataset.item_size,
        target_item_size=train_dataset.target_item_size,
    )
    recommend_network.load_state_dict(recommender_state.get('network_state_dict'))
    recommend_network.eval()

    network = ImpressionRankNetwork(
        config=config,
        device=device,
        item_size=train_dataset.item_size,
    )
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=reduce_factor, patience=reduce_patience)

    start_epoch = 0
    best_score_so_far = None
    best_baseline_so_far = None

    if state:
        optimizer.load_state_dict(state['optimizer_state_dict'])
        network.load_state_dict(state['network_state_dict'])
        start_epoch = state['epoch']
        best_score_so_far = state['best_score_so_far']

    datasets = {
        "train": train_dataset,
        "train_val": train_dataset,
        "val": val_dataset,
    }

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                 collate_fn=train_dataset.collator),
        "train_val": DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                collate_fn=train_val_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                          collate_fn=val_dataset.collator),
    }

    sizes = {
        "train": int(len(train_dataset) / batch_size) + 1,
        "train_val": int(len(train_dataset) / batch_size) + 1,
        "val": int(len(val_dataset) / batch_size) + 1,
    }

    losses = np.zeros(int(len(train_dataset) / batch_size + 1.0))

    network = network.to(device)
    recommend_network = recommend_network.to(device)

    cur_patience = 0
    print("Uses CUDA: {0}".format(use_cuda))
    for epoch in range(start_epoch, num_epochs):
        print('-' * 15, "Epoch: ", epoch + 1, '\t', '-' * 15)

        cur_phase = phases

        for phase in cur_phase:  # 'train_val',
            cur_dataset = datasets[phase]
            cur_prediction = Prediction(
                dataset=cur_dataset,
                device=device,
                use_cosine_similarity=False,
            )
            do_validation = phase != 'train'
            if phase == 'train':
                network.train()
            else:
                network.eval()

            losses.fill(0)
            with progressbar.ProgressBar(max_value=sizes[phase], redirect_stdout=True) as bar:
                for idx, data in enumerate(data_loaders[phase]):
                    sessions, session_lengths, _, item_impressions, impression_ids, target_index, prices, ids = data

                    sessions = sessions.to(device)

                    if phase == 'train':
                        optimizer.zero_grad()
                        item_scores = recommend_network(sessions, session_lengths).float()

                        with torch.set_grad_enabled(True):
                            predicted = network(item_impressions, item_scores, prices)
                            target_index = torch.stack(target_index, dim=0).to(device)
                            loss = loss_function(predicted, target_index)
                            loss.backward()
                            optimizer.step()
                            losses[idx] = loss.item()
                    else:
                        with torch.set_grad_enabled(False):
                            item_scores = recommend_network(sessions, session_lengths).float()
                            selected_impression = network(item_impressions, item_scores, prices)

                            cur_prediction.add_predictions(
                                ids=ids,
                                impression_ids=impression_ids,
                                item_impressions=item_impressions,
                                item_scores=item_scores,
                                selected_impression=selected_impression,
                            )
                    bar.update(idx)
            if do_validation:
                score, _ = cur_prediction.get_score()

                print(phase, " Score: ", score)
                lr_scheduler.step(score)
                lr_scheduler.step(score)
                if phase == 'val':
                    if best_score_so_far is None or score > best_score_so_far:
                        best_score_so_far = score
                        torch.save({
                            'epoch': epoch,
                            'best_score_so_far': best_score_so_far,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'network_state_dict': network.state_dict(),
                            'config': config,
                        }, MODEL_PATH)
                        cur_patience = 0
                        print("New best \\o/")
                    else:
                        cur_patience += 1
                        if cur_patience > patience:
                            print("Not patient anymore => Quit")
                            break
            if not do_validation:
                print(phase, " Loss: ", losses.mean())
        if cur_patience > patience:
            break

    print("Final best model: ", best_score_so_far)
    target_path = os.path.join(
            MODEL_BASE_PATH,
            '{}_{}.pth'.format(MODEL_NAME, round(best_score_so_far, 2))
    )
    shutil.move(
        MODEL_PATH,
        target_path
    )

    return best_score_so_far, best_baseline_so_far, target_path


def train_ranking(model=None):
    print("Train Ranking")
    best_score_so_far = 0
    best_config = None
    best_path = None
    best_baseline_so_far = 0
    for config in ranking_configs:
        print("Train config ", config['name'])
        current_score, current_baseline, current_path = train(config, model=model)
        if current_score > best_score_so_far:
            best_score_so_far = current_score
            best_baseline_so_far = current_baseline
            best_config = config
            best_path = current_path

    print("-" * 30)
    print("Best Config: ", str(best_config))
    print("Best Score: ", str(best_score_so_far))
    print("Best Baseline: ", str(best_baseline_so_far))
    print("Best Path: ", str(best_path))

    return best_path


if __name__ == '__main__':
    train_ranking()


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

from network.impression_network import ImpressionRankNetwork
from ranking_configs import ranking_configs, prepare_config
from utility.helpers import get_string_timestamp
from utility.split_utility import AllSamplesExceptStrategy, RandomSampleStrategy, AllSamplesIncludeStrategy

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')

MODEL_NAME = get_string_timestamp()
MODEL_BASE_PATH = os.path.join(DATA_PATH, 'ranking_model')
MODEL_PATH = os.path.join(DATA_PATH, 'ranking_model', MODEL_NAME + ".pth")

rank_net_data = None


def load_ranknet_network(path, device):
    state = torch.load(path)

    config = prepare_config(state.get('config'))
    network_state_dict = state.get('network_state_dict') or state

    network = RankNetNetwork(
        config=config,
        device=device,
        item_feature_size=202  # TODO do not hardcode
    )
    network.load_state_dict(network_state_dict)
    network.eval()

    return network


def get_rank_net_data(dataset_size):
    global rank_net_data

    if rank_net_data is None:
        rank_net_data = RankNetData(
            name='rank_data.p',
            size=dataset_size,
        )

    return rank_net_data


def train(config, state=None):
    random.seed(42)

    config = prepare_config(config)

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

    rank_net_data = get_rank_net_data(dataset_size=config.get('dataset_size'))

    train_dataset = RankNetDataset(
        data=rank_net_data,
        split_strategy=RandomSampleStrategy(split=0.7),
    )
    train_val_dataset = RankNetDataset(
        data=rank_net_data,
        split_strategy=AllSamplesIncludeStrategy(include=train_dataset.session_ranking_indices),
    )
    val_dataset = RankNetDataset(
        data=rank_net_data,
        split_strategy=AllSamplesExceptStrategy(exclude=train_dataset.session_ranking_indices),
    )

    network = ImpressionRankNetwork(
        config=config,
        device=device,
        item_size=202,
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
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        "train_val": DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }

    sizes = {
        "train": int(len(train_dataset) / batch_size) + 1,
        "train_val": int(len(train_dataset) / batch_size) + 1,
        "val": int(len(val_dataset) / batch_size) + 1,
    }

    losses = np.zeros(int(len(train_dataset) / batch_size + 1.0))

    network = network.to(device)

    cur_patience = 0
    print("Uses CUDA: {0}".format(use_cuda))
    for epoch in range(start_epoch, num_epochs):
        print('-' * 15, "Epoch: ", epoch + 1, '\t', '-' * 15)

        cur_phase = phases
        if epoch == num_epochs - 1:
            cur_phase = ['train', 'train_val', 'val']  # for last epoch do also train_val to find out if there was overfitting

        for phase in cur_phase:
            cur_dataset = datasets[phase]

            dict_data = {
                "predicted": [0] * len(cur_dataset),
                "target": [0] * len(cur_dataset),
                "baseline": [0] * len(cur_dataset),
            }
            cur_prediction = pd.DataFrame.from_dict(dict_data)
            prediction_ptr = 0
            if phase == 'train':
                network.train()
            else:
                network.eval()

            do_validation = phase != 'train'
            losses.fill(0)
            with progressbar.ProgressBar(max_value=sizes[phase], redirect_stdout=True) as bar:
                for idx, data in enumerate(data_loaders[phase]):
                    impressions, target = data

                    target = target.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        predicted = network(
                            impressions=impressions,
                            target=target,
                        ).float()

                        if phase == 'train':
                            loss = loss_function(predicted, target)
                            loss.backward()
                            optimizer.step()
                            losses[idx] = loss.item()

                    if do_validation:
                        predicted = F.sigmoid(predicted)

                        predicted = predicted \
                            .detach().cpu().numpy().flatten()
                        target = target \
                            .detach().cpu().numpy().flatten()

                        cur_prediction['predicted'][prediction_ptr: prediction_ptr + len(predicted)] = predicted
                        cur_prediction['target'][prediction_ptr: prediction_ptr + len(target)] = target
                        cur_prediction['baseline'][prediction_ptr: prediction_ptr + len(target)] = F.cosine_similarity(
                            item1,
                            item2
                        ).cpu().numpy()

                        prediction_ptr += len(target)
                    bar.update(idx)
            if do_validation:
                score = f1_score(
                    cur_prediction['predicted'].values > 0.5,
                    cur_prediction['target'].values
                )

                baseline_pred = ((1 + cur_prediction['baseline'].values)/2) > 0.5
                baseline = f1_score(
                    baseline_pred > 0.5,
                    cur_prediction['target'].values
                )

                print(phase, " Score: ", score, "Baseline:", baseline)
                lr_scheduler.step(score)
                if phase == 'val':
                    if best_score_so_far is None or score > best_score_so_far:
                        best_score_so_far = score
                        best_baseline_so_far = baseline
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
            if phase == 'train':
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


if __name__ == '__main__':
    print("Train Ranking")
    best_score_so_far = 0
    best_config = None
    best_path = None
    best_baseline_so_far = 0
    for config in ranking_configs:
        print("Train config ", config['name'])
        current_score, current_baseline, current_path = train(config)
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


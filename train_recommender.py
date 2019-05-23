import os
import random
import shutil

import progressbar
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from create_submission import create_submission
from recommender_configs import recommender_configs, prepare_config
from network.recommender_network import RecommenderNetwork
from dataset.recsys_dataset import RecSysDataset, RecSysData

import numpy as np

from utility.helpers import get_string_timestamp
from utility.prediction import Prediction

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')

MODEL_NAME = get_string_timestamp()
MODEL_BASE_PATH = os.path.join(DATA_PATH, 'model')
MODEL_PATH = os.path.join(DATA_PATH, 'model', MODEL_NAME + ".pth")


DEBUG = False

rec_sys_data = None


def get_rec_sys_data():
    global rec_sys_data

    if rec_sys_data is None:
        rec_sys_data = RecSysData(mode='train')

    return rec_sys_data


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

    train_dataset = RecSysDataset(
        rec_sys_data=get_rec_sys_data(),
        split=0.7,
        before=True,
        include_impressions=False,
        train_mode=True
    )
    train_val_dataset = RecSysDataset(
        rec_sys_data=get_rec_sys_data(),
        split=0.7,
        before=True,
        include_impressions=True
    )
    val_dataset = RecSysDataset(
        rec_sys_data=get_rec_sys_data(),
        split=0.7,
        before=False,
        include_impressions=True
    )

    if 'config_constraint' in config:
        max_dataset_size = config['config_constraint']['max_dataset_size']
        if len(train_dataset) + len(val_dataset) > max_dataset_size:
            print("Skip config, constraints not satisfied")
            return 0, ''

    network = RecommenderNetwork(
        config=config,
        item_size=train_dataset.item_size,
        target_item_size=train_dataset.target_item_size,
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=reduce_factor, patience=reduce_patience)

    start_epoch = 0
    best_score_so_far = None

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
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            collate_fn=train_dataset.collator),
        "train_val": DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                collate_fn=train_val_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                          collate_fn=val_dataset.collator),
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

        for phase in cur_phase:  # 'train_val',
            cur_dataset = datasets[phase]
            cur_prediction = Prediction(
                dataset=cur_dataset,
                device=device,
            )
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
                        cur_prediction.add_predictions(
                            ids=ids,
                            impression_ids=impression_ids,
                            item_impressions=item_impressions,
                            item_scores=item_scores
                        )
                    bar.update(idx)
            if do_validation:
                score = cur_prediction.get_score()

                print(phase, " Score: ", score)
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

    return best_score_so_far, target_path


if __name__ == '__main__':
    best_score_so_far = 0
    best_config = None
    best_path = None
    for config in recommender_configs:
        print("Train config ", config['name'])
        current_score, current_path = train(config)
        if current_score > best_score_so_far:
            best_score_so_far = current_score
            best_config = config
            best_path = current_path

    print("-" * 30)
    print("Best Config: ", str(best_config))
    print("Best Score: ", str(best_score_so_far))
    print("Best Path: ", str(best_path))

    create_submission(
        path=best_path
    )


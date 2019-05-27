import os
import random
import shutil

import progressbar
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from create_submission import create_submission
from network.impression_network import ImpressionRankNetwork
from recommender_configs import recommender_configs, prepare_config
from network.recommender_network import RecommenderNetwork
from dataset.recsys_dataset import RecSysDataset, RecSysData
from utility.split_utility import RandomSampleStrategy, AllSamplesExceptStrategy, AllSamplesIncludeStrategy

import numpy as np

from utility.helpers import get_string_timestamp
from utility.prediction import Prediction

DATA_PATH = './data/'
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')

MODEL_NAME = get_string_timestamp()
MODEL_BASE_PATH = os.path.join(DATA_PATH, 'model')
MODEL_PATH = os.path.join(DATA_PATH, 'model', MODEL_NAME + ".pth")


DEBUG = False

DO_SUBMISSION = False

rec_sys_data = None


def get_rec_sys_data(size):
    global rec_sys_data

    if rec_sys_data is not None and rec_sys_data.size != size:
        rec_sys_data = None

    if rec_sys_data is None:
        rec_sys_data = RecSysData(mode='train', size=size)

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
    use_cosine_similarity = config.get('use_cosine_similarity')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data = get_rec_sys_data(
        size=config.get('dataset_size')
    )

    train_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=RandomSampleStrategy(split=0.7),
        include_impressions=False,
        train_mode=True
    )
    train_rank_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=AllSamplesIncludeStrategy(include=train_dataset.session_ids),
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

    impression_rank_network = ImpressionRankNetwork(
        config=config,
        item_size=train_dataset.target_item_size,
        device=device
    )

    recommend_network = RecommenderNetwork(
        config=config,
        item_size=train_dataset.item_size,
        target_item_size=train_dataset.target_item_size,
    )
    loss_function = nn.MSELoss()
    impression_loss_function = nn.CrossEntropyLoss()
    rc_optimizer = optim.Adam(recommend_network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ir_optimizer = optim.Adam(impression_rank_network.parameters(), lr=0.01, weight_decay=0.0)
    rc_lr_scheduler = ReduceLROnPlateau(rc_optimizer, mode='max', factor=reduce_factor, patience=reduce_patience)
    ir_lr_scheduler = ReduceLROnPlateau(ir_optimizer, mode='max', factor=reduce_factor, patience=reduce_patience)

    start_epoch = 0
    best_score_so_far = None

    if state:
        rc_optimizer.load_state_dict(state['rc_optimizer_state_dict'])
        ir_optimizer.load_state_dict(state['ir_optimizer_state_dict'])
        ir_lr_scheduler.load_state_dict(state['ir_optimizer_state_dict'])
        rc_lr_scheduler.load_state_dict(state['rc_optimizer_state_dict'])
        recommend_network.load_state_dict(state['network_state_dict'])
        start_epoch = state['epoch']
        best_score_so_far = state['best_score_so_far']
        impression_rank_network = state['impression_rank_network']

    datasets = {
        "train": train_dataset,
        "train_rank": train_rank_dataset,
        "train_val": train_dataset,
        "val": val_dataset,
    }

    data_loaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                            collate_fn=train_dataset.collator),
        "train_rank": DataLoader(train_rank_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                            collate_fn=train_rank_dataset.collator),
        "train_val": DataLoader(train_val_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                collate_fn=train_val_dataset.collator),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                          collate_fn=val_dataset.collator),
    }

    sizes = {
        "train": int(len(train_dataset) / batch_size) + 1,
        "train_rank": int(len(train_dataset) / batch_size) + 1,
        "train_val": int(len(train_dataset) / batch_size) + 1,
        "val": int(len(val_dataset) / batch_size) + 1,
    }

    losses = np.zeros(int(len(train_dataset) / batch_size + 1.0))

    recommend_network = recommend_network.to(device)
    impression_rank_network = impression_rank_network.to(device)

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
                add_reference=False,
                use_cosine_similarity=use_cosine_similarity,
            )
            do_validation = phase != 'train' and phase != 'train_rank'
            if phase == 'train':
                recommend_network.train()
                impression_rank_network.eval()
            elif phase == 'train_rank':
                recommend_network.eval()
                impression_rank_network.train()
            else:
                recommend_network.eval()
                impression_rank_network.eval()

            losses.fill(0)
            with progressbar.ProgressBar(max_value=sizes[phase], redirect_stdout=True) as bar:
                for idx, data in enumerate(data_loaders[phase]):
                    if phase != 'train':
                        sessions, session_lengths, session_targets, item_impressions, impression_ids, target_index, ids = data
                    else:
                        sessions, session_lengths, session_targets = data
                        impression_ids = None
                        item_impressions = None
                        ids = None
                        target_index = None


                    sessions = sessions.to(device)
                    session_targets = session_targets.to(device)

                    if phase == 'train':
                        rc_optimizer.zero_grad()
                        with torch.set_grad_enabled(True):
                            item_scores = recommend_network(sessions, session_lengths).float()
                            loss = loss_function(item_scores, session_targets)
                            loss.backward()
                            rc_optimizer.step()
                            losses[idx] = loss.item()
                    elif phase == 'train_rank':
                        if use_cosine_similarity:
                            print("Weird settings train_rank vs cosine_similarity")
                        ir_optimizer.zero_grad()
                        item_scores = recommend_network(sessions, session_lengths).float()

                        with torch.set_grad_enabled(True):
                            predicted = impression_rank_network(item_impressions, item_scores)
                            target_index = torch.stack(target_index, dim=0).to(device)
                            loss = impression_loss_function(predicted, target_index)
                            loss.backward()
                            ir_optimizer.step()
                            losses[idx] = loss.item()
                    else:
                        with torch.set_grad_enabled(False):
                            item_scores = recommend_network(sessions, session_lengths).float()
                            selected_impression = impression_rank_network(item_impressions, item_scores)

                            cur_prediction.add_predictions(
                                ids=ids,
                                impression_ids=impression_ids,
                                item_impressions=item_impressions,
                                item_scores=item_scores,
                                selected_impression=selected_impression,
                            )
                    bar.update(idx)
            if do_validation:
                score = cur_prediction.get_score()

                print(phase, " Score: ", score)
                rc_lr_scheduler.step(score)
                ir_lr_scheduler.step(score)
                if phase == 'val':
                    if best_score_so_far is None or score > best_score_so_far:
                        best_score_so_far = score
                        torch.save({
                            'epoch': epoch,
                            'best_score_so_far': best_score_so_far,
                            'rc_optimizer_state_dict': rc_optimizer.state_dict(),
                            'ir_optimizer_state_dict': ir_optimizer.state_dict(),
                            'network_state_dict': recommend_network.state_dict(),
                            'impression_rank_network': impression_rank_network.state_dict(),
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
    if DO_SUBMISSION:
        create_submission(
            path=best_path
        )


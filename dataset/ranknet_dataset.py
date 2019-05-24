import os
import pickle

import torch
from torch.utils.data import Dataset

from create_submission import do_prediction
from dataset.recsys_dataset import RecSysDataset, RandomSampleStrategy
from train_recommender import get_rec_sys_data
import pandas as pd


def get_ranknet_data(model_path):
    from train_recommender import DATA_PATH

    file = os.path.basename(model_path)

    pickle_path = os.path.join(DATA_PATH, 'rank_data', file)

    if os.path.exists(pickle_path):
        result = pickle.load(open(pickle_path, "rb"))
        return result
    else:
        predictions = do_prediction(
            torch.load(model_path),
            RecSysDataset(
                rec_sys_data=get_rec_sys_data(),
                split_strategy=RandomSampleStrategy(split=1.0),
                include_impressions=True,
            ),
            add_reference=True
        ).predictions

        pickle.dump(predictions, open(pickle_path, "wb"), protocol=4)


class RankNetData(object):
    def __init__(self, model_path):
        self.session_rankings_df = get_ranknet_data(
            model_path=model_path
        )


class RankNetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

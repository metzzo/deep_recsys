import os
import pickle
from random import randint, random

import torch
from torch.utils.data import Dataset

from create_submission import do_prediction
from dataset.load_items import load_items
from dataset.recsys_dataset import RecSysDataset
from utility.split_utility import RandomSampleStrategy
from train_recommender import get_rec_sys_data, RAW_DATA_PATH
import pandas as pd
import numpy as np


def get_ranknet_data(model_path, session_data):
    from train_recommender import DATA_PATH

    file = os.path.basename(model_path)

    pickle_path = os.path.join(DATA_PATH, 'rank_data', file)
    shared_pickle_path = os.path.join(DATA_PATH, 'shared.p')
    shared = pickle.load(open(shared_pickle_path, "rb"))

    if os.path.exists(pickle_path):
        predictions = pickle.load(open(pickle_path, "rb"))
        print("loaded ranknet data from pickle")
    else:
        """
        predictions = do_prediction(
            torch.load(model_path),
            RecSysDataset(
                rec_sys_data=get_rec_sys_data(),
                split_strategy=RandomSampleStrategy(split=1.0),
                include_impressions=True,
            ),
            add_reference=True
        ).predictions
        """
        print("Read CSV")
        raw_df = pd.read_csv(
            os.path.join(RAW_DATA_PATH, 'train.csv'),
            sep=',',
           # nrows=1000
        )
        raw_df = raw_df[raw_df['action_type'] == 'clickout item']
        raw_df = raw_df[['session_id',  'reference', 'impressions']]
        #raw_df['session_id'] = shared['label_encoders']['session_id'].transform(raw_df['session_id'].values)
        raw_df['reference'] = raw_df['reference'].astype(int)
        print("Prepare raw_df")
        item_impressions_df = raw_df['impressions'].str \
            .split("|", expand=True) \
            .apply(pd.to_numeric)
        print("count")
        count = 25 - item_impressions_df.isnull().sum(axis=1)
        count = pd.DataFrame({
            'count': count
        })
        print("fillna & cast to int")
        item_impressions_df.fillna(-1, inplace=True)
        item_impressions_df = item_impressions_df.astype(int)
        print("COncat")
        predictions = pd.concat([raw_df[['reference']], item_impressions_df, count], axis=1)
        print("save")
        pickle.dump(predictions, open(pickle_path, "wb"), protocol=4)
    return predictions


class RankNetData(object):
    def __init__(self, model_path):
        #self.session_data = get_rec_sys_data()
        self.session_rankings_df = get_ranknet_data(
            model_path=model_path,
            session_data=None #self.session_data,
        )
        self.item_df, self.item_vectorizer = load_items()


class RankNetDataset(Dataset):
    def __init__(self, data: RankNetData, split_strategy):
        self.data = data
        self.item_feature_size = len(self.data.item_vectorizer.get_feature_names())
        self.null_item = np.zeros(self.item_feature_size).astype(float)

        self.session_ranking_indices = split_strategy.sample(
            array=self.data.session_rankings_df.index
        )

    def __getitem__(self, index):
        row = self.data.session_rankings_df.loc[
            self.session_ranking_indices[index]
        ]
        """
        session_indices = self.data.session_data.groups[
            row['session_id']
        ]
        session = self.data.session_data.session_df.loc[
            session_indices
        ]
        clickout = session[-1]

        prices = clickout['prices'].str.split('|')
        """

        selected_row = randint(0, row['count'] - 1)

        true_item_id = row['reference']
        predicted_item_id = row[selected_row]

        try:
            true_item = self.data.item_df.loc[true_item_id]
        except KeyError:
            true_item = self.null_item
            true_item_id = -1.0

        if random() > 0.5:
            predicted_item_id = true_item_id
            predicted_item = true_item
        else:
            try:
                predicted_item = self.data.item_df.loc[predicted_item_id]
            except KeyError:
                predicted_item = self.null_item
                predicted_item_id = -1.0

        if true_item_id == predicted_item_id:
            true_value = 1.0  # 1.0 - float(selected_row)/24.0
        else:
            true_value = 0.0  # selected_row / 24.0

        if random() > 0.5:
            swap = true_item
            true_item = predicted_item
            predicted_item = swap

        return torch.tensor(true_item).float(), \
               torch.tensor(predicted_item).float(), \
               torch.tensor([true_value]).float()

    def __len__(self):
        return len(self.session_ranking_indices)

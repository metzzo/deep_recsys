import collections
import os
import pickle
import random

import torch

from create_submission import do_prediction
from dataset.recsys_dataset import RecSysDataset
from train_recommender import get_rec_sys_data, DATA_PATH
from utility.prediction import Prediction

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

from utility.split_utility import RandomSampleStrategy, AllSamplesExceptStrategy

RECOMMENDER_MODEL_PATH = './data/model/2019_05_28_02_14_02_0.59.pth'
RANKING_MODEL_PATH = './data/ranking_model/2019_05_28_09_25_11_63.pth'

def session_length_performance(prediction: Prediction, dataset: RecSysDataset):
    _, score = prediction.get_score()
    score.reset_index(level=1, inplace=True)
    score.set_index(['session_id'], inplace=True)

    performances = {}
    for index in range(len(dataset)):
        indices = np.array(dataset.rec_sys_data.groups[
            dataset.session_ids[index]
        ])
        session = dataset.rec_sys_data.session_df.loc[
            indices
        ]
        length = len(session)
        session_id = session['session_id'].iloc[0]
        if length not in performances:
            performances[length] = []
        performances[length].append(score.loc[session_id]['score'])

    for length in performances.keys():
        arr = np.array(performances[length])
        performances[length] = (arr.mean(), len(arr))

    return performances


def draw_plot(performances, dataset: RecSysDataset):
    new_performances = {}
    upper_limit = []

    total_count = 0

    for key, value in performances.items():
        mean, count = value
        if key >= 200:
            upper_limit.append(mean * count)
            total_count += count
        else:
            new_performances[key] = mean
    new_performances[200] = np.array(upper_limit).sum() / total_count
    performances = new_performances
    performances = collections.OrderedDict(sorted(performances.items()))

    """
    num_buckets = 100
    num_per_bucket = len(dataset) / num_buckets

    cur_bucket = []
    for length, performance in performances.items():
    """

    x = list(performances.keys())
    y = list(performances.values())

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='Session Length', ylabel='MRR',
           title='Session Length vs MRR')
    ax.grid()

    fig.savefig("session_length_performance.png")
    plt.show()


if __name__ == '__main__':
    random.seed(42)

    recommender_state = torch.load(RECOMMENDER_MODEL_PATH)
    ranking_state = torch.load(RANKING_MODEL_PATH)

    data = get_rec_sys_data(
        size=None
    )

    train_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=RandomSampleStrategy(split=0.7),
        include_impressions=True,
        train_mode=False
    )
    val_dataset = RecSysDataset(
        rec_sys_data=data,
        split_strategy=AllSamplesExceptStrategy(exclude=train_dataset.session_ids),
        include_impressions=True
    )

    prediction = do_prediction(
        recommender_state=recommender_state,
        ranking_state=ranking_state,
        dataset=val_dataset,
    )

    performance = session_length_performance(prediction, val_dataset)
    pickle.dump(prediction, open(os.path.join(DATA_PATH, 'visualization_data.pickle'), "wb"), protocol=4)
    draw_plot(performances=performance, dataset=val_dataset)

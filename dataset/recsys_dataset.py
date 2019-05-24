from collections import namedtuple
from functools import partial
from random import randint, random

from torch.utils.data.dataset import Dataset

import pandas as pd
import numpy as np
import torch

from dataset.load_items import load_items
from dataset.load_sessions import load_train_sessions, load_test_sessions
from utility.score import SUBM_INDICES

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

RecSysSampleWithImpressions = namedtuple('RecSysSampleWithImpressions', [
    'sessions',
    'session_lengths',
    'session_targets',
    'impressions',
    'impression_ids',
    'ids',
], verbose=False)

RecSysSample = namedtuple('RecSysSample', [
    'sessions',
    'session_lengths',
    'session_targets',
], verbose=False)


def _internal_collator(data, session_targets, item_size):
    session_targets = torch.stack(session_targets, 0)

    lengths = [len(cap) for cap in data]
    sessions = torch.zeros(len(data), max(lengths), item_size).float()
    for i, cap in enumerate(data):
        end = lengths[i]
        sessions[i, :end] = cap[:end]

    return sessions, torch.tensor(lengths), session_targets


def collator_with_impressions(items, item_size):
    # TODO: adapt if labels are missing
    items.sort(key=lambda x: len(x[0]), reverse=True)

    data, impressions, impression_ids, labels, ids = zip(*items)

    sessions, session_lengths, session_targets = _internal_collator(data, labels, item_size)

    return RecSysSampleWithImpressions(
        sessions=sessions,
        session_lengths=session_lengths,
        session_targets=session_targets,
        impressions=impressions,
        impression_ids=impression_ids,
        ids=ids,
    )


def collator_without_impressions(items, item_size):
    items.sort(key=lambda x: len(x[0]), reverse=True)
    # TODO: adapt if labels are missing
    data, labels = zip(*items)
    sessions, session_lengths, session_targets = _internal_collator(data, labels, item_size)
    return RecSysSample(
        sessions=sessions,
        session_lengths=session_lengths,
        session_targets=session_targets,
    )


class RecSysData(object):
    def __init__(self, mode):
        self.item_df, self.item_vectorizer = load_items()

        if mode == 'train':
            load_func = load_train_sessions
        elif mode == 'test':
            load_func = load_test_sessions
        else:
            raise NotImplementedError()

        session_data = load_func(
            item_df=self.item_df
        )

        self.session_df = session_data['session']
        self.grouped = session_data['grouped']
        self.relevant_session_ids = session_data['relevant_session_ids']
        self.item_properties = session_data['item_properties']
        self.session_hot_encoders = session_data['hot_encoders']
        self.session_label_encoders = session_data['label_encoders']

        self.groups = self.grouped.groups


class SplitStrategy(object):
    def sample(self, array):
        raise NotImplementedError()


class RandomSampleStrategy(SplitStrategy):
    def __init__(self, split):
        self.split = split

    def sample(self, array):
        samples_drawn = int(len(array) * self.split)
        import random
        indices = sorted(random.sample(range(len(array)), samples_drawn))
        return array[indices]


class AllSamplesExceptStrategy(SplitStrategy):
    def __init__(self, exclude):
        self.exclude = set(exclude)

    def sample(self, array):
        return np.array([x for x in array if x not in set(self.exclude)])


class RecSysDataset(Dataset):
    def __init__(self, rec_sys_data, split_strategy, include_impressions, train_mode=False):
        self.rec_sys_data = rec_sys_data
        self.include_impressions = include_impressions
        item_feature_size = len(self.rec_sys_data.item_vectorizer.get_feature_names())
        # size of item vector
        # + action_type one hot encoding
        self.item_size = item_feature_size + 6
        self.target_item_size = item_feature_size
        self.train_mode = train_mode

        sid = self.rec_sys_data.relevant_session_ids
        self.session_ids = split_strategy.sample(array=sid)

        self.empty_array = np.array(0)

    def __getitem__(self, index):
        indices = self.rec_sys_data.groups[
            self.session_ids[index]
        ]

        # augment data by making this session shorter
        if self.train_mode and len(indices) > 3:
            start_pos = randint(0, len(indices) - 3)
            end_pos = randint(start_pos + 2, len(indices) - 1)
            indices = indices.to_numpy()[start_pos:end_pos]

        if self.train_mode and random() > 0.5:
            indices = np.flip(indices)

        session = self.rec_sys_data.session_df.loc[
            indices
        ]

        target_properties = self.rec_sys_data.item_properties.loc[
            indices[-1]
        ]

        item_properties = np.array(self.rec_sys_data.item_properties.loc[
            indices[:-1]
        ])

        action_type_encoder = self.rec_sys_data.session_hot_encoders['action_type']
        session_action_type = session['action_type'][:-1]
        action_type = action_type_encoder.transform(np.array(session_action_type).reshape(-1, 1)).toarray()

        result = [np.hstack([
            item_properties,
            action_type,
#            session['delta_time'].values.reshape(-1, 1)[:-1],
#            session['step'].values.reshape(-1, 1)[:-1],
        ])]
        simple_result = []
        if self.include_impressions:
            last_row = session.iloc[-1]

            last_impressions = last_row['impressions']
            item_impressions_df = pd.DataFrame(map(int, last_impressions.split('|')), columns=['impression'])

            item_impressions_id = np.array(item_impressions_df['impression'])
            item_impressions = None

            create_impression = False
            try:
                item_impressions = np.array(self.rec_sys_data.item_df.loc[item_impressions_id])
            except KeyError:
                create_impression = True

            if create_impression or len(item_impressions) == 0:
                print("Create impressions ", self.session_ids[index], session['user_id'])
                # TODO: do something more clever, e.g.: by sorting by popularity
                items = self.rec_sys_data.item_df.sample(25)
                item_impressions_id = items.index.values
                item_impressions = np.array(items)

            result += [item_impressions, item_impressions_id]
            simple_result += [last_row[SUBM_INDICES]]

        if target_properties is not None:
            result = result + [target_properties]

        return list(map(torch.tensor, result)) + simple_result

    def __len__(self):
        return len(self.session_ids)

    @property
    def collator(self):
        if self.include_impressions:
            col = collator_with_impressions
        else:
            col = collator_without_impressions

        return partial(col, item_size=self.item_size)

    def get_submission(self):
        grouped = self.rec_sys_data.grouped
        lasts = grouped.tail(1)

        lasts_selected = lasts[SUBM_INDICES]

        lasts_selected = lasts_selected.assign(
            reference=lasts['reference'],
            impressions=lasts['impressions'],
            prices=lasts['prices']
        )

        lasts_selected.set_index(SUBM_INDICES, inplace=True)
        return lasts_selected


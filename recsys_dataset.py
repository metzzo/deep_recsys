from collections import namedtuple
from functools import partial
from random import randint, shuffle, random

from torch.utils.data.dataset import Dataset

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from functions import SUBM_INDICES

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


def split_row(df, column, sep):
    s = df[column].str.split(sep, expand=True)
    s = s.rename(lambda x: '{}_{}'.format(column, x), axis=1)
    df = df.drop(column, axis=1)

    result = pd.concat([
        df,
        s
    ], axis=1)
    del df
    del s
    return result


def hot_encode_labels(df, columns):
    label_encoders = {col: LabelEncoder() for col in columns}
    hot_encoders = {col: OneHotEncoder() for col in columns}
    for col, label_encoder in label_encoders.items():
        print("Hot encode label ", col)
        df[col].fillna('', inplace=True)
        label_encoded = label_encoder.fit_transform(df[col])
        df[col] = label_encoded
        hot_encoders[col].fit(label_encoded.reshape(-1, 1))
        del label_encoded

    return hot_encoders, label_encoders


def prepare_reference(df: pd.DataFrame, action_types):
    for action_type in action_types:
        print("Prepare reference ", action_type)
        df = df[~df['action_type'].isin(action_types)]
        """
        query = df['action_type'] == action_type
        old_df = df
        df = df.assign(**{
            action_type: df[query]['reference']
        })
        del old_df
        # TODO: stranger danger - pandas warning SettingWithCopyWarning
        df['reference'][query] = ''
        """
    df.reset_index(inplace=True, drop=True)
    return df


def load_sessions(item_df):
    from main import RAW_DATA_PATH, DATA_PATH, DEBUG

    train_data_path = os.path.join(RAW_DATA_PATH, 'train.csv')
    pickle_path = os.path.join(DATA_PATH, 'sessions.p')
    pickle_path_item_properties = os.path.join(DATA_PATH, 'item_properties_sessions.p')
    #pickle_path_target_item_properties = os.path.join(DATA_PATH, 'target_item_properties_sessions.p')

    if os.path.exists(pickle_path):
        result = pickle.load(open(pickle_path, "rb"))
        item_properties = pickle.load(open(pickle_path_item_properties, "rb"))
        #target_item_properties = pickle.load(open(pickle_path_target_item_properties, "rb"))
        return result + (item_properties, ) #target_item_properties)
    else:
        print("load csv")
        raw_df = pd.read_csv(
            train_data_path,
            sep=',',
            nrows=1000 if DEBUG else 10000 #TODO: change back to None for full dataset
        )
        raw_df = split_row(raw_df, column='city', sep=',')

        # extract search_for poi impression into own column
        prepare_action_types = [
            'search for poi',
            'change of sort order',
            'filter selection',
            'search for destination',
        ]
        raw_df = prepare_reference(raw_df, prepare_action_types)

        # encode labels
        encoders, label_encoders = hot_encode_labels(raw_df, columns=[
            'session_id',
            'action_type',
            'city_0',
            'city_1',
            'platform',
            'device',
        ]) # + prepare_action_types)
        print("Remove invalid references...")
        raw_df['reference'] = pd.to_numeric(raw_df['reference'], errors='coerce').fillna(-1).astype(int)

        clickout_type = label_encoders['action_type'].transform(['clickout item'])[0]
        print("filter references which do not exist")

        referencing_action_type = label_encoders['action_type'].transform([
            'clickout item',
            'interaction item rating',
            'interaction item info',
            'interaction item image',
            'interaction item deals',
            'search for item',
        ])
        item_properties = item_df.loc[raw_df['reference']]
        item_properties.reset_index(inplace=True, drop=True)
        raw_df = raw_df[~((item_properties[0].isnull()) & (raw_df['action_type'].isin(referencing_action_type)))]
        raw_df.reset_index(inplace=True)

        print("filter session_ids where the last entry is not a clickout")
        next_session_id = raw_df["session_id"].shift(-1)
        to_delete = raw_df[(raw_df['session_id'] != next_session_id) & (raw_df['action_type'] != clickout_type)]['session_id']
        raw_df = raw_df[~raw_df['session_id'].isin(to_delete)]
        raw_df.reset_index(inplace=True, drop=True)

        print("prepare reference item_ids")
        item_properties = item_df.loc[raw_df['reference']]
        item_properties.reset_index(inplace=True, drop=True)
        item_properties.fillna(0.0, inplace=True)

        print("get delta times")
        last_timestamp = raw_df["timestamp"].shift(-1)
        next_session_id = raw_df["session_id"].shift(-1)
        raw_df['delta_time'] = last_timestamp - raw_df["timestamp"]
        raw_df.loc[(raw_df['session_id'] != next_session_id), 'delta_time'] = 0

        delta_time_scaler = StandardScaler()
        scaled = raw_df[['delta_time', 'step']].values.reshape(-1, 1)
        scaled = delta_time_scaler.fit_transform(scaled)
        raw_df['delta_time'], raw_df['step'] = scaled[:, 0], scaled[:, 1]
        encoders['delta_time_step'] = delta_time_scaler

        """
        print("split into target_item_properties vs item_properties")
        next_session_id = raw_df["session_id"].shift(-1)
        target_item_properties = item_properties[
            (raw_df['session_id'] != next_session_id)
        ]
        item_properties[
            (raw_df['session_id'] != next_session_id)
        ] = 0.0
        """

        raw_df.drop([
            'index',
        ], axis=1, inplace=True)

        print("groupby")
        grouped = raw_df.groupby(by='session_id')
        print("extract session ids")
        session_ids = np.array(list(grouped.groups.keys()))
        np.random.shuffle(session_ids)

        train_sessions = grouped[['step']].count()

        train_session_ids = np.array(list(train_sessions[train_sessions['step'] > 1].index))
        print("shuffle train session")
        np.random.shuffle(train_session_ids)

        print("shuffle session")
        print("write to disk")
        result = raw_df, grouped, encoders, session_ids, train_session_ids
        pickle.dump(result, open(pickle_path, "wb"))
        pickle.dump(item_properties, open(pickle_path_item_properties, "wb"), protocol=4)
        #pickle.dump(target_item_properties, open(pickle_path_target_item_properties, "wb"), protocol=4)

        result += (item_properties, )# target_item_properties)

    return result


def load_items():
    from main import RAW_DATA_PATH, DATA_PATH, DEBUG

    pickle_path = os.path.join(DATA_PATH, 'items.p')

    if os.path.exists(pickle_path):
        result = pickle.load(open(pickle_path, "rb"))
    else:
        items_path = os.path.join(RAW_DATA_PATH, 'item_metadata.csv')
        raw_df = pd.read_csv(
            items_path,
            sep=',',
            #nrows=1000 if DEBUG else None
        )
        tfidf_vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            #min_df=0.01,
            binary=True,
            use_idf=False,
            norm=None
        )
        item_properties = tfidf_vectorizer.fit_transform(raw_df['properties'])
        item_properties = pd.DataFrame(item_properties.toarray())
        item_properties.set_index(raw_df['item_id'], inplace=True)

        result = item_properties, tfidf_vectorizer
        pickle.dump(result, open(pickle_path, "wb"))
    return result





class RecSysData(object):
    def __init__(self):
        self.item_df, self.item_vectorizer = load_items()
        # , self.target_item_properties
        self.session_df, self.grouped, self.session_encoders, self.session_ids, self.train_session_ids, self.item_properties = load_sessions(
            item_df=self.item_df
        )
        self.groups = self.grouped.groups

rec_sys_data = None


class RecSysDataset(Dataset):
    def __init__(self, split, before, include_impressions, train_mode=False):
        global rec_sys_data
        if rec_sys_data is None:
            rec_sys_data = RecSysData()
        self.rec_sys_data = rec_sys_data
        self.include_impressions = include_impressions
        item_feature_size = len(self.rec_sys_data.item_vectorizer.get_feature_names())
        # size of item vector
        # + action_type one hot encoding
        self.item_size = item_feature_size + 6
        self.target_item_size = item_feature_size
        self.train_mode = train_mode

        # TODO: neural network does not support sessions with length 1
        sid = self.rec_sys_data.train_session_ids # if not train_mode else self.rec_sys_data.train_session_ids
        split_index = int(len(sid) * split)
        if before:
            self.session_ids = sid[:split_index]
        else:
            self.session_ids = sid[split_index:]

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

        action_type_encoder = self.rec_sys_data.session_encoders['action_type']
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
            item_impressions = np.array(self.rec_sys_data.item_df.loc[item_impressions_id])
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


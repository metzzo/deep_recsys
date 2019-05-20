from functools import partial

from torch.utils.data.dataset import Dataset

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

from functions import SUBM_INDICES

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


def collator(items, item_size):
    # TODO: adapt if labels are missing
    data, impressions, impression_ids, labels = zip(*items)

    labels = torch.stack(labels, 0)

    lengths = [len(cap) for cap in data]
    padded_items = torch.zeros(len(data), max(lengths), item_size).float()
    for i, cap in enumerate(data):
        end = lengths[i]
        padded_items[i, :end] = cap[:end]

    return padded_items, impressions, impression_ids, labels


def split_row(df, column, sep):
    s = df[column].str.split(sep, expand=True)
    s = s.rename(lambda x: '{}_{}'.format(column, x), axis=1)
    df = df.drop(column, axis=1)

    return pd.concat([
        df,
        s
    ], axis=1)


def hot_encode_labels(df, columns):
    label_encoders = {col: LabelEncoder() for col in columns}
    hot_encoders = {col: OneHotEncoder() for col in columns}
    for col, label_encoder in label_encoders.items():
        print("Hot encode label ", col)
        df[col].fillna('', inplace=True)
        label_encoded = label_encoder.fit_transform(df[col])
        df[col] = label_encoded
        hot_encoders[col].fit(label_encoded.reshape(-1, 1))

    return hot_encoders, label_encoders


def prepare_reference(df: pd.DataFrame, action_types):
    for action_type in action_types:
        print("Prepare reference ", action_type)
        search_for_poi_query = df['action_type'] == action_type
        df = df.assign(**{
            action_type: df[search_for_poi_query]['reference']
        })
        df['reference'][search_for_poi_query] = ''
    return df


def load_sessions(item_df):
    from main import RAW_DATA_PATH, DATA_PATH, DEBUG

    train_data_path = os.path.join(RAW_DATA_PATH, 'train.csv')
    pickle_path = os.path.join(DATA_PATH, 'sessions.p')

    if os.path.exists(pickle_path):
        result = pickle.load(open(pickle_path, "rb"))
    else:
        print("load csv")
        raw_df = pd.read_csv(
            train_data_path,
            sep=',',
            nrows=1000 if DEBUG else None
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
        ] + prepare_action_types)
        print("Prepare references...")
        raw_df['reference'] = pd.to_numeric(raw_df['reference'], errors='coerce').fillna(-1).astype(int)

        clickout_type = label_encoders['action_type'].transform(['clickout item'])[0]
        print("filter references which do not exist")
        join_item_df = item_df.copy()
        join_item_df['properties'] = ''
        joined_raw_df = pd.merge(
            raw_df[['reference']],
            join_item_df,
            left_on='reference',
            right_index=True,
            how='left',
        )
        referencing_action_type = label_encoders['action_type'].transform([
            'clickout item',
            'interaction item rating',
            'interaction item info',
            'interaction item image',
            'interaction item deals',
            'search for item',
        ])
        raw_df = raw_df[~((joined_raw_df['properties'].isnull()) & (raw_df['action_type'].isin(referencing_action_type)))]
        raw_df.reset_index(inplace=True)

        print("filter session_ids where the last entry is not a clickout")
        next_session_id = raw_df["session_id"].shift(-1)
        to_delete = raw_df[(raw_df['session_id'] != next_session_id) & (raw_df['action_type'] != clickout_type)]['session_id']
        raw_df = raw_df[~raw_df['session_id'].isin(to_delete)]
        raw_df.reset_index(inplace=True)
        print("groupby")
        grouped = raw_df.groupby(by='session_id')
        print("extract session ids")
        session_ids = np.array(list(grouped.groups.keys()))
        print("shuffle session")
        np.random.shuffle(session_ids)
        print("write to disk")
        result = raw_df, grouped, encoders, session_ids
        pickle.dump(result, open(pickle_path, "wb"))

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
            min_df=0.01
        )
        tfidf_vectorizer.fit(raw_df['properties'])

        raw_df.set_index('item_id', inplace=True)

        result = raw_df, tfidf_vectorizer
        pickle.dump(result, open(pickle_path, "wb"))
    return result


class RecSysData(object):
    def __init__(self):
        self.item_df, self.item_vectorizer = load_items()
        self.session_df, self.grouped, self.session_encoders, self.session_ids = load_sessions(
            item_df=self.item_df
        )
        self.groups = self.grouped.groups


rec_sys_data = None


class RecSysDataset(Dataset):
    def __init__(self, split, before):
        global rec_sys_data
        if rec_sys_data is None:
            rec_sys_data = RecSysData()
        self.rec_sys_data = rec_sys_data

        sid = self.rec_sys_data.session_ids
        split_index = int(len(sid) * split)
        if before:
            self.session_ids = sid[:split_index]
        else:
            self.session_ids = sid[split_index:]

    def __getitem__(self, index):
        session = self.rec_sys_data.session_df.iloc[
            self.rec_sys_data.groups[
                self.session_ids[index]
            ]
        ]

        merged_with_reference = pd.merge(
            session,
            self.rec_sys_data.item_df,
            left_on='reference',
            right_index=True,
            how='left',
        )

        item_vectorizer = self.rec_sys_data.item_vectorizer

        last_row = merged_with_reference.iloc[-1]
        last_reference = last_row['reference']
        last_impressions = last_row['impressions']

        target_properties = None
        if last_reference != -1:
            # this is our label
            merged_with_reference['properties'].iat[-1] = ''
            target_raw_properties = self.rec_sys_data.item_df.loc[last_reference]
            target_properties = np.array(item_vectorizer.transform(target_raw_properties).todense()).flatten()

        raw_properties = merged_with_reference['properties']
        raw_properties.fillna('', inplace=True)

        item_properties = item_vectorizer.transform(raw_properties).todense()

        item_impressions_df = pd.DataFrame(map(int, last_impressions.split('|')), columns=['impression'])
        item_impressions_df = pd.merge(
            item_impressions_df,
            self.rec_sys_data.item_df,
            left_on='impression',
            right_index=True,
            how='left',
        )
        item_impressions_id_df = np.array(item_impressions_df['impression'])
        impression_properties = item_impressions_df['properties']
        impression_properties.fillna('', inplace=True)
        item_impressions_df = np.array(item_vectorizer.transform(impression_properties).todense())

        result = [item_properties, item_impressions_df, item_impressions_id_df]
        if target_properties is not None:
            result = result + [target_properties]

        return list(map(torch.tensor, result))

    def __len__(self):
        return len(self.session_ids)

    @property
    def item_size(self):
        return len(self.rec_sys_data.item_vectorizer._tfidf.idf_)

    @property
    def collator(self):
        return partial(collator, item_size=self.item_size)

    def get_submission(self, predictions=None):
        grouped = self.rec_sys_data.grouped
        lasts = grouped.tail(1)

        lasts_selected = lasts[SUBM_INDICES]

        if predictions is None:
            lasts_selected = lasts_selected.assign(
                reference=lasts['reference'],
                impressions=lasts['impressions'],
                prices=lasts['prices']
            )
        else:
            lasts_selected = lasts_selected.assign(
                item_recommendations=predictions
            )

        lasts_selected.set_index(SUBM_INDICES, inplace=True)
        return lasts_selected


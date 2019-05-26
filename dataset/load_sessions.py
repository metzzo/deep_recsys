import os
import pickle

import numpy as np
import pandas as pd


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

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


def hot_encode_labels(df, columns, label_encoders=None, hot_encoders=None):
    label_encoders = label_encoders or {col: LabelEncoder() for col in columns}
    hot_encoders = hot_encoders or {col: OneHotEncoder() for col in columns}
    for col, label_encoder in label_encoders.items():
        print("Hot encode label ", col)
        df[col].fillna('', inplace=True)
        if hasattr(label_encoder, 'classes_'):
            label_encoded = label_encoder.transform(df[col])
        else:
            label_encoded = label_encoder.fit_transform(df[col])
            hot_encoders[col].fit(label_encoded.reshape(-1, 1))
        df[col] = label_encoded
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


def load_train_sessions(item_df, size):
    return base_load_sessions(
        item_df=item_df,
        csv_file='train.csv',
        # needed to fit the encoders etc properly, this is okay, because the last reference wont have a reference anyway
        # so it will get filtered anyway
        secondary_csv_file='test.csv',
        output_file='train_sessions_{}.p'.format(size),
        shared_output_file='shared.p',
        label_encoders=None,
        hot_encoders=None,
        nrows=size
    )


def load_test_sessions(item_df, size):
    from train_recommender import DATA_PATH
    shared_pickle_path = os.path.join(DATA_PATH, 'shared.p')
    shared_data = pickle.load(open(shared_pickle_path, "rb"))

    return base_load_sessions(
        item_df=item_df,
        csv_file='test.csv',
        secondary_csv_file=None,
        output_file='test_sessions_{}.p'.format(size),
        shared_output_file='shared.p',
        label_encoders=shared_data['label_encoders'],
        hot_encoders=shared_data['hot_encoders'],
        nrows=size
    )


def get_referencing_action_types(is_test):
    return [
        'interaction item rating',
        'interaction item info',
        'interaction item image',
        'interaction item deals',
        'search for item',
    ] + ([] if is_test else ['clickout item'])


def base_load_sessions(
        item_df,
        csv_file,
        secondary_csv_file,
        output_file,
        shared_output_file,
        label_encoders,
        hot_encoders,
        nrows,
):
    from train_recommender import RAW_DATA_PATH, DATA_PATH

    pickle_path = os.path.join(DATA_PATH, output_file)
    shared_pickle_path = os.path.join(DATA_PATH, shared_output_file)

    is_test = secondary_csv_file is None

    if os.path.exists(pickle_path):
        result = pickle.load(open(pickle_path, "rb"))
        result.update(
            pickle.load(open(shared_pickle_path, "rb"))
        )
        return result
    else:
        data_path = os.path.join(RAW_DATA_PATH, csv_file)
        print("load csv")
        raw_df = pd.read_csv(
            data_path,
            sep=',',
            nrows=nrows
        )

        if secondary_csv_file is not None:
            secondary_df = pd.read_csv(
                os.path.join(RAW_DATA_PATH, secondary_csv_file),
                sep=',',
                nrows=nrows
            )
            raw_df = pd.concat([
                raw_df,
                secondary_df
            ], ignore_index=True)
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
        hot_encoders, label_encoders = hot_encode_labels(raw_df, columns=[
            'session_id',
            'action_type',
            'city_0',
            'city_1',
            'platform',
            'device',
        ], label_encoders=label_encoders, hot_encoders=hot_encoders)
        print("Remove invalid references...")
        raw_df['reference'] = pd.to_numeric(raw_df['reference'], errors='coerce').fillna(-1).astype(int)

        clickout_type = label_encoders['action_type'].transform(['clickout item'])[0]
        print("filter references which do not exist")

        referencing_action_type = label_encoders[
            'action_type'
        ].transform(get_referencing_action_types(is_test))
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

        raw_df.drop([
            'index',
        ], axis=1, inplace=True)

        print("groupby")
        grouped = raw_df.groupby(by='session_id')

        print("shuffle & extract session ids")
        session_sizes = grouped[['step']].count()
        session_sizes = session_sizes[session_sizes['step'] > 1]
        noise = np.random.normal(0, 2, [len(session_sizes), 1]).astype(int) # locally shuffle by sorting by "noised length"
        noised_session_sizes = session_sizes + noise
        noised_session_sizes.sort_values(by='step', inplace=True)
        train_session_ids = np.array(list(noised_session_sizes.index))

        print("write to disk")

        result = {
            "session": raw_df,
            "relevant_session_ids": train_session_ids,
            "item_properties": item_properties,
            "grouped": grouped,
        }
        shared_result = {
            "hot_encoders": hot_encoders,
            "label_encoders": label_encoders,
        }

        pickle.dump(result, open(pickle_path, "wb"), protocol=4)
        pickle.dump(shared_result, open(shared_pickle_path, "wb"), protocol=4)
        result.update(
            shared_result
        )

    return result

import os

import torch
from verify_submission.verify_subm import main as verify_subm
from torch.utils.data import DataLoader

from utility.helpers import get_string_timestamp

MODEL_PATH = './data/model/2019_05_23_08_59_54.model'
SUBMISSION_BATCH_SIZE = 1024

import pandas as pd


def get_baseline():
    from train_recommender import RAW_DATA_PATH

    return pd.read_csv(os.path.join(RAW_DATA_PATH, 'submission_popular.csv'))


def create_submission(path):
    from dataset.recsys_dataset import RecSysData, RecSysDataset
    from network.recommender_network import RecommenderNetwork
    from recommender_configs import prepare_config
    from utility.prediction import Prediction
    from train_recommender import DATA_PATH

    print("Create Submission File for: ", path)

    state = torch.load(path)

    config = prepare_config(state.get('config'))
    network_state_dict = state.get('network_state_dict') or state

    test_rec_sys_data = RecSysData(mode='test')

    dataset = RecSysDataset(
        rec_sys_data=test_rec_sys_data,
        split=1.0,
        before=True,
        include_impressions=True,
        train_mode=False
    )

    network = RecommenderNetwork(
        config=config,
        item_size=dataset.item_size,
        target_item_size=dataset.target_item_size,
    )
    network.load_state_dict(network_state_dict)
    network.eval()

    data_loader = DataLoader(
        dataset,

        batch_size=SUBMISSION_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collator
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cur_prediction = Prediction(
        dataset=dataset,
        device=device,
    )

    network = network.to(device)
    for idx, data in enumerate(data_loader):
        sessions, session_lengths, _, item_impressions, impression_ids, ids = data
        sessions = sessions.to(device)

        item_scores = network(sessions, session_lengths).float()

        cur_prediction.add_predictions(
            ids=ids,
            impression_ids=impression_ids,
            item_impressions=item_impressions,
            item_scores=item_scores
        )

    # TODO: merge missing with baseline
    sess_ids = test_rec_sys_data.relevant_session_ids


    session_encoder = test_rec_sys_data.session_label_encoders['session_id']
    predictions = cur_prediction.predictions
    baseline = get_baseline()
    tested_session_ids = pd.DataFrame(
        session_encoder.inverse_transform(
            sess_ids
        ), columns=['session_id']
    )
    predictions['session_id'] = pd.Series(
        session_encoder.inverse_transform(
            predictions['session_id'].astype(int).values
        )
    )

    not_in_common = baseline[(~baseline['session_id'].isin(tested_session_ids['session_id']))]
    df_out = pd.concat([predictions, not_in_common], ignore_index=True)
    submission_path = os.path.join(DATA_PATH, 'submissions', '{}.csv'.format(get_string_timestamp()))  # TODO: add timestamp
    df_out.to_csv(submission_path, index=False)

    print("Do submission")

    verify_subm.callback(
        data_path='/',
        submission_file=os.path.abspath(submission_path),
        test_file=os.path.join(os.path.abspath(DATA_PATH), 'raw/test.csv')
    )

if __name__ == '__main__':
    create_submission(MODEL_PATH)

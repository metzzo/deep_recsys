import os

import baseline_algorithm.functions as f
import score_submission
import torch
import verify_submission
from torch.utils.data import DataLoader


MODEL_PATH = 'TODO'
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
    from train_recommender import DATA_PATH, get_rec_sys_data

    print("Create Submission File for: ", path)

    state = torch.load(path)

    config = prepare_config(state.get('config'))
    network_state_dict = state.get('network_state_dict') or state

    train_rec_sys_data = get_rec_sys_data()
    trained_session_ids = pd.DataFrame(
        train_rec_sys_data.session_label_encoders['session_id'].inverse_transform(
            train_rec_sys_data.session_df['session_id'].values
        ), columns=['session_id']
    )

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
    df_out = cur_prediction.predictions
    #df_out = get_baseline()
    #print(df_out)
    submission_path = os.path.join(DATA_PATH, 'submissions', 'submission.csv')  # TODO: add timestamp
    df_out.to_csv(submission_path, index=False)


if __name__ == '__main__':
    create_submission(MODEL_PATH)

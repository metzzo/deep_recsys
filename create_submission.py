import os
import pickle

import progressbar
import torch
from torch.utils.data import DataLoader
from verify_submission.verify_subm import main as verify_subm

from dataset.recsys_dataset import RandomSampleStrategy
from utility.helpers import get_string_timestamp

MODEL_PATH = './data/model/2019_05_23_22_09_43_0.58_full.pth'
SUBMISSION_BATCH_SIZE = 512

# these are missing in submission popular for some reason
MISSING_IN_POPULAR = ["f4e7686de96f6", "28aed6ff4f5cf", "1dc624176b25a", "3146144dd180c", "6395e6166436d", "5fd68f0c5a86f", "0c558de5076d3", "be040dde80c92", "692367c2a1e0e", "2fea505733796", "9f73d9d10c1e3", "f3fcfa14e3c1b", "51d1160090a5e", "9b38c0d8c7736", "f761e020a59f6", "a22b00a80121c", "a9641c5908155", "76b4000458cba", "c18c1f77a080b", "4df3c262efcf1", "2555579f7370b", "dd8438cfdd337", "3a4e144c181a5", "1e65b48e85ac6", "16cb478a81ef9", "263608e619c16", "095a051b27180", "e1f2b85aa8440", "ff30a418d7821"]

import pandas as pd


def get_baseline():
    from train_recommender import RAW_DATA_PATH

    return pd.read_csv(os.path.join(RAW_DATA_PATH, 'submission_popular.csv'))

def do_prediction(state, dataset, add_reference):
    from network.recommender_network import RecommenderNetwork
    from recommender_configs import prepare_config
    from utility.prediction import Prediction

    config = prepare_config(state.get('config'))
    network_state_dict = state.get('network_state_dict') or state

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
        add_reference=add_reference
    )

    network = network.to(device)
    print("Begin predicting...")
    with progressbar.ProgressBar(max_value=int(len(dataset) / SUBMISSION_BATCH_SIZE + 1), redirect_stdout=True) as bar:
        for idx, data in enumerate(data_loader):
            sessions, session_lengths, _, item_impressions, impression_ids, ids = data
            sessions = sessions.to(device)

            item_scores = network(sessions, session_lengths).float()

            cur_prediction.add_predictions(
                ids=ids,
                impression_ids=impression_ids,
                item_impressions=item_impressions,
                item_scores=item_scores,
            )
            bar.update(idx)

    return cur_prediction


def create_submission(path):
    from dataset.recsys_dataset import RecSysData, RecSysDataset
    from train_recommender import DATA_PATH

    print("Create Submission File for: ", path)

    state = torch.load(path)

    test_rec_sys_data = RecSysData(mode='test')

    dataset = RecSysDataset(
        rec_sys_data=test_rec_sys_data,
        split_strategy=RandomSampleStrategy(split=1.0),
        include_impressions=True,
        train_mode=False
    )

    cur_prediction = do_prediction(state, dataset, add_reference=False)
    pickle.dump(cur_prediction, open(os.path.join(DATA_PATH, 'prediction_dump.pickle'), "wb"), protocol=4)

    session_encoder = test_rec_sys_data.session_label_encoders['session_id']
    predictions = cur_prediction.predictions
    baseline = get_baseline()
    predictions['session_id'] = pd.Series(
        session_encoder.inverse_transform(
            predictions['session_id'].astype(int).values
        )
    )
    predictions = predictions[~predictions['session_id'].isin(MISSING_IN_POPULAR)]

    not_in_common = baseline[~baseline['session_id'].isin(predictions['session_id'])]
    df_out = pd.concat([predictions, not_in_common], ignore_index=True)
    submission_path = os.path.join(DATA_PATH, 'submissions', '{}.csv'.format(get_string_timestamp()))
    df_out.to_csv(submission_path, index=False)

    print("Do submission")

    verify_subm.callback(
        data_path='/',
        submission_file=os.path.abspath(submission_path),
        test_file=os.path.join(os.path.abspath(DATA_PATH), 'raw/test.csv')
    )

if __name__ == '__main__':
    create_submission(MODEL_PATH)

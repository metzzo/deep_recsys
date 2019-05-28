import os
import pickle

import progressbar
import torch
from torch.utils.data import DataLoader
from verify_submission.verify_subm import main as verify_subm

from utility.split_utility import RandomSampleStrategy
from utility.helpers import get_string_timestamp
import pandas as pd

RECOMMENDER_MODEL_PATH = './data/model/2019_05_27_01_05_01_0.63_full.pth'
RANKING_MODEL_PATH = './data/ranking_model/2019_05_28_01_21_00_0.45.pth'
SUBMISSION_BATCH_SIZE = 512

# these are missing in submission popular for some reason
#MISSING_IN_POPULAR = ["f4e7686de96f6", "28aed6ff4f5cf", "1dc624176b25a", "3146144dd180c", "6395e6166436d", "5fd68f0c5a86f", "0c558de5076d3", "be040dde80c92", "692367c2a1e0e", "2fea505733796", "9f73d9d10c1e3", "f3fcfa14e3c1b", "51d1160090a5e", "9b38c0d8c7736", "f761e020a59f6", "a22b00a80121c", "a9641c5908155", "76b4000458cba", "c18c1f77a080b", "4df3c262efcf1", "2555579f7370b", "dd8438cfdd337", "3a4e144c181a5", "1e65b48e85ac6", "16cb478a81ef9", "263608e619c16", "095a051b27180", "e1f2b85aa8440", "ff30a418d7821"]

MODE = 'test'
MERGE_WITH_REST = True

def get_baseline():
    from train_recommender import RAW_DATA_PATH

    return pd.read_csv(os.path.join(RAW_DATA_PATH, 'submission_popular.csv'))


def do_prediction(recommender_state, ranking_state, dataset):
    from recommender_configs import prepare_config
    from utility.prediction import Prediction
    from network.recommender_network import RecommenderNetwork
    from network.impression_network import ImpressionRankNetwork
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    recommender_state_config = prepare_config(recommender_state.get('config'))
    rc_network_state_dict = recommender_state.get('network_state_dict')

    ir_state_config = prepare_config(ranking_state.get('config'))
    ir_network_state_dict = ranking_state.get('network_state_dict')

    recommender_network = RecommenderNetwork(
        config=recommender_state_config,
        item_size=dataset.item_size,
        target_item_size=dataset.target_item_size,
    )
    recommender_network.load_state_dict(rc_network_state_dict)
    recommender_network.eval()

    rank_network = ImpressionRankNetwork(
        config=ir_state_config,
        item_size=dataset.item_size,
        device=device,
    )
    rank_network.load_state_dict(ir_network_state_dict)
    rank_network.eval()

    data_loader = DataLoader(
        dataset,

        batch_size=SUBMISSION_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=dataset.collator
    )

    cur_prediction = Prediction(
        dataset=dataset,
        device=device,
    )

    recommender_network = recommender_network.to(device)
    rank_network = rank_network.to(device)
    print("Begin predicting...")
    with progressbar.ProgressBar(max_value=int(len(dataset) / SUBMISSION_BATCH_SIZE + 1), redirect_stdout=True) as bar:
        for idx, data in enumerate(data_loader):
            sessions, session_lengths, _, item_impressions, impression_ids, _, ids = data
            sessions = sessions.to(device)

            item_scores = recommender_network(sessions, session_lengths).float()
            selected_impression = rank_network(item_impressions, item_scores)

            cur_prediction.add_predictions(
                ids=ids,
                impression_ids=impression_ids,
                item_impressions=item_impressions,
                item_scores=item_scores,
                selected_impression=selected_impression
            )
            bar.update(idx)

    return cur_prediction


def create_submission(recommender_path, ranking_path):
    from dataset.recsys_dataset import RecSysData, RecSysDataset
    from train_recommender import DATA_PATH

    print("Create Submission File for: ", recommender_path, ranking_path)

    recommender_state = torch.load(recommender_path)
    ranking_state = torch.load(ranking_path)

    test_rec_sys_data = RecSysData(mode=MODE, size=None)

    dataset = RecSysDataset(
        rec_sys_data=test_rec_sys_data,
        split_strategy=RandomSampleStrategy(split=1.0),
        include_impressions=True,
        train_mode=False
    )

    cur_prediction = do_prediction(
        recommender_state=recommender_state,
        ranking_state=ranking_state,
        dataset=dataset
    )
    pickle.dump(cur_prediction, open(os.path.join(DATA_PATH, 'prediction_dump.pickle'), "wb"), protocol=4)
    #cur_prediction = pickle.load(open(os.path.join(DATA_PATH, 'prediction_dump.pickle'), "rb"))
    submission_path = os.path.join(DATA_PATH, 'submissions', '{}.csv'.format(get_string_timestamp()))

    session_encoder = test_rec_sys_data.session_label_encoders['session_id']
    predictions = cur_prediction.predictions

    predictions['session_id'] = pd.Series(
        session_encoder.inverse_transform(
            predictions['session_id'].astype(int).values
        )
    )

    if MERGE_WITH_REST:
        baseline = get_baseline()
        #predictions = predictions[~predictions['session_id'].isin(MISSING_IN_POPULAR)]

        not_in_common = baseline[~baseline['session_id'].isin(predictions['session_id'])]
        df_out = pd.concat([predictions, not_in_common], ignore_index=True)
        df_out.to_csv(submission_path, index=False)

        print("Do submission")

        verify_subm.callback(
            data_path='/',
            submission_file=os.path.abspath(submission_path),
            test_file=os.path.join(os.path.abspath(DATA_PATH), 'raw/test.csv')
        )
    else:
        predictions.to_csv(submission_path, index=False)

if __name__ == '__main__':
    create_submission(
        recommender_path=RECOMMENDER_MODEL_PATH,
        ranking_path=RANKING_MODEL_PATH
    )

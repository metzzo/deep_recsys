import os

import pandas as pd
import torch.nn.functional as F
import torch

from utility.score import score_submissions, get_reciprocal_ranks, SUBM_INDICES

rank_net = None

class Prediction(object):
    def __init__(self, dataset, device, add_reference=True):
        super()
        self.dataset = dataset
        self.device = device
        self.prediction_ptr = 0

        dict_data = {
            'user_id': [''] * len(dataset),
            'session_id': [''] * len(dataset),
            'timestamp': [''] * len(dataset),
            'step': [''] * len(dataset),
            'item_recommendations': [''] * len(dataset),
        }
        if add_reference:
            dict_data['reference'] = [''] * len(dataset)

        self.predictions = pd.DataFrame.from_dict(dict_data)
        self.add_reference = add_reference

        # load rank network
        """
        from train_ranking import load_ranknet_network
        from train_ranking import DATA_PATH

        global rank_net

        if rank_net is None:
            rank_net = load_ranknet_network(
                path=os.path.join(DATA_PATH, 'ranking_model', '2019_05_26_14_34_49.pth'),
                device=device
            )
            rank_net = rank_net.to(device)
        self.rank_net = rank_net
        """

    def rank(self, predicted, impressions, impreession_ids):
        item_score_repeated = predicted.repeat(len(impressions), 1)
        sim = F.cosine_similarity(item_score_repeated, impressions)
        """
        sim = self.rank_net.predict(
            input_1=item_score_repeated,
            input_2=impressions
        ).flatten()
        """
        sorted = torch.argsort(sim, descending=True)
        sorted_impressions = ' '.join(
            torch.gather(impreession_ids, 0, sorted).cpu().numpy().astype(str))

        return sorted_impressions


    def add_predictions(self, ids, impression_ids, item_impressions, item_scores):
        item_scores = item_scores.to(self.device)

        for id, impression_id, item_impression, item_score in zip(ids,
                                                                  impression_ids,
                                                                  item_impressions,
                                                                  item_scores):

            impression_id = impression_id.to(self.device)
            item_impression = item_impression.to(self.device).float()
            sorted_impressions = self.rank(
                predicted=item_score,
                impressions=item_impression,
                impreession_ids=impression_id,

            )
            cur_pred = self.predictions.iloc[self.prediction_ptr]
            cur_pred.at['item_recommendations'] = sorted_impressions
            cur_pred.at['user_id'] = id['user_id']
            cur_pred.at['session_id'] = id['session_id']
            cur_pred.at['timestamp'] = id['timestamp']
            cur_pred.at['step'] = id['step']
            if self.add_reference:
                cur_pred.at['reference'] = self.dataset.rec_sys_data.session_df.loc[
                    self.dataset.rec_sys_data.groups[id['session_id']]
                ]['reference'].iloc[-1]
            self.prediction_ptr += 1
            #except e:
            #    print("Prediction Error", impression_id.detach().cpu().numpy(), item_impression.detach().cpu().numpy())

    def get_score(self):
        predictions = self.predictions
        predictions.set_index(SUBM_INDICES, inplace=True)

        ground_truth_df = self.dataset.get_submission()
        predicted_df = predictions

        return score_submissions(
            df_subm=predicted_df,
            df_gt=ground_truth_df,
            objective_function=get_reciprocal_ranks
        )


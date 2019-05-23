import pandas as pd
import torch.nn.functional as F
import torch

from utility.score import score_submissions, get_reciprocal_ranks, SUBM_INDICES


class Prediction(object):
    def __init__(self, dataset, device):
        super()
        self.dataset = dataset
        self.device = device
        self.prediction_ptr = 0
        self.predictions = pd.DataFrame.from_dict({
            'user_id': [''] * len(dataset),
            'session_id': [''] * len(dataset),
            'timestamp': [''] * len(dataset),
            'step': [''] * len(dataset),
            'item_recommendations': [''] * len(dataset),
        })

    def add_predictions(self, ids, impression_ids, item_impressions, item_scores):
        item_scores = item_scores.to(self.device)

        for id, impression_id, item_impression, item_score in zip(ids,
                                                                  impression_ids,
                                                                  item_impressions,
                                                                  item_scores):
            impression_id = impression_id.to(self.device)
            item_impression = item_impression.to(self.device).float()

            item_score_repeated = item_score.repeat(len(item_impression), 1)
            sim = F.cosine_similarity(item_score_repeated, item_impression)
            sorted = torch.argsort(sim, descending=True)
            sorted_impressions = ' '.join(
                torch.gather(impression_id, 0, sorted).cpu().numpy().astype(str))
            cur_pred = self.predictions.iloc[self.prediction_ptr]
            cur_pred.at['item_recommendations'] = sorted_impressions
            cur_pred.at['user_id'] = id['user_id']
            cur_pred.at['session_id'] = id['session_id']
            cur_pred.at['timestamp'] = id['timestamp']
            cur_pred.at['step'] = id['step']
            self.prediction_ptr += 1

    def get_score(self):
        self.predictions.set_index(SUBM_INDICES, inplace=True)

        ground_truth_df = self.dataset.get_submission()
        predicted_df = self.predictions

        return score_submissions(
            df_subm=predicted_df,
            df_gt=ground_truth_df,
            objective_function=get_reciprocal_ranks
        )


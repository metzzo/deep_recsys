import pandas as pd
import torch.nn.functional as F
import torch

from utility.score import score_submissions, get_reciprocal_ranks, SUBM_INDICES


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

    def add_predictions(self, ids, impression_ids, item_impressions, item_scores):
        item_scores = item_scores.to(self.device)

        for id, impression_id, item_impression, item_score in zip(ids,
                                                                  impression_ids,
                                                                  item_impressions,
                                                                  item_scores):

            impression_id = impression_id.to(self.device)
            item_impression = item_impression.to(self.device).float()
            try:
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
                if self.add_reference:
                    cur_pred.at['reference'] = id['reference']
                self.prediction_ptr += 1
            except:
                print("Error", impression_id.detach().cpu().numpy(), item_impression.detach().cpu().numpy())

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


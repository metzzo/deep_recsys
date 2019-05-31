import os

import pandas as pd
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utility.score import score_submissions, get_reciprocal_ranks, SUBM_INDICES

rank_net = None

IDF_RANKING = False

class Prediction(object):
    def __init__(self, dataset, device, use_cosine_similarity=False):
        super()
        self.dataset = dataset
        self.device = device
        self.prediction_ptr = 0
        self.tfidf_vectorizer = dataset.rec_sys_data.item_vectorizer

        dict_data = {
            'user_id': [''] * len(dataset),
            'session_id': [''] * len(dataset),
            'timestamp': [''] * len(dataset),
            'step': [''] * len(dataset),
            'item_recommendations': [''] * len(dataset),
        }

        self.predictions = pd.DataFrame.from_dict(dict_data)
        self.use_cosine_similarity = use_cosine_similarity

    def rank(self, predicted, impressions, impreession_ids, selected_impression):
        if self.use_cosine_similarity:
            if IDF_RANKING:
                predicted = np.multiply(self.tfidf_vectorizer.idf_, predicted.detach().cpu().numpy())
                impressions = impressions.detach().cpu().numpy()
                count = impressions.shape[0]
                idf = np.tile(self.tfidf_vectorizer.idf_, (count, 1))
                impressions = np.multiply(impressions, idf)
                predicted = np.tile(predicted, (1, 1))
                similarities = -cosine_similarity(predicted, impressions).reshape(-1)
                sorted = np.argsort(similarities)
                sorted_impressions = ' '.join(
                    np.take(impreession_ids.detach().cpu().numpy(), sorted).astype(str)
                )
            else:
                item_score_repeated = predicted.repeat(len(impressions), 1)
                sim = F.cosine_similarity(item_score_repeated, impressions)
                sorted = torch.argsort(sim, descending=True)

                sorted_impressions = ' '.join(
                    torch.gather(impreession_ids, 0, sorted).cpu().numpy().astype(str)
                )
        else:
            impreession_ids = impreession_ids.detach().cpu().numpy()
            selected_impression = selected_impression.detach().cpu().numpy()[:len(impreession_ids)]
            sorted = np.argsort(-selected_impression)
            sorted_impressions = ' '.join(
                impreession_ids[sorted].astype(str)
            )

        return sorted_impressions

    def add_predictions(self, ids, impression_ids, item_impressions, item_scores, selected_impression=None):
        item_scores = item_scores.to(self.device)
        if selected_impression is None:
            selected_impression = [0] * len(ids)

        for id, impression_id, item_impression, item_score, selected_impression in zip(
                ids,
                impression_ids,
                item_impressions,
                item_scores,
                selected_impression):

            impression_id = impression_id.to(self.device)
            item_impression = item_impression.to(self.device).float()
            sorted_impressions = self.rank(
                predicted=item_score,
                impressions=item_impression,
                impreession_ids=impression_id,
                selected_impression=selected_impression,

            )
            cur_pred = self.predictions.iloc[self.prediction_ptr]
            cur_pred.at['item_recommendations'] = sorted_impressions
            cur_pred.at['user_id'] = id['user_id']
            cur_pred.at['session_id'] = id['session_id']
            cur_pred.at['timestamp'] = id['timestamp']
            cur_pred.at['step'] = id['step']

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


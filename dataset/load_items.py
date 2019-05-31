import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_items():
    from train_recommender import RAW_DATA_PATH, DATA_PATH

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
            binary=False,
            use_idf=True,
        )

        item_properties = tfidf_vectorizer.fit_transform(raw_df['properties'])
        item_properties = (item_properties > 0.00001).astype(float)
        item_properties = pd.DataFrame(item_properties.toarray())
        item_properties.set_index(raw_df['item_id'], inplace=True)

        result = item_properties, tfidf_vectorizer
        pickle.dump(result, open(pickle_path, "wb"))
    return result
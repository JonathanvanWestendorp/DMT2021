import lightgbm as lgb
import pandas as pd
import numpy as np

from dataset import PropRanking


def add_cheapest(raw):
    cheapest_indices = raw[['srch_id', 'price_usd']].groupby('srch_id')['price_usd'].idxmin()
    cheapest_bool = np.zeros(len(raw))
    cheapest_bool[cheapest_indices] = 1
    raw['cheapest_bool'] = cheapest_bool
    return raw


def train(data):
    raw = pd.read_pickle("../Data/training_set_VU_DM.pkl")

    groups = raw.srch_id.value_counts(sort=False).sort_index()
    split = round(0.98 * raw.srch_id.nunique())
    split_srch_id = groups.index[split]
    data_split = raw[raw.srch_id == split_srch_id].index[0]

    # Remove if to many NaN values
    raw = raw.dropna(thresh=len(raw) * .995, axis=1)

    # Else fill with zeros
    raw = raw.fillna(0)

    # Increase booking weight
    targets = (raw.click_bool + 4 * raw.booking_bool).values

    # Add cheapest hotel per srch_id to features
    data, raw = add_cheapest(data), add_cheapest(raw)

    # Irrelevant features
    to_remove = ['srch_id', 'date_time', 'position', 'click_bool', 'booking_bool']
    features = raw.drop(to_remove, axis=1)

    # Does not help
    # features = ((features - features.mean()) / (features.std())).values

    X_train, X_val = features[:data_split], features[data_split:]
    y_train, y_val = targets[:data_split], targets[data_split:]
    print(len(X_train), len(X_val))

    groups_train, groups_val = groups[:split], groups[split:]

    gbm = lgb.LGBMRanker(learning_rate=.1, num_iterations=400, num_leaves=200, subsample=1., feature_fraction=.8)

    gbm.fit(X_train, y_train, group=groups_train, eval_set=[(X_val, y_val)],
            eval_group=[groups_val], eval_at=[5, 10])

    n_queries = data.srch_id.nunique()
    print("Predicting...")
    # for i, (query, query_group) in enumerate(data.groupby('srch_id')):
    #     query_batch = query_group.drop('srch_id', axis=1).values
    #     out = gbm.predict(query_batch)
    #
    #     order = out.argsort()[::-1]
    #
    #     # Order by learned ranking
    #     result = query_group.reset_index()[['srch_id', 'prop_id']].reindex(order)
    #
    #     # Save results
    #     result.to_csv("../Predictions/LGBM_tuned_split.csv", mode='a', header=(i == 0), index=False)
    #
    #     if i % 100 == 0:
    #         print(f"Predicted {i}/{n_queries}")


if __name__ == '__main__':
    dataset = PropRanking(test_path="../Data/test_set_VU_DM.pkl")
    data = dataset.get_test()
    train(data)

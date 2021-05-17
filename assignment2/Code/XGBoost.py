import xgboost as xgb
import pandas as pd

from dataset import PropRanking


def train(data):
    raw = pd.read_pickle("../Data/training_set_VU_DM.pkl")

    groups = raw.srch_id.value_counts(sort=False).sort_index()

    # Remove if to many NaN values
    raw = raw.dropna(thresh=len(raw) * .8, axis=1)

    # Else fill with zeros
    raw = raw.fillna(0)

    # Increase booking weight
    y_train = (raw.click_bool + 4 * raw.booking_bool).values

    # Irrelevant features
    to_remove = ['srch_id', 'date_time', 'position', 'click_bool', 'booking_bool']

    X_train = raw.drop(to_remove, axis=1).values

    model = xgb.XGBRanker(
        tree_method='gpu_hist',
        booster='gbtree',
        objective='rank:ndcg',
        random_state=42,
        learning_rate=0.05,
        colsample_bytree=0.9,
        eta=0.05,
        max_depth=6,
        n_estimators=110,
        subsample=0.75
    )

    model.fit(X_train, y_train, group=groups, verbose=True)

    n_queries = data.srch_id.nunique()
    print("Predicting...")
    for i, (query, query_group) in enumerate(data.groupby('srch_id')):
        query_batch = query_group.drop('srch_id', axis=1).values
        out = model.predict(query_batch)

        order = out.argsort()[::-1]

        # Order by learned ranking
        result = query_group.reset_index()[['srch_id', 'prop_id']].reindex(order)

        # Save results
        result.to_csv("../Predictions/XGB_ndcg.csv", mode='a', header=(i == 0), index=False)

        if i % 100 == 0:
            print(f"Predicted {i}/{n_queries}")


if __name__ == '__main__':
    dataset = PropRanking(test_path="../Data/test_set_VU_DM.pkl")
    data = dataset.get_test()
    train(data)

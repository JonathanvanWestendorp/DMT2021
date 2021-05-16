import torch

from dataset import PropRanking
from model import NeuralModule


def predict(data_path, model_path, out_path):
    print("Loading data...")
    dataset = PropRanking(test_path=data_path)
    data = dataset.get_test()

    n_queries = data.srch_id.nunique()

    print("Loading model...")
    model = NeuralModule(19, 1)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Predicting...")
    for i, (query, query_group) in enumerate(data.groupby('srch_id')):
        query_batch = torch.tensor(query_group.drop('srch_id', axis=1).values, dtype=torch.float)
        out = model(query_batch).squeeze()
        order = out.argsort(descending=True).numpy()

        # Order by learned ranking
        result = query_group.reset_index()[['srch_id', 'prop_id']].reindex(order)

        # Save results
        result.to_csv(out_path, mode='a', header=(i == 0), index=False)

        if i % 100 == 0:
            print(f"Predicted {i}/{n_queries}")

    print("Predicting finished")
    return True


if __name__ == '__main__':
    predict("../Data/test_set_VU_DM.pkl", "../Models/pointwise_epochs3_seed0_nan_cols_removed", '../Predictions/pointwise_epochs3_seed0_nan_cols_removed.csv')

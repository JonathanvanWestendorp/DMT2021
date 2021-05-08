from dataset import PropRanking


def predict(path):
    print("Loading data...")
    dataset = PropRanking(test_path=path)
    data = dataset.get_test()
    for query, query_group in data.groupby('srch_id'):
        print(query)
        print(query_group)
        # TODO get ranking for group and sort


if __name__ == '__main__':
    predict("../Data/test_set_VU_DM.pkl")

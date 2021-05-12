import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import mplcursors

def perform_PCA(data, subset=None, norm=None, transform=None):
    if subset:
        data = data[:subset].copy(deep=True)

    if norm:
        data = (data - data.mean()) / (data.max() - data.min())

    if transform:
        data = data.T
        hue = data.index.values
        ncolors = len(hue)

    else:
        ncolors = data.srch_id.nunique()
        hue = "srch_id"

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(data)

    data['pca-one'] = pca_result[:, 0]
    data['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue=hue,
        palette=sns.color_palette("hls", ncolors),
        data=data,
        legend="full",
        alpha=0.8
    )

    if transform:
        mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(hue[sel.target.index]))

    plt.title("PCA")
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.legend(fontsize="x-small")

    plt.show()

def perform_tsne(data, subset=None, norm=None, transform=None):
    if subset:
        data = data[:subset].copy(deep=True)

    if norm:
        data = (data - data.mean()) / (data.max() - data.min())

    if transform:
        data = data.T
        hue = data.index.values
        ncolors = len(hue)
    else:
        ncolors = data.srch_id.nunique()
        hue = "srch_id"

    print(f"Amount of columns/features in dataing set: {len(data.columns.values)}")
    print(f"Amount of search queries in dataset: {len(data)}")
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time() - time_start))
    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=hue,
        palette=sns.color_palette("hls", ncolors),
        data=data,
        legend="full",
        alpha=0.8)

    plt.title("T-SNE")
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.legend(fontsize="x-small")
    plt.show()

if __name__ == "__main__":
    train = pd.read_pickle("../Data/trainingset.pkl")
    train = train.fillna(0)
    del train["date_time"]

    #perform_tsne(train, subset=20000, norm=True, transform=True)
    perform_PCA(train, subset=10000, norm=True, transform=True)

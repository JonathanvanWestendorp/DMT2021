import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time
import mplcursors

def count_bookings(data, subset=None, save=None):
    if subset:
        data = data[:subset].copy(deep=True)

    print(f"In total there were {data.srch_id.nunique()} searches done which gave {len(data)} results")
    print(f"Out of these {data.srch_id.nunique()} searches {len(data[data['booking_bool'] == 1])} properties were booked")
    print(f"The booked properties had an average star rating of {round(data[data['booking_bool'] == 1].prop_starrating.mean(),2)}")
    print(f"The properties that weren't booked had an average star rating of {round(data[data['booking_bool'] == 0].prop_starrating.mean(),2)}")

    # Count bookings for each star rating and reorder array based on star rating
    star_counts = data[data['booking_bool'] == 1].prop_starrating.value_counts().sort_index()
    sns.barplot(x=star_counts.index, y=star_counts.values)
    plt.title("Booked properties per star rating")
    plt.xlabel("Star rating (0 means no rating available)")
    plt.ylabel("Number of bookings")

    if save:
        plt.savefig("bookingcounts", dpi=300)

    plt.show()

def average_price(data, subset=None, save=None):
    if subset:
        data = data[:subset].copy(deep=True)

    countries = np.sort(data[data['booking_bool'] == 1].prop_country_id.unique())
    mean_prices = np.array(data[data['booking_bool'] == 1].groupby("prop_country_id").mean()["price_usd"])
    indices = [i for i, elem in enumerate(mean_prices) if elem >= 1000]

    outliers = mean_prices[indices]
    outliers_countries = countries[indices]

    mean_prices = mean_prices[mean_prices < 1000]
    countries = np.delete(countries, indices)
    plt.scatter(countries, mean_prices)
    plt.plot(countries, mean_prices)
    plt.xlabel("Country ID")
    plt.ylabel("Mean price in USD")
    plt.title("Mean price of booked properties per country")

    if save:
        plt.savefig("prices_per_country", dpi=300)
    plt.show()

    plt.scatter(outliers_countries, outliers)
    plt.xlabel("Country ID")
    plt.ylabel("Mean price in USD")
    plt.title("Mean price of booked properties per country (outliers)")
    plt.xticks(outliers_countries)

    if save:
        plt.savefig("prices_per_country_outliers", dpi=300)
    plt.show()

def promotion(data, subset = None, save=None):
    if subset:
        data = data[:subset].copy(deep=True)
    countings = data[data['booking_bool'] == 1].promotion_flag.value_counts()
    sns.barplot(x=countings.index, y=countings.values)
    plt.xticks([0, 1], ["No Sale", "Sale"])
    plt.title("Amount of booked properties if in sale or not")
    plt.ylabel("Amount of booked properties")

    if save:
        plt.savefig("promotions", dpi=300)
    plt.show()

def perform_PCA(data, subset=None, norm=None, transform=None, save=None):
    if subset:
        data = data[:subset].copy(deep=True)

    if norm:
        data = (data - data.mean()) / (data.max() - data.min())

    if transform:
        data = data.T
        legend = "full"
        hue = data.index.values
        ncolors = len(hue)
    else:
        ncolors = data.srch_id.nunique()
        hue = "srch_id"
        legend = False

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
        legend=legend,
        alpha=0.8
    )

    plt.title("PCA")
    plt.xlabel("First component")
    plt.ylabel("Second component")

    if transform:
        plt.legend(fontsize="x-small")
        mplcursors.cursor().connect("add", lambda sel: sel.annotation.set_text(hue[sel.target.index]))
        if save:
            plt.savefig("PCA_transposed", dpi=300)
    elif save:
        plt.savefig("PCA", dpi=300)
    plt.show()

def perform_tsne(data, subset=None, norm=None, transform=None, save=None):
    if subset:
        data = data[:subset].copy(deep=True)

    if norm:
        data = (data - data.mean()) / (data.max() - data.min())

    if transform:
        legend = "full"
        data = data.T
        hue = data.index.values
        ncolors = len(hue)
    else:
        ncolors = data.srch_id.nunique()
        legend = False
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
        legend=legend,
        alpha=0.8)

    plt.title("T-SNE")
    plt.xlabel("First component")
    plt.ylabel("Second component")

    if transform:
        plt.legend(fontsize="x-small")
        if save:
            plt.savefig("TSNE_transposed", dpi=300)
    elif save:
        plt.savefig("TSNE", dpi=300)
    plt.show()

if __name__ == "__main__":
    train = pd.read_pickle("../Data/trainingset.pkl")
    train = train.fillna(0)
    del train["date_time"]

    count_bookings(train, save=True)
    average_price(train, save=True)
    promotion(train, save=True)

    perform_tsne(train, subset=20000, norm=True, transform=False, save=True)
    perform_PCA(train, subset=20000, norm=True, transform=False, save=True)

    perform_tsne(train, subset=20000, norm=True, transform=True, save=True)
    perform_PCA(train, subset=20000, norm=True, transform=True, save=True)

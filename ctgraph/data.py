import datetime
import pdb

import numpy as np
import pandas as pd
import logging
from torch_geometric.data import HeteroData

from ctgraph import instagram, features, logger, Task, task

import torch_geometric.transforms as T

AMAZON_PATH = "amazon"
HM_PATH = "hm"

AMAZON_DATASETS = ['beauty', 'cd', 'games', 'movie']


# Prep
def refine_time(data):
    """
    assures items bought by a user don't have the exact same time
    5, 1, 2, 2, 8 -> 1, 2, 3, 5, 8
    """

    # Ensure every t is a natural number
    assert np.sum((data['t'].round() - data['t']).abs()) == 0

    # There needs to be a difference in time, but it can be tiny
    # Since the t's are nats, we can add random noise in between 0 or 1 without changing
    # the transaction order. I also divide by 4 to reduce the noise and shuffle to remove bias.
    # With the arange I ensure every value becomes unique.
    noise = np.arange(len(data))
    np.random.shuffle(noise)
    data['t'] = data['t'].values + noise / len(data) / 4

    assert len(data['t']) == len(data['t'].unique())

    return data

def normalize_all_time(data):
    """
    moves the times to a region between 0 and 1
    """
    start = data['t'].min()
    end = data['t'].max()

    duration = end - start

    data['t'] = (data['t'] - start) / duration

# noinspection PyTypeChecker
def amazon_dataset(name):
    task = Task(f"Loading amazon/{name} dataset").start()

    df = pd.read_csv(f"{AMAZON_PATH}/{name.capitalize()}.csv").rename({"user_id": "u", "item_id": "i", "time": "t"},
                                                                      axis=1)

    # make time unique per user by adding super tiny bit of noise
    df = df.groupby('u').apply(refine_time).reset_index(drop=True)
    df['t'] = df['t'].astype('float64')


    # normalize time
    normalize_all_time(df)

    customers = df['u'].unique()
    articles = df['i'].unique()

    assert customers.max() + 1 == len(customers)
    assert articles.max() + 1 == len(articles)

    data = HeteroData()

    customer_indexes = df['u'].to_numpy()
    article_indexes = df['i'].to_numpy()

    data['u', 'b', 'i'].edge_index = np.concatenate([customer_indexes[None, :], article_indexes[None, :]], axis=0)

    data['u'].code = customers
    data['i'].code = articles
    data['u', 'b', 'i'].code = np.arange(len(df))

    data['u', 'b', 'i'].t = df["t"].to_numpy()

    task.done()
    return data


class HMData(object):
    def __init__(self, load=True, features=True):
        if load:
            self._load_data()
        if features:
            self._create_features()

    @task("Loading H&M data")
    def _load_data(self):
        self.df_transactions = pd.read_parquet(f'{HM_PATH}/transactions_parquet.parquet')
        self.df_customers = pd.read_parquet(f'{HM_PATH}/customers_parquet.parquet')
        self.df_articles = pd.read_parquet(f'{HM_PATH}/articles_parquet.parquet')

        # Convert date column to datetime
        self.df_transactions["date"] = pd.to_datetime(self.df_transactions["t_dat"], format="%Y-%m-%d")

    @task("Creating H&M features")
    def _create_features(self):
        articles_ = self.df_articles
        articles = articles_.pipe(instagram.add_post_data_articles, instapath=f"{HM_PATH}/hm/")
        articles_features = articles.pipe(features.one_hot_concat,
                                          categorical=['product_type_name', 'product_group_name',
                                                       'graphical_appearance_name',
                                                       'colour_group_name', 'perceived_colour_value_name',
                                                       'perceived_colour_master_name',
                                                       'department_name', 'index_name', 'index_group_name',
                                                       'section_name',
                                                       'garment_group_name',
                                                       ],
                                          scalar=['post_likes_total', 'post_count'],
                                          index='article_id',
                                          merge_threshold=0.99,
                                          verbose=False)

        customers = self.df_customers
        customers_features = customers.pipe(features.one_hot_concat,
                                            categorical=['FN', 'Active', 'club_member_status',
                                                         'fashion_news_frequency'],
                                            scalar=['age'],
                                            index='customer_id',
                                            merge_threshold=1,
                                            verbose=False)

        transactions_ = self.df_transactions
        transactions = transactions_.pipe(instagram.add_post_data_transactions)
        transactions_features = transactions.pipe(features.one_hot_concat,
                                                  categorical=['sales_channel_id'],
                                                  scalar=['price'],
                                                  index=None,
                                                  merge_threshold=1,
                                                  verbose=False
                                                  )
        self.farticles = articles_features
        self.fcustomers = customers_features
        self.ftransactions = transactions_features

    def subset(self, days=7, keep_all_customers=False):
        """
        Create a subset of this data keeping the last train_days+test_days transactions.

        :param keep_all_customers: Keep all customers even if they have no transactions in the test or train sets.
                                   Useful for making a submission.
        :return: Framework
        """
        subdata = HMData(load=False, features=False)
        t = self.df_transactions["date"].max()
        tminus = datetime.timedelta(days=days)
        subdata.df_transactions = self.df_transactions.loc[self.df_transactions['date'] > t - tminus].copy()

        if keep_all_customers:
            subdata.df_customers = self.df_customers.copy()
        else:
            subdata.df_customers = self.df_customers.loc[
                self.df_customers['customer_id'].isin(subdata.df_transactions['customer_id'])].copy()

        subdata.df_articles = self.df_articles.loc[
            self.df_articles['article_id'].isin(subdata.df_transactions['article_id'])].copy()

        subdata.farticles = self.farticles.loc[subdata.df_articles['article_id']]
        subdata.fcustomers = self.fcustomers.loc[subdata.df_customers['customer_id']]
        subdata.ftransactions = self.ftransactions.loc[subdata.df_transactions.index]
        return subdata

    # noinspection PyTypeChecker
    @task("Creating H&M graph")
    def as_graph(self):
        # task = Task("Creating H&M graph").start()
        data = HeteroData()
        data['u'].x = self.fcustomers.to_numpy()
        data['i'].x = self.farticles.to_numpy()

        # Create a map from customer and article ids to node indices
        ci = self.fcustomers.assign(index=np.arange(len(self.fcustomers)))
        customer_indexes = ci.loc[self.df_transactions['customer_id'], 'index'].to_numpy()
        ai = self.farticles.assign(index=np.arange(len(self.farticles)))
        article_indexes = ai.loc[self.df_transactions['article_id'], 'index'].to_numpy()

        data['u', 'b', 'i'].edge_index = np.concatenate(
            [customer_indexes[None, :], article_indexes[None, :]], axis=0)
        data['u', 'b', 'i'].edge_attr = self.ftransactions.to_numpy()

        data['u'].code = np.arange(self.fcustomers.shape[0])  # fcustomers.index.to_numpy())
        data['i'].code = np.arange(self.farticles.shape[0])  # farticles.index.to_numpy())
        data['u', 'b', 'i'].code = np.arange(self.ftransactions.shape[0])

        # # Rich date information
        # data['u', 'b', 'i'].dates = self.df_transactions["date"].to_numpy()

        # Day numbers since unix epoch
        data['u', 'b', 'i'].t = (self.df_transactions["date"] - np.datetime64('1970')).dt.days.to_numpy()

        # task.done()
        return data

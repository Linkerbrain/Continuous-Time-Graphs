import datetime

import numpy as np
import pandas as pd
import logging
from torch_geometric.data import HeteroData

from sgat import instagram, features, logger

import torch_geometric.transforms as T

AMAZON_PATH = "amazon"
HM_PATH = "hm"

AMAZON_DATASETS = ['beauty', 'cd', 'games', 'movie']


# noinspection PyTypeChecker
def amazon_dataset(name):
    logger.info(f"Loading amazon/{name} dataset...")

    df = pd.read_csv(f"{AMAZON_PATH}/{name.capitalize()}.csv").rename({"user_id": "u", "item_id": "i", "time": "t"}, axis=1)

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

    logger.info("Done")
    return data


class HMData(object):
    def __init__(self, load=True, features=True):
        if load:
            self._load_data()
        if features:
            self._create_features()

    def _load_data(self):
        logger.info("Loading H&M data...")
        self.df_transactions = pd.read_parquet(f'{HM_PATH}/transactions_parquet.parquet')
        self.df_customers = pd.read_parquet(f'{HM_PATH}/customers_parquet.parquet')
        self.df_articles = pd.read_parquet(f'{HM_PATH}/articles_parquet.parquet')
        logger.info("Done")

        # Convert date column to datetime
        self.df_transactions["date"] = pd.to_datetime(self.df_transactions["t_dat"], format="%Y-%m-%d")


    def _create_features(self):
        logger.info("Creating H&M features...")
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
        logger.info("Done")
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
    def as_graph(self):
        logger.info("Creating H&M graph...")
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

        # Rich date information
        data['u', 'b', 'i'].dates = self.df_transactions["date"].to_numpy()

        # Day numbers since unix epoch
        data['u', 'b', 'i'].t = (self.df_transactions["date"] - np.datetime64('1970')).dt.days.to_numpy()
        return data


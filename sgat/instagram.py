import os
import re

import numpy as np
import pandas as pd


def get_posts(instapath):
    times = []
    codes_ = []
    likes_ = []
    for p in os.listdir(instapath):
        if p[-4:] == ".txt":
            txt = open(instapath + p, 'r').read()
            likes = re.findall(r'^XXLIKESXX:([0-9]+)', txt)
            if len(likes) == 0:
                continue
            caption = txt[txt.find('XXCAPTIONXX:'):]
            codes = re.findall(r'([0-9]{5}[0-9]*)', caption)
            for code in codes:
                times.append(p[:p.find('_')])
                likes_.append(int(likes[0]))
                codes_.append(code)
    posts = pd.DataFrame({'post_date': times, 'article_id': codes_, 'likes': likes_})
    posts['post_date'] = pd.to_datetime(posts['post_date'])
    return posts


def add_post_data_articles(articles, instapath):
    posts = get_posts(instapath)

    # Add whether the transacted item was posted
    # transactions['posted'] = transactions['article_id'].isin(posts['article_id'])

    post_likes_total = posts.groupby('article_id').sum()['likes']
    post_counts = posts.groupby('article_id').count()['likes']

    articles['post_likes_total'] = articles['article_id'].map(post_likes_total).replace(np.nan, 0)
    articles['post_count'] = articles['article_id'].map(post_counts).replace(np.nan, 0)
    return articles

def add_post_data_transactions(transactions):
    #transactions['post_likes'] = transactions['article_id'].map(likes_total).replace(np.NAN, 0)
    return transactions

    # Add the post date
    dated = transactions.merge(posts, on='article_id', how='left')

    print(transactions)
    nearest = transactions.groupby(['post_date']).apply(lambda x: print(x)) #x.sort_values(by='likes', ascending=False).iloc[0])
    transactions = transactions.loc[nearest]

    # Add the number of likes on the post
    transactions = transactions.merge(posts[['article_id', 'likes']], on='article_id', how='left')

    # How many other items it was posted with
    post_counts = posts.groupby(['post_date'])['article_id'].count().rename(columns={'article_id': 'post_with_count'})
    article_counts = post
    transactions['post_with'] = transactions.merge(counts, on='article_id', how='left')

    # The time difference between the post and the transaction
    transactions['post_time_diff'] = transactions.apply()

    return transactions

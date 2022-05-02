import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


# noinspection PyTypeChecker
def make_subset(data, filter_transactions=slice(None)):
    subdata = HeteroData()

    forward = ('u', 'bought', 'i')
    subdata[forward].edge_index = data[forward].edge_index[:, filter_transactions]

    for k, v in data[forward].items():
        if k != 'edge_index':
            subdata[forward][k] = v[filter_transactions]

    customers = np.unique(subdata[forward].edge_index[0])
    articles = np.unique(subdata[forward].edge_index[1])

    for k, v in data['u'].items():
        subdata['u'][k] = v[customers]
    for k, v in data['i'].items():
        subdata['i'][k] = v[articles]

    return subdata


class TemporalDataset(Dataset):
    # noinspection PyTypeChecker
    def __init__(self, graph, dates, chunk_days=7, embedding_chunks=4, supervision_chunks=1, val_splits=2,
                 test_splits=0,
                 negative_examples_ratio=1.0):
        """
        Step 0 (this class): Iterate over weeks of the data
        Step 1: Make a rolling dataset of the "past" as weeks progress, with the current week as target
        Step 2: For each timestep, make a neighbourloader for every customer that bought anything in the target week
                Neighbourloader can be biased towards recent transactions (n-latest transactions from user)
                Also an idea: Make neighbourloader iterate over similar customers based on their features or embeddings
                so that batches contain mostly related customers. This would definetly help with cold-starting
        Step 3: Train SageNet on predicting target week purchases based on neighbourloader samples.
                Multiple neighbourloader rounds can be done per timestep (one round goes through all customers)

        This model would likely take days if not weeks to train fully
        """
        self.embedding_chunks = embedding_chunks
        self.supervision_chunks = supervision_chunks
        self.val_splits = val_splits
        self.test_splits = test_splits
        self.graph = graph

        if negative_examples_ratio != 1.0:
            raise NotImplementedError()

        week = dates.floor(pd.Timedelta(chunk_days, 'D'))
        timesteps = []
        for w in week.unique():
            subset = week == w  # ordering of transactions and edges on data is the same
            # subdata = make_subset(data, filter_transactions=subset)
            # timesteps.append(subdata)
            timesteps.append(subset)
        self.timesteps = list(timesteps)

    # noinspection PyTypeChecker
    def make_split(self, chunks):
        embedding_graph = make_subset(self.graph,
                                      filter_transactions=np.bitwise_or.reduce(chunks[:self.embedding_chunks]))

        supervision_graph = make_subset(self.graph,
                                        filter_transactions=np.bitwise_or.reduce(chunks[self.embedding_chunks:]))
        edges = supervision_graph['u', 'bought', 'i']

        # Fake edges added as negative examples
        fake_edges = np.zeros_like(edges.edge_index)
        fake_edges[0, :] = np.random.randint(supervision_graph['u'].code.shape[0], size=edges.shape[1])
        fake_edges[1, :] = np.random.randint(supervision_graph['i'].code.shape[0], size=edges.shape[1])
        edges.edge_index = np.concatenate([edges.edge_index, fake_edges], axis=1)

        edges.label = np.ones_like(edges.edge_index)
        edges.label[-fake_edges.shape[1]:] = 0

        return embedding_graph, supervision_graph

    def roll(self, timesteps):
        """ Do a rolling window over te time steps given and split into embedding graph and supervision graph"""
        zippers = list(
            timesteps[i:len(self.timesteps) - self.embedding_chunks + self.supervision_chunks + i] for i in
            range(self.embedding_chunks + self.supervision_chunks)
        )
        for chunks in zip(*zippers):
            yield self.make_split(chunks)

    def train_dataloader(self):
        yield from self.roll(self.timesteps[:-self.test_splits - self.val_splits])

    def val_dataloader(self):
        yield from self.roll(
            self.timesteps[-self.embedding_chunks - self.val_splits - self.test_splits: -self.test_splits])

    def test_dataloader(self):
        yield from self.roll(self.timesteps[-self.embedding_chunks - self.test_splits:])

    def __len__(self):
        return len(self.timesteps)

    def __getitem__(self, idx):
        return self.timesteps[idx]


def add_transaction_order(data, dates):
    """
    Adds the following edge attributes:

    Add attribute 'oui' which means a transaction is the oui'th transaction from the user.
    Add attribute 'oiu' which means a user is the oiu'th purchaser of the item
    """

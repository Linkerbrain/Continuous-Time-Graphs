import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import numpy_indexed as npi


def numpy_to_torch(data):
    """
    Convert numpy-graph to torch-graph

    Torch geometric always expects torch-graphs

    Args:
        data:

    Returns:

    """
    for batch in data if type(data) is not HeteroData else [data]:
        graph = HeteroData()

        for k, v in batch.items():
            for attribute, value in v.items():
                graph[k][attribute] = torch.tensor(value, dtype=torch.long if attribute == 'edge_index' else None)

        if type(data) is HeteroData:
            return graph
        else:
            yield graph


# noinspection PyTypeChecker
def make_subset(data, filter_transactions=None, filter_customers=None, filter_articles=None):
    """

    Args:
        data:
        filter_transactions: Must be mask
        filter_customers: Must be mask
        filter_articles: Must be mask

    Returns:

    """
    subdata = HeteroData()

    forward = ('u', 'b', 'i')

    if filter_transactions is None:
        filter_transactions = np.ones(data[forward].code.shape[0], dtype=np.bool)

    if filter_customers is not None:
        # Edges must also be from wanted customers
        customer_indices = np.argwhere(filter_customers)
        filter_transactions &= np.isin(data[forward].edge_index[0], customer_indices)

    if filter_articles is not None:
        # And from wanted articles
        article_indices = np.argwhere(filter_articles)
        filter_transactions &= np.isin(data[forward].edge_index[1], article_indices)

    # Remove the unwanted edges
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

    # Make the edges point to the smaller set of users/items
    subdata[forward].edge_index[0] = npi.indices(customers, subdata[forward].edge_index[0])
    subdata[forward].edge_index[1] = npi.indices(articles, subdata[forward].edge_index[1])

    return subdata


class TemporalDataset(Dataset):
    # noinspection PyTypeChecker
    def __init__(self, graph, chunk_size=7, embedding_chunks=4, supervision_chunks=1, val_splits=2,
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

        t = graph['u', 'b', 'i'].t

        # if np.issubdtype(t.dtype, np.datetime64):
        #     # week = t.floor(pd.Timedelta(chunk_days, 'D'))
        #     chunk = np.floor_divide(t, pd.Timedelta(chunk_size, 'D'))
        # else:
        chunk = np.floor_divide(t, chunk_size)
        timesteps = []
        for w in np.unique(chunk):
            subset = chunk == w  # ordering of transactions and edges on data is the same
            # subdata = make_subset(data, filter_transactions=subset)
            # timesteps.append(subdata)
            timesteps.append(subset)
        self.timesteps = list(timesteps)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--chunk_size', type=int, default=7,
                            help="This is either days for hm data or a number for amazon data. For Beauty 10000000 gives 42 chunks")
        parser.add_argument('--embedding_chunks', type=int, default=4)
        parser.add_argument('--supervision_chunks', type=int, default=1)
        parser.add_argument('--val_splits', type=int, default=2)
        parser.add_argument('--test_splits', type=int, default=0)

    # noinspection PyTypeChecker
    def make_split(self, chunks):
        # TODO: Resample embedding_graph only from users in the supervision graph

        embedding_graph = make_subset(self.graph,
                                      filter_transactions=np.bitwise_or.reduce(chunks[:self.embedding_chunks]))

        supervision_graph = make_subset(self.graph,
                                        filter_transactions=np.bitwise_or.reduce(chunks[self.embedding_chunks:]))

        # noinspection PyUnreachableCode
        if True:  # TODO: Add as option to remove or not cold starts
            # Remove cold starts
            filter_customers = np.isin(supervision_graph['u'].code, embedding_graph['u'].code)
            filter_articles = np.isin(supervision_graph['i'].code, embedding_graph['i'].code)
            supervision_graph = make_subset(supervision_graph,
                                            filter_customers=filter_customers, filter_articles=filter_articles)
        else:
            # Add isolated nodes to embedding_graph?
            pass

        edges = supervision_graph['u', 'b', 'i']

        # Pointer to the index of these nodes in embedding_graph, by this point it should be guaranteed that all
        # nodes in the supervision graph are also in the embedding graph
        supervision_graph['u'].ptr = npi.indices(embedding_graph['u'].code, supervision_graph['u'].code)
        supervision_graph['i'].ptr = npi.indices(embedding_graph['i'].code, supervision_graph['i'].code)
        supervision_graph['u', 'b', 'i'].ptr = np.concatenate(
            (supervision_graph['u'].ptr[edges.edge_index[0]], supervision_graph['i'].ptr[edges.edge_index[1]]),
            axis=0
        )

        # Fake edges added as negative examples
        fake_edges = np.zeros_like(edges.edge_index)
        fake_edges[0, :] = np.random.randint(supervision_graph['u'].code.shape[0], size=edges.edge_index.shape[1])
        fake_edges[1, :] = np.random.randint(supervision_graph['i'].code.shape[0], size=edges.edge_index.shape[1])
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


def add_transaction_order(data):
    """
    Adds the following edge attributes:

    Add attribute 'oui' which means a transaction is the oui'th transaction from the user.
    Add attribute 'oiu' which means a user is the oiu'th purchaser of the item
    """
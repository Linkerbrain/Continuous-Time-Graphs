import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
import numpy_indexed as npi

from sgat import logger


def numpy_to_torch(data):
    """
    Convert numpy-graph to torch-graph

    Torch geometric always expects torch-graphs

    Args:
        data:

    Returns:

    """
    graph = HeteroData()

    for k in data.node_types + data.edge_types:
        for attribute, value in data[k].items():
            if attribute == 'edge_index':
                dtype = torch.long
            elif value.dtype == np.float:
                dtype = torch.float
            else:
                dtype = None
            graph[k][attribute] = torch.tensor(value, dtype=dtype)
    return graph

def check_graph(graph):
    for k in graph.edge_types:
        for attribute, value in graph[k].items():
            s = k[0]
            t = k[-1]
            if attribute == 'edge_index' and value[0].max() > graph[s].code.shape[0]:
                return False
            if attribute == 'edge_index' and value[1].max() > graph[t].code.shape[0]:
                return False
    return True


def compare_graphs(graph1, graph2, one_way=False):
    for k in graph1.node_types + graph1.edge_types:
        for attribute, value in graph1[k].items():
            if np.all(graph2[k][attribute] != value):
                return False
    if not one_way:
        return compare_graphs(graph2, graph1, True)
    return True


def nodes_codified(graph, node_type):
    return graph[node_type].code


def edges_codified(graph, edge_type):
    edges = np.zeros_like(graph[edge_type].edge_index, dtype=np.int64)
    edges[0, :] = graph[edge_type[0]].code[graph[edge_type].edge_index[0, :]]
    edges[1, :] = graph[edge_type[-1]].code[graph[edge_type].edge_index[1, :]]
    return edges


# noinspection PyTypeChecker
def make_subset(data, filter_transactions=None, filter_customers=None, filter_articles=None, inplace=False):
    """

    Args:
        data:
        filter_transactions: Must be mask
        filter_customers: Must be mask
        filter_articles: Must be mask

    Returns:

    """
    if not inplace:
        subdata = HeteroData()
    else:
        subdata = data

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

    # Index = new, Value = old index
    customers = np.unique(subdata[forward].edge_index[0])
    articles = np.unique(subdata[forward].edge_index[1])

    for k, v in data['u'].items():
        subdata['u'][k] = v[customers]
    for k, v in data['i'].items():
        subdata['i'][k] = v[articles]

    # Make the forward edges point to the smaller set of users/items
    subdata[forward].edge_index[0] = npi.indices(customers, subdata[forward].edge_index[0])
    subdata[forward].edge_index[1] = npi.indices(articles, subdata[forward].edge_index[1])

    for edge_type in data.edge_types:
        # Filtered edges are already dealt with in a different way
        if edge_type == forward:
            continue
        # Make the edges point to the smaller set of users/items
        # And remove edges that no longer point to valid nodes
        e0_ma = npi.indices(customers if edge_type[0] == 'u' else articles,
                            data[edge_type].edge_index[0], missing='mask')
        e1_ma = npi.indices(articles if edge_type[-1] == 'i' else customers,
                            data[edge_type].edge_index[1], missing='mask')
        all_present = ~e0_ma.mask & ~e1_ma.mask
        edge_index = np.zeros((2, np.sum(all_present)), dtype=np.int64)
        edge_index[0] = e0_ma.data[all_present]
        edge_index[1] = e1_ma.data[all_present]
        subdata[edge_type].edge_index = edge_index

        # Also update all the edge fields
        for k, v in data[edge_type].items():
            if k != 'edge_index':
                subdata[edge_type][k] = v[all_present]

    if not inplace:
        return subdata


def sample_neighbourhood(graph, roots, hops):
    edges = graph['u', 'b', 'i']
    return sample_neighbourhood_(graph, 'u', roots, roots, hops, np.zeros(edges.code.shape[0], dtype=bool), [])


# noinspection PyTypeChecker
def sample_neighbourhood_(graph, node_type, root_sources, root_targets, hops, mask, neighbours):
    if hops == 0:
        subgraph = copy.deepcopy(graph)
        subgraph['u', 'n', 'i'].edge_index = np.concatenate(neighbours[0::2], axis=1)
        subgraph['u', 'n', 'u'].edge_index = np.concatenate(neighbours[1::2], axis=1)
        make_subset(subgraph, filter_transactions=mask, inplace=True)
        return subgraph

    edges = graph['u', 'b', 'i']

    # & ~mask is to not backtrack
    dmask = np.isin(edges.edge_index[0 if node_type == 'u' else 1], root_targets) & ~mask

    # root_sources = ...
    # root_targets = root_targets

    sources = edges.edge_index[0 if node_type == 'u' else 1, dmask]  # Same as root_targets but bigger
    targets = edges.edge_index[1 if node_type == 'u' else 0, dmask]

    new_root_sources = root_sources[npi.indices(root_targets, sources)]
    new_neighbours = neighbours + [np.stack((new_root_sources, targets), axis=0)]

    return sample_neighbourhood_(graph, 'i' if node_type == 'u' else 'u', new_root_sources, targets, hops - 1,
                                 mask | dmask, new_neighbours)


def add_random_eval_edges(graph, true_edges, num_items, n):
    true_users = np.unique(true_edges[1, :])

    random_items = np.random.randint(0, num_items, size=(n))


class TemporalDataset(Dataset):
    # noinspection PyTypeChecker
    def __init__(self, graph, params):
        self.chunk_size = params.chunk_size
        self.embedding_chunks = params.embedding_chunks
        self.supervision_chunks = params.supervision_chunks
        self.val_splits = params.val_splits
        self.test_splits = params.test_splits
        self.skip_chunks = params.skip_chunks
        self.hops = params.hops

        self.graph = graph

        if params.negative_examples_ratio != 1.0:
            raise NotImplementedError()

        t = graph['u', 'b', 'i'].t

        chunk = np.floor_divide(t, self.chunk_size)
        timesteps = []
        for w in np.unique(chunk)[self.skip_chunks:]:
            subset = chunk == w  # ordering of transactions and edges on data is the same
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
        parser.add_argument('--skip_chunks', type=int, default=0, help="Skip the first n chunks")
        parser.add_argument('--negative_examples_ratio', type=float, default=1.0)
        parser.add_argument('--hops', type=int, default=3)

    # noinspection PyTypeChecker
    def make_split(self, chunks):

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

        # Supervised 'bought' in the supervised graph
        b = supervision_graph['u', 'b', 'i']

        # Pointer to the index of these nodes in embedding_graph, by this point it should be guaranteed that all
        # nodes in the supervision graph are also in the embedding graph
        supervision_graph['u'].ptr = npi.indices(embedding_graph['u'].code, supervision_graph['u'].code)
        supervision_graph['i'].ptr = npi.indices(embedding_graph['i'].code, supervision_graph['i'].code)
        b.ptr = np.stack(
            (supervision_graph['u'].ptr[b.edge_index[0]], supervision_graph['i'].ptr[b.edge_index[1]]),
            axis=0
        )

        # Fake edges added as negative examples
        fake_edges = np.zeros_like(b.ptr)
        # Use same users so the model focuses on predicting what users will buy
        # rather than whether they will buy anything or not
        fake_edges[0, :] = b.ptr[0, :]
        # Pick random items from all the items available in the graph.
        # TODO: Pick only from the local neighbourhood of each user
        fake_edges[1, :] = np.random.randint(embedding_graph['i'].code.shape[0], size=b.ptr.shape[1])
        # Remove fake edges that are the same as real edges to prevent contradicting supervisions signals
        fake_edges = fake_edges[:, ~npi.contains(b.ptr, fake_edges, axis=1)]

        # 'Supervised' bought in the embedding graph
        s = embedding_graph['u', 's', 'i']
        s.edge_index = np.concatenate([b.ptr, fake_edges], axis=1)

        s.label = np.ones(s.edge_index.shape[1], dtype=np.float)
        s.label[-fake_edges.shape[1]:] = 0

        # Resample embedding_graph to the local neighborhood of users in the supervision graph
        sampled_graph = sample_neighbourhood(embedding_graph, supervision_graph['u'].ptr, self.hops)

        add_random_eval_edges(embedding_graph, true_edges=b.ptr, num_items=self.graph['u'].code.shape[0], n=100)

        return sampled_graph

    def roll(self, timesteps):
        """ Do a rolling window over te time steps given and split into embedding graph and supervision graph"""
        zippers = list(
            timesteps[i:len(self.timesteps) - self.embedding_chunks + self.supervision_chunks + i] for i in
            range(self.embedding_chunks + self.supervision_chunks)
        )
        for chunks in zip(*zippers):
            yield self.make_split(chunks)

    def train_data(self):
        yield from self.roll(self.timesteps[:-self.test_splits - self.val_splits
        if self.test_splits + self.val_splits > 0 else len(self.timesteps)])

    def val_data(self):
        yield from self.roll(
            self.timesteps[
            -self.embedding_chunks - self.val_splits - self.test_splits:
            -self.test_splits if self.test_splits > 0 else len(self.timesteps)])

    def test_data(self):
        yield from self.roll(self.timesteps[-self.embedding_chunks - self.test_splits:])

    def train_data_len(self):
        return len(self.timesteps) - self.test_splits - self.val_splits - self.skip_chunks\
               - self.embedding_chunks - self.supervision_chunks + 1

    def __getitem__(self, idx):
        return self.timesteps[idx]


def add_transaction_order(data):
    """
    Adds the following edge attributes:

    Add attribute 'oui' which means a transaction is the oui'th transaction from the user.
    Add attribute 'oiu' which means a user is the oiu'th purchaser of the item
    """

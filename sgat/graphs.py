import copy

import numpy as np
import pandas as pd
import torch
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
    graph = HeteroData()

    for k in data.node_types + data.edge_types:
        for attribute, value in data[k].items():
            if attribute in ['edge_index', 'oui', 'oiu']:
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


def add_random_eval_edges(graph, true_edges, num_items, n, graph_item_codes):
    true_users = np.unique(true_edges[0, :])

    random_items = np.random.randint(0, num_items, size=(n))

    # remap true items to absolute codes since random items are also absolute codes
    true_edges[1, :] = graph_item_codes[true_edges[1, :]]

    # make every combination of true user and the random items
    eval_edges = [np.repeat(true_users, n),
                np.tile(random_items, len(true_users))]

    eval_u = np.hstack((eval_edges[0], true_edges[0]))
    eval_i = np.hstack((eval_edges[1], true_edges[1]))

    eval_and_true_edges = np.vstack((eval_u, eval_i))

    graph['u', 'eval', 'i'].edge_index = eval_and_true_edges # eval_and_true_edges


def add_oui_and_oiu(graph):
    """
    Adds the following edge attributes:

    Add attribute 'oui' which means a transaction is the oui'th transaction from the user.
    Add attribute 'oiu' which means a user is the oiu'th purchaser of the item
    """
    edges = graph[('u', 'b', 'i')].edge_index

    df = pd.DataFrame({'u':edges[0], 'i':edges[1]})

    # KLOPT NIKS VAN
    # sort in same way as pytorch will do
    df = df.sort_values(['u', 'i'])

    oui = df.groupby("u")['i'].rank("first")
    oiu = df.groupby("i")['u'].rank("first")

    graph[('u', 'b', 'i')].oui = oui.values
    graph[('u', 'b', 'i')].oiu = oiu.values
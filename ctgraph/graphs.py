import copy

import numpy as np
import pandas as pd
import torch
import numpy_indexed as npi

from ctgraph.datasets.ui_hetero_data import UIHeteroData
from ctgraph.np_utils import cumcount


def numpy_to_torch(data):
    """
    Convert numpy-graph to torch-graph

    Torch geometric always expects torch-graphs

    Args:
        data:

    Returns:

    """
    graph = UIHeteroData()

    for k in data.node_types + data.edge_types:
        # When converting an edge index to a sparse matrix, and coalescing, its indices get sorted
        # We need to do this ahead of time so we can also sort the corresponding metadata like the time
        if 'edge_index' in data[k].keys():
            argsorted_ind = np.lexsort((data[k]["edge_index"][1], data[k]["edge_index"][0]))
        else:
            argsorted_ind = None

        for attribute, value in data[k].items():
            # choose datatype
            if attribute in ['edge_index', 'oui', 'oiu', 'last_i', 'last_u']:
                dtype = torch.long
            elif type(value) is int:
                dtype= torch.long
            elif value.dtype == np.float:
                dtype = torch.float
            else:
                dtype = None

            # sort metadata and indices
            if argsorted_ind is not None:
                if value.ndim == 1:
                    value = value[argsorted_ind]
                elif value.ndim == 2:
                    value = value[:, argsorted_ind]
                else:
                    raise NotImplementedError()

            # save converted data
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
        subdata = UIHeteroData()
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



def add_random_eval_edges(graph, num_items, true_u_index=None, true_i_code=None, n=100):
    if true_u_index is None or true_i_code is None:
        assert true_u_index is None and true_i_code is None
        true_edges = graph['u', 's', 'i'].edge_index[:, graph['u', 's', 'i'].label == 1]
        true_u_index = true_edges[0]
        true_i_code = graph['i'].code[true_edges[0]]

    true_users = np.unique(true_u_index)

    random_item_codes = np.random.randint(0, num_items, size=(n,))

    # make every combination of true user and the random items
    eval_edges = [np.repeat(true_users, n), np.tile(random_item_codes, len(true_users))]

    # Put the true items after the random items
    eval_u = np.hstack((eval_edges[0], true_u_index))
    eval_i_codes = np.hstack((eval_edges[1], true_i_code))

    label = np.zeros(eval_i_codes.shape[0])
    label[n:] = 1

    graph['eval'].u_index = eval_u
    graph['eval'].i_code = eval_i_codes
    graph['eval'].label = label

def add_oui_and_oiu(graph):
    # oiu is the order of u−i interaction, that is,
    # the position of item i in all items that the u has interacted with

    # oui refers to the order of u in all user nodes that
    # have interacted with item i
    edges = graph[('u', 'b', 'i')].edge_index
    edges_t = graph[('u', 'b', 'i')].t
    # prepare arrays
    oui = np.zeros_like(edges_t)
    oiu = np.zeros_like(edges_t)

    # sort by time
    trans_order = np.argsort(edges_t)

    sorted_time = edges_t[:][trans_order]

    # oui = user's xth transaction, so the cumcount of that users occurence
    sorted_users = edges[0, :][trans_order]
    oui[trans_order] = cumcount(sorted_users) + 1

    # oiu = item's xth transaction, so the cumcount of that items occurence
    sorted_items = edges[1, :][trans_order]
    oiu[trans_order] = cumcount(sorted_items) + 1

    graph[('u', 'b', 'i')].oui = oui
    graph[('u', 'b', 'i')].oiu = oiu


def add_relative_time(graph):
    df = pd.DataFrame({'u': graph['u', 'b', 'i'].edge_index[0], 't': graph['u', 'b', 'i'].t})
    groupby = df.groupby('u').max()

    graph['u'].t_max = groupby.sort_values('u')['t'].values

    df = pd.DataFrame({'i': graph['u', 'b', 'i'].edge_index[1], 't': graph['u', 'b', 'i'].t})
    groupby = df.groupby('i').max()

    graph['i'].t_max = groupby.sort_values('i')['t'].values

    df = pd.DataFrame({'u': graph['u', 'b', 'i'].edge_index[0], 't': graph['u', 'b', 'i'].t})
    groupby = df.groupby('u').min()

    graph['u'].t_min = groupby.sort_values('u')['t'].values

    df = pd.DataFrame({'i': graph['u', 'b', 'i'].edge_index[1], 't': graph['u', 'b', 'i'].t})
    groupby = df.groupby('i').min()

    graph['i'].t_min = groupby.sort_values('i')['t'].values

    assert np.all(graph['u'].t_max >= graph['u'].t_min)
    assert np.all(graph['i'].t_max >= graph['i'].t_min)


def add_last(graph):
    edges = graph[('u', 'b', 'i')].edge_index
    edges_t = graph[('u', 'b', 'i')].t

    # order edges on time t
    trans_order = np.argsort(edges_t)

    # sorted users and items on time
    sorted_time = edges_t[:][trans_order]
    sorted_users = edges[0, :][trans_order]
    sorted_items = edges[1, :][trans_order]

    # get last index of elements by reversing index of unique elements
    unique_users_indexes = sorted_items[sorted_users.shape[0] - 1 - np.unique(sorted_users[::-1], return_index=True)[1]]
    unique_items_indexes = sorted_users[sorted_items.shape[0] - 1 - np.unique(sorted_items[::-1], return_index=True)[1]]

    # make array with unique users and last item and vice versa
    graph['last_u'].u_code = graph['u'].code
    graph['last_u'].i_code = graph['i'].code[unique_users_indexes]
    graph['last_u'].t_code = sorted_time[sorted_users.shape[0] - 1 - np.unique(sorted_users[::-1], return_index=True)[1]]

    graph['last_i'].u_code = graph['u'].code[unique_items_indexes]
    graph['last_i'].i_code = graph['i'].code
    graph['last_i'].t_code = sorted_time[sorted_items.shape[0] - 1 - np.unique(sorted_items[::-1], return_index=True)[1]]
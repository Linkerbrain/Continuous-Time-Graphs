import numpy as np
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_hetero_data

import numpy_indexed as npi

from ctgraph.data import amazon_dataset
from ctgraph.graphs import sample_neighbourhood, compare_graphs, edges_codified, check_graph, add_last, add_oui_and_oiu

GRAPH = amazon_dataset('beauty')


def test_sample_neighbourhood():
    roots, hops = np.arange(7), 4
    sampled = sample_neighbourhood(GRAPH, roots, hops)
    resampled = sample_neighbourhood(sampled, roots, hops)
    assert compare_graphs(sampled, resampled)
    assert (sampled['u', 'n', 'i'].edge_index.shape[1] + sampled['u', 'n', 'u'].edge_index.shape[1]) == (
        sampled['u', 'b', 'i'].edge_index.shape[1])

    sampled = sample_neighbourhood(GRAPH, np.arange(1), hops)
    assert (sampled['u', 'n', 'i'].edge_index.shape[1] + sampled['u', 'n', 'u'].edge_index.shape[1]) == (
        sampled['u', 'b', 'i'].edge_index.shape[1])

    sampled2 = sample_neighbourhood(GRAPH, roots, 2)
    assert npi.contains(edges_codified(resampled, ('u', 'b', 'i')),
                        edges_codified(sampled2, ('u', 'b', 'i')), axis=1).all()
    assert npi.contains(edges_codified(resampled, ('u', 'n', 'i')),
                        edges_codified(sampled2, ('u', 'n', 'i')), axis=1).all()
    assert npi.contains(edges_codified(resampled, ('u', 'n', 'u')),
                        edges_codified(sampled2, ('u', 'n', 'u')), axis=1).all()

    assert npi.contains(resampled['u'].code, sampled2['u'].code).all()
    assert npi.contains(resampled['i'].code, sampled2['i'].code).all()

    assert check_graph(sampled)
    assert check_graph(resampled)
    assert check_graph(sampled2)


def test_last_and_oui_oiu():
    # checking add__last
    # check graph
    graph = GRAPH
    print(graph)
    add_oui_and_oiu(graph)
    add_last(graph)
    assert check_graph(graph)


    edges = graph[('u', 'b', 'i')].edge_index
    edges_t = graph[('u', 'b', 'i')].t

    # order edges on time t
    trans_order = np.argsort(edges_t)

    # sorted users and items on time
    sorted_time = edges_t[:][trans_order]
    sorted_users = edges[0, :][trans_order]
    sorted_items = edges[1, :][trans_order]
    # check shapes

    assert graph['last_u'].u_code.shape == np.unique(sorted_users).shape and graph['last_i'].u_code.shape == np.unique(sorted_items).shape
    assert graph['last_u'].i_code.shape == np.unique(sorted_users).shape and graph['last_i'].i_code.shape == np.unique(sorted_items).shape
    assert graph['last_u'].t_code.shape == np.unique(sorted_users).shape and graph['last_i'].t_code.shape == np.unique(sorted_items).shape

    # check if index of assigned element is last
    random_user_number = np.random.randint(0, np.unique(sorted_users).shape[0])
    random_item_number = np.random.randint(0, np.unique(sorted_items).shape[0])
    assert np.where(sorted_users==sorted_users[random_user_number])[0][-1] == sorted_users.shape[0] - 1 - np.unique(sorted_users[::-1], return_index=True)[1][sorted_users[random_user_number]]
    assert np.where(sorted_items==sorted_items[random_item_number])[0][-1] == sorted_items.shape[0] - 1 - np.unique(sorted_items[::-1], return_index=True)[1][sorted_items[random_item_number]]

    # check if time is correct of last elements
    random_user_time = sorted_time[np.where(sorted_users == sorted_users[random_user_number])[0]]
    random_item_time = sorted_time[np.where(sorted_items==sorted_items[random_item_number])[0]]
    assert np.all(random_user_time[:-1] <= random_user_time[1:])
    assert np.all(random_item_time[:-1] <= random_item_time[1:])

    # --------

    # checking add_oui_and_oiu

    random_user = np.random.randint(0, np.unique(sorted_users).shape[0])
    random_item = np.random.randint(0, np.unique(sorted_items).shape[0])
    random_item_loc = np.where(edges[1, :] == random_item)
    random_user_loc = np.where(edges[0, :] == random_user)
    oui = graph[('u', 'b', 'i')].oui
    oiu = graph[('u', 'b', 'i')].oiu
    edges_t = graph[('u', 'b', 'i')].t

    # check graph
    assert check_graph(graph)

    # check shapes
    assert oui.shape == edges_t.shape and oiu.shape == edges_t.shape

    # check order
    assert np.all(np.arange(0, random_item_loc[0].shape[0]) + 1 == oiu[random_item_loc])
    assert np.all(np.arange(0, random_user_loc[0].shape[0]) + 1 == oui[random_user_loc])



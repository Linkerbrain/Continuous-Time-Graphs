import numpy as np
from torch_geometric.loader.neighbor_loader import NeighborSampler
from torch_geometric.loader.utils import filter_hetero_data

import numpy_indexed as npi

from sgat.data import amazon_dataset
from sgat.graphs import sample_neighbourhood, compare_graphs, edges_codified

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

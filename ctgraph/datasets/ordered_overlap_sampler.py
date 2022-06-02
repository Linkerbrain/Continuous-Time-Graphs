"""
TODO

Sampler that, for a given static graph, samples several subsets in an order such that central nodes in the previous
batch are outer edges in the next, so that rich memorized embeddings are availbale for outer edges during the mayority
of training.

"""


import numpy as np
import numpy_indexed as npi
from torch.utils.data import Dataset

from ctgraph.graphs import make_subset, sample_neighbourhood, add_random_eval_edges


class PeriodicDataset(Dataset):
    """
    Deprecated
    """
    # noinspection PyTypeChecker
    def __init__(self, graph, params):
        self.chunk_size = params.chunk_size
        self.embedding_chunks = params.embedding_chunks
        self.supervision_chunks = params.supervision_chunks
        self.val_splits = params.val_splits
        self.test_splits = params.test_splits
        self.skip_chunks = params.skip_chunks
        self.hops = params.hops
        self.max_target_users = params.max_target_users

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
        parser.add_argument('--max_target_users', type=int, default=None)
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

        # Add random edges
        add_random_eval_edges(sampled_graph, num_items=self.graph['i'].code.shape[0])

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
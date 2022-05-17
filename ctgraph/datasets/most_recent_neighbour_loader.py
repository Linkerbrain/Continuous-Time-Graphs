import pandas as pd

from ctgraph.datasets.subset_loader import SubsetLoader
from ctgraph.datasets.recent_sampler import RecentSampler
from ctgraph.datasets.paper_sampler import PaperSampler

from torch_geometric.loader import DataLoader
from ctgraph import logger

from tqdm.auto import tqdm

class MostRecentNeighbourLoader(SubsetLoader):
    """
    MostRecentNeighbourLoader samples and loads the most recent m-hop neighbourhood of each user
     uses the RecentSampler to sample the most recent transactions of each user
     uses the SubsetLoader to convert to data objects
    """
    @staticmethod
    def add_args(parser):
        parser.add_argument('--n_max_trans', type=int, default=20)
        parser.add_argument('--m_order', type=int, default=2)
        parser.add_argument('--num_users', type=int, default=None)
        parser.add_argument('--newsampler', action='store_true')
        parser.add_argument('--sample_all', action='store_true')

    def __init__(self, graph, params, *args, **kwargs):
        # Initiate subset dataset which sorts the dataset (self.ordered_trans)
        super().__init__(graph, params, *args, **kwargs)

        if params.num_users is None:
            num_users = graph['u'].code.shape[0]
        else:
            num_users = params.num_users

        
        if params.newsampler:
            self.rs = PaperSampler(self.ordered_trans, self.ordered_trans_t, n=params.n_max_trans, m=params.m_order, sample_all=params.sample_all)
        else:
            self.rs = RecentSampler(self.ordered_trans, self.ordered_trans_t, n=params.n_max_trans, m=params.m_order)

        self._prepare_idxs(num_users)

    """
    Neighbour sampling
    """

    def _prepare_idxs(self, num_users):
        logger.info(f'[NeighbourDataset] Limiting training/validation/test data to only {num_users} users.')
        users = range(num_users) # DEBUG LIMIT
        
        self.train_idx = []
        self.val_idx = []
        self.test_idx = []

        fails = 0
        for u in tqdm(users):
            sample = self.rs.neighbour_idx_of(u)

            if not sample["valid"]:
                fails +=1
                continue

            for x, y in zip(sample['x_train'], sample['y_train']):
                self.train_idx.append((x, y))
            for x, y in zip(sample['x_val'], sample['y_val']):
                self.val_idx.append((x, y))
            for x, y in zip(sample['x_test'], sample['y_test']):
                self.test_idx.append((x, y))

        logger.info(f'[NeighbourDataset] Skipped {fails}/{num_users} users due to them not having enough transactions.')

    """
    Generators:
    """

    def yield_train(self):
        for x_idx, y_idx in self.train_idx:
            yield self.create_subgraph(x_idx, y_idx)

    def yield_val(self):
        for x_idx, y_idx in self.val_idx:
            yield self.create_subgraph(x_idx, y_idx)

    def yield_test(self):
        for x_idx, y_idx in self.test_idx:
            yield self.create_subgraph(x_idx, y_idx)

    def make_val_dataloader(self, batch_size=8, shuffle=True):
        # create batches
        self.val_data = []

        # iterate over sampled indices
        for x_idx, y_idx in self.val_idx:
            # create datapoint
            user_graph_data = self.create_subgraph(x_idx, y_idx)

            self.val_data.append(user_graph_data)

        # it is strongly recommended to turn off shuffling for val/test dataloaders
        self.val_loader = DataLoader(self.val_data, batch_size=batch_size, shuffle=False, num_workers=0) #, num_workers=12)

        return self.val_loader

    def make_test_dataloader(self, batch_size=8, shuffle=True):
        # create batches
        self.test_data = []

        # iterate over sampled indices
        for x_idx, y_idx in self.test_idx:
            # create datapoint
            user_graph_data = self.create_subgraph(x_idx, y_idx)

            self.test_data.append(user_graph_data)

        # it is strongly recommended to turn off shuffling for val/test dataloaders
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False, num_workers=0) #, num_workers=12)

        return self.test_loader

    def train_data_len(self):
        return len(self.train_data)

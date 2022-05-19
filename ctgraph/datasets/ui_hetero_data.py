from torch_geometric.data import HeteroData

class UIHeteroData(HeteroData):
    """
    HeteroData class that properly updates u and i indices when batching
    """
    def __init__(self):
        super().__init__()
        
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'u_index':
            return self['u'].num_nodes
        if key == 'i_index':
            return self['u'].num_nodes
        return super().__inc__(key, value, *args, **kwargs)
import numpy as np
import pandas as pd
import torch
from os import path
from tqdm.notebook import tqdm
from torch_geometric.loader import DataLoader
import numpy_indexed as npi

"""
Yes, this is ugly code but it is only to verify real quick if the pretty code did its job
"""


# load in data to be investigated

target_dir = "../precomputed_data"
target_file = "beauty_neighbour_n50_m1_numuser2500_newsampled_sampleall"

loc = path.join(target_dir, target_file)
dataset = torch.load(loc)
dataset.neptune_logger = None

graph, train_data, val_data, test_data = dataset.graph, dataset.train_data, dataset.val_data, dataset.test_data

train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

print("Loaded dataset", dataset)
print("graph:", graph)

print("train_data has ", len(train_data), "graphs")
print("val_data has ", len(val_data), "graphs")
print("test_data has ", len(test_data), "graphs")

# load real data
df = pd.read_csv("../amazon/Beauty.csv").astype(int)
df['ui'] = df.apply(lambda x: str(x['user_id'])+","+str(x['item_id']), axis=1)




# test 1 check if graph is correct
sample_size = 1000

samples = np.random.randint(low=0, high=len(graph[('u', 'b', 'i')].edge_index[0]), size=(sample_size,))

passed = True
for row in tqdm(graph[('u', 'b', 'i')].edge_index[:, samples].T):
    u = row[0]
    i = row[1]
    
    ui = str(u) + ',' + str(i)
    
    if ui not in df['ui'].values:
        passed = False
        print(ui)
        break
        
if not passed:
    raise AssertionError("A transaction of the graph object is not present in the original data")

print("PASSED TEST 1, GRAPH IS CORRECT")

# test 2 check if target trans are correct (and batches are okay)
test_2_limit = 100

passed = True
for loader in [train_dataloader, val_dataloader, test_dataloader]:
    print("working on", loader)
    for i, batch in enumerate(loader):
        if i > test_2_limit:
            break
            
        u_target_codes = batch['u'].code[batch['target']['u_index']]
        i_target_codes = batch['target']['i_code']

        for u, i in zip(u_target_codes, i_target_codes):
            trans = str(int(u))+","+str(int(i))
            if ui not in df['ui'].values:
                passed = False
                print(loader, ui)
                break

if not passed:
    raise AssertionError("A target transaction of the loader object is not present in the original data")

print("PASSED TEST 2, TARGETS ARE CORRECT")

# test 3 check if neighbourhoods are legit
test_3_limit = 1000

def sample_neighbourhood_(edges, node_type, root_sources, root_targets, hops, mask):
    # 
    if hops == 0:
        return mask

    # & ~mask is to not backtrack
    dmask = np.isin(edges[0 if node_type == 'u' else 1], root_targets) & ~mask

    # root_sources = ...
    # root_targets = root_targets

    sources = edges[0 if node_type == 'u' else 1, dmask]  # Same as root_targets but bigger
    targets = edges[1 if node_type == 'u' else 0, dmask]

    new_root_sources = root_sources[npi.indices(root_targets, sources)]

    return sample_neighbourhood_(edges, 'i' if node_type == 'u' else 'u', new_root_sources, targets, hops - 1,
                                 mask | dmask)

def sample_neighbourhood(edges, roots, hops):
    mask = sample_neighbourhood_(edges, 'u', roots, roots, hops, np.zeros(edges.shape[1], dtype=bool))
    return np.where(mask)[0]

passed = True
for dat in [train_data, val_data, test_data]:
    print("working!!")
    sample = np.random.randint(0, len(dat), test_3_limit)
    for i, s in enumerate(sample):
        a = dat[s]
        
        if i > test_3_limit:
            break
            
        t_idx = sample_neighbourhood(a[('u', 'b', 'i')].edge_index, np.array([0]), 10)
        
        if not len(t_idx) == len(a[('u', 'b', 'i')].edge_index[0]):
            print(t_idx)
            print(a)
            print("Not all nodes are connected for", a)
            passed = False
            break

if not passed:
    raise AssertionError("Not all nodes are connected in the sampled graph")

print("PASSED TEST 3, SAMPLED NEIGHBOURHOODS ARE CONNECTED")
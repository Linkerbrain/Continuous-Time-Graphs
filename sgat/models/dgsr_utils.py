import torch

def sparse_dense_mul(s, d):
    """
    elementwise multiply sparse and dense matrix of same size

    Parameters:
        s: sparse matrix
        d: dense matrix
    """
    i = s._indices()
    v = s._values()
    dv = d[i[0,:], i[1,:]]  # get values from relevant entries of dense matrix
    return torch.sparse_coo_tensor(i, v * dv, s.size())

def pass_messages(messages, adjacency, pVui):
    """
    add messages together based on adjacency matrix

    Parameters:
        messages: tensor (i, h)
        adjacency: sparse tensor (u, i)
        pVui: tensor (t, h)
    """
    # parse adjacency matrix
    user_per_trans, item_per_trans = adjacency._indices()
    alpha = adjacency._values().unsqueeze(-1)

    # prepare output
    output = torch.zeros((adjacency.shape[0], messages.shape[1]), dtype=torch.float).to(adjacency.device)

    # add messages
    output.index_add_(0, user_per_trans, messages[item_per_trans] * alpha)

    # add embeddings
    output.index_add_(0, user_per_trans, pVui * alpha)

    # TODO: also scale by edge weight ?

    return output
def pass_messages_no_possitional(messages, adjacency):
    """
    add messages together based on adjacency matrix

    Parameters:
        messages: tensor (i, h)
        adjacency: sparse tensor (u, i)
    """
    # parse adjacency matrix
    user_per_trans, item_per_trans = adjacency._indices()
    alpha = adjacency._values().unsqueeze(-1)

    # prepare output
    output = torch.zeros((adjacency.shape[0], messages.shape[1]), dtype=float)

    # add messages
    output.index_add_(0, user_per_trans, messages[item_per_trans] * alpha)

    # add embeddings
    output.index_add_(0, user_per_trans, alpha)

    return output

def relative_order(oui, by_who, n=10):
    """
    Parameters:
        oui (or oiu) : tensor (t) (the order #'s of each transaction)
        by_who : tensor (t) (the one to count the # of neighbours from)
        n : int max neighbourhood
    """
    # compute amount of transactions of each user
    neighbourhood_sizes = torch.bincount(by_who)
    neighbourhood_sizes_per_trans = neighbourhood_sizes[by_who]

    # relative_order = Neighbourhood_size - order (zodat nieuwste altijd hetzelfde hebben)
    rui = torch.clip(neighbourhood_sizes_per_trans, max=n) - torch.clip(oui, max=n)

    return rui

def get_last(by_who, what, code):

    # compute amount of transactions of each user
    neighbourhood_sizes = torch.bincount(by_who)

    # compute cumulative indices of user
    cum_ind = torch.cumsum(neighbourhood_sizes, dim=0) - 1

    # select indices out of preferred transactions
    last_indices = torch.index_select(what, 0, cum_ind)

    # get item id's from graph
    last_ids = torch.index_select(torch.Tensor(code), 0, last_indices)
    return last_ids

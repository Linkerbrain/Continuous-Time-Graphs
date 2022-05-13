import itertools

import numpy as np

from tqdm.auto import tqdm

def mean_average_precision(y_true, y_pred, k=12):
    """
    Fast compute Mean Average Precision
    Args:
        y_true: list of lists, np matrix, array of lists... etc
        y_pred: same format as y_true
        k:

    Returns:

    """
    # compute the Rel@K for all items
    rel_at_k = np.zeros((len(y_true), k), dtype=int)

    # collect the intersection indexes (for the ranking vector) for all pairs
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        _, _, inter_idxs = np.intersect1d(truth, pred[:k], return_indices=True)
        rel_at_k[idx, inter_idxs] = 1

    # Calculate the intersection counts for all pairs
    intersection_count_at_k = rel_at_k.cumsum(axis=1)

    # we have the same denominator for all ranking vectors
    ranks = np.arange(1, k + 1, 1)

    # Calculating the Precision@K for all Ks for all pairs
    precisions_at_k = intersection_count_at_k / ranks
    # Multiply with the Rel@K for all pairs
    precisions_at_k = precisions_at_k * rel_at_k

    # Calculate the average precisions @ K for all pairs
    average_precisions_at_k = precisions_at_k.mean(axis=1)

    # calculate the final MAP@K
    map_at_k = average_precisions_at_k.mean()

    return map_at_k


def compute_eval_metrics(all_ranks, users):
    """
    Copy-paste from the DGSR code (DGSR_utils.py) with edits to work in our framework.

    Their code calculates the dcg but calls it ndgg for some reason.

    Args:
        all_top:
        random_rank:

    Returns:

    """
    recall5, recall10, recall20, dcg5, dcg10, dcg20 = [], [], [], [], [], []
    recall5_tmp, recall10_tmp, recall20_tmp, dcg5_tmp, dcg10_tmp, dcg20_tmp = [], [], [], [], [], []
    last_u = users[0]
    for u, rank in tqdm(itertools.chain(zip(users, all_ranks), [(None, None)]), total=len(users)+1):
        if u != last_u:
            recall5.append(sum(recall5_tmp)/len(recall5_tmp))
            recall10.append(sum(recall10_tmp)/len(recall10_tmp))
            recall20.append(sum(recall20_tmp)/len(recall20_tmp))
            dcg5.append(sum(dcg5_tmp))
            dcg10.append(sum(dcg10_tmp))
            dcg20.append(sum(dcg20_tmp))
            recall5_tmp, recall10_tmp, recall20_tmp, dcg5_tmp, dcg10_tmp, dcg20_tmp = [], [], [], [], [], []
        last_u = u

        if u is None and rank is None:
            break

        if rank < 20:
            dcg20_tmp.append(1 / np.log2(rank + 2))
            recall20_tmp.append(1)
        else:
            dcg20_tmp.append(0)
            recall20_tmp.append(0)
        if rank < 10:
            dcg10_tmp.append(1 / np.log2(rank + 2))
            recall10_tmp.append(1)
        else:
            dcg10_tmp.append(0)
            recall10_tmp.append(0)
        if rank < 5:
            dcg5_tmp.append(1 / np.log2(rank + 2))
            recall5_tmp.append(1)
        else:
            dcg5_tmp.append(0)
            recall5_tmp.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(dcg5), np.mean(dcg10), np.mean(dcg20)

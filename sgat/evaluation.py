import numpy as np


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


def compute_eval_metrics(all_ranks):
    """
    Copy-paste from the DGSR code (DGSR_utils.py) with some edits to work in our framework.

    Their code calculates the dcg but calls it ndgg for some reason.

    Args:
        all_top:
        random_rank:

    Returns:

    """
    recall5, recall10, recall20, dcg5, dcg10, dcg20 = [], [], [], [], [], []
    for rank in all_ranks:
        if rank < 20:
            dcg20.append(1 / np.log2(rank + 2))
            recall20.append(1)
        else:
            dcg20.append(0)
            recall20.append(0)
        if rank < 10:
            dcg10.append(1 / np.log2(rank + 2))
            recall10.append(1)
        else:
            dcg10.append(0)
            recall10.append(0)
        if rank < 5:
            dcg5.append(1 / np.log2(rank + 2))
            recall5.append(1)
        else:
            dcg5.append(0)
            recall5.append(0)
    return np.mean(recall5), np.mean(recall10), np.mean(recall20), np.mean(dcg5), np.mean(dcg10), np.mean(dcg20)

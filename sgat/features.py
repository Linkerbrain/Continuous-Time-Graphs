import numpy as np
import pandas as pd


def bin_column(column, merge_threshold, verbose=False):
    """
    Group the least frequent values in a column into a single value.
    :param column: Pandas Series
    :param merge_threshold: Lower bound on the ratio of datapoints that must retain their original value.
    :param verbose: Print stuff
    :return: Numpy array of integer representations of the categories.
    """
    counts = column.value_counts()
    if verbose: print("Categories:", len(counts))

    cs = np.cumsum(counts)
    merge_mask = (cs / cs.iloc[-1]) > merge_threshold
    merge_categories = counts.index[merge_mask]
    if verbose: print("Kept categories:", len(counts.index[~merge_mask]),
                      f"({counts.index[~merge_mask].to_list()})")
    if verbose: print("Merged categories:", len(merge_categories), f"({merge_categories.to_list()})")

    column_merged = column.astype('category').cat.remove_categories(merge_categories)

    binned = column_merged.cat.codes.to_numpy().copy()
    binned[binned < 0] = np.max(binned) + 1

    if verbose: print("New total:", np.max(binned) + 1)
    return binned


def one_hot_array(array):
    categories = np.max(array) + 1
    one_hot = np.zeros((array.shape[0], categories))
    one_hot[np.arange(array.shape[0]), array] = 1
    return one_hot


def one_hot_concat(dataframe: pd.DataFrame, categorical, scalar, index=None, normalizer=lambda x: x / np.max(x),
                   merge_threshold=1, verbose=False):
    columns = []
    for column_name in categorical:
        if verbose: print("\nColumn:", column_name, "(categorical)")
        binned = bin_column(dataframe[column_name], merge_threshold, verbose=verbose)
        one_hot = one_hot_array(binned)
        columns.append(one_hot)

    for column_name in scalar:
        if verbose: print("\nColumn:", column_name, "(scalar)")
        if verbose: print("Mean, std before normalization:", dataframe[column_name].mean(),
                          dataframe[column_name].std())
        numeric = pd.to_numeric(dataframe[column_name], errors='coerce')
        # Pandas .mean() automatically ignores NaN's
        imputed = numeric.fillna(numeric.mean())
        normalized = normalizer(imputed.to_numpy())
        if verbose: print("Mean, std after normalization:", normalized.mean(), normalized.std())
        columns.append(normalized[:, None])

    idx = dataframe[index] if index is not None else dataframe.index

    matrix = np.concatenate(columns, axis=1)

    return pd.DataFrame(matrix, index=idx)

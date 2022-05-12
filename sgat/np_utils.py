import numpy as np

"""
Cumcount returns how many times each element has appeared in an array so far (from left to right)
https://stackoverflow.com/questions/40602269/how-to-use-numpy-to-get-the-cumulative-count-by-unique-values-in-linear-time
"""

def dfill(a):
    n = a.size
    b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
    return np.arange(n)[b[:-1]].repeat(np.diff(b))

def argunsort(s):
    n = s.size
    u = np.empty(n, dtype=np.int64)
    u[s] = np.arange(n)
    return u

def cumcount(a):
    n = a.size
    s = a.argsort(kind='mergesort')
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]
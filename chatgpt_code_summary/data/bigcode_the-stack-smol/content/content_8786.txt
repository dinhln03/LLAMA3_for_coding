from typing import List

import numpy as np


def mask_nan(arrays: List[np.ndarray]) -> List[np.ndarray]:
    """
    Drop indices from equal-sized arrays if the element at that index is NaN in
    any of the input arrays.

    Parameters
    ----------
    arrays : List[np.ndarray]
        list of ndarrays containing NaNs, to be masked

    Returns
    -------
    List[np.ndarray]
        masked arrays (free of NaNs)

    Notes
    -----
    This function find the indices where one or more elements is NaN in one or
    more of the input arrays, then drops those indices from all arrays.
    For example:
    >> a = np.array([0, 1, np.nan, 3])
    >> b = np.array([np.nan, 5, np.nan, 7])
    >> c = np.array([8, 9, 10, 11])
    >> mask_nan([a, b, c])
    [array([ 1.,  3.]), array([ 5.,  7.]), array([ 9, 11])]

    """
    n = arrays[0].size
    assert all(a.size == n for a in arrays[1:])
    mask = np.array([False] * n)
    for arr in arrays:
        mask = np.logical_or(mask, np.isnan(arr))
    return [arr[np.where(~mask)[0]] for arr in arrays]

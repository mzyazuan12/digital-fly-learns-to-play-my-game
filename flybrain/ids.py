"""Integer body-ID handling.

Biological IDs must never pass through float64. JavaScript and pandas object
dtypes are fine; IEEE floats are not.
"""

from __future__ import annotations

import numpy as np


def exact_ids(values) -> np.ndarray:
    """Return nonnegative uint64 IDs. Reject floats."""
    items = np.asarray(values)
    if items.dtype.kind == "f":
        raise ValueError("Neuron IDs must be integers or decimal strings, never floats.")
    if items.dtype.kind in "iu":
        if np.any(items < 0):
            raise ValueError("Neuron IDs cannot be negative.")
        return items.astype(np.uint64, copy=False)
    text = [str(value) for value in items]
    if any(not value.isascii() or not value.isdecimal() for value in text):
        raise ValueError("Neuron IDs must be nonnegative decimal integers.")
    return np.asarray(text, dtype=np.uint64)


def index_edges(ids: np.ndarray, pre, post, counts):
    """Map body IDs onto retained indices. Return kept edges plus a keep mask."""
    if ids.size == 0 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Node IDs must be nonempty, unique, and sorted.")
    pre_ids, post_ids = exact_ids(pre), exact_ids(post)
    weights = np.asarray(counts)
    if not (len(pre_ids) == len(post_ids) == len(weights)):
        raise ValueError("Edge columns have different lengths.")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 1)
        or np.any(weights != np.floor(weights))
        or np.any(weights > np.uint32(2**32 - 1))
    ):
        raise ValueError("Synapse counts must be positive uint32-compatible integers.")
    i = np.searchsorted(ids, pre_ids)
    j = np.searchsorted(ids, post_ids)
    n = len(ids)
    keep = (i < n) & (j < n)
    keep &= ids[np.minimum(i, n - 1)] == pre_ids
    keep &= ids[np.minimum(j, n - 1)] == post_ids
    return (
        i[keep].astype(np.uint32, copy=False),
        j[keep].astype(np.uint32, copy=False),
        weights[keep].astype(np.uint32, copy=False),
        keep,
    )

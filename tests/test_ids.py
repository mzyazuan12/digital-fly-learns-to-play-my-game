from flybrain.ids import exact_ids, index_edges
import numpy as np
import pytest


def test_exact_ids_rejects_floats():
    with pytest.raises(ValueError):
        exact_ids(np.array([1.0, 2.0]))


def test_exact_ids_accepts_uint():
    got = exact_ids([10, 20, 30])
    assert got.dtype == np.uint64
    assert list(got) == [10, 20, 30]


def test_index_edges_drops_unknown_endpoints():
    ids = np.array([10, 20, 30], dtype=np.uint64)
    i, j, w, keep = index_edges(
        ids,
        pre=[10, 10, 99],
        post=[20, 40, 30],
        counts=[3, 4, 5],
    )
    assert list(i) == [0]
    assert list(j) == [1]
    assert list(w) == [3]
    assert keep.sum() == 1

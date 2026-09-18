"""Full MaleCNS load test. Skipped unless the Feather files and cache exist."""

from pathlib import Path

import pytest

from flybrain.loader import DEFAULT_DATA, EXPECTED_RETAINED, EDGE_FILE, load_connectome

pytestmark = pytest.mark.full


def test_full_graph_neuron_count():
    if not (DEFAULT_DATA / EDGE_FILE).exists():
        pytest.skip("MaleCNS Feather files are not in data/malecns_v1")
    g = load_connectome(progress=False)
    assert g.n == EXPECTED_RETAINED
    assert g.n_edges > 1_000_000
    assert g.pre_ptr.shape == (g.n + 1,)

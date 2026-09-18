import numpy as np

from flybrain.loader import synthetic_connectome
from flybrain.network import LIFNetwork, LIFParams
from flybrain.plasticity import RewardModulatedPlasticity


def test_synthetic_graph_is_sparse_and_small():
    g = synthetic_connectome(n=32, extra_edges=10, seed=1)
    assert g.n == 32
    assert g.n_edges < 32 * 32
    assert g.anatomical.dtype == np.uint32
    assert g.pre_ptr[-1] == g.n_edges


def test_stimulation_propagates_along_chain():
    g = synthetic_connectome(n=24, extra_edges=0, seed=0)
    net = LIFNetwork(g, params=LIFParams(dt=0.1), seed=0)
    net.inject([0], 40.0)
    counts = net.run_ms(20.0)
    assert counts[0] > 0
    # Neuron 1 is the unique anatomical target of neuron 0 in the chain.
    assert counts[1] > 0
    assert int(counts.sum()) >= 2


def test_reset_clears_membrane_not_anatomy():
    g = synthetic_connectome(n=16, extra_edges=4, seed=2)
    net = LIFNetwork(g, seed=0)
    net.inject([0], 40.0)
    net.run_ms(10)
    edges_before = net.connectome.n_edges
    net.reset()
    assert np.allclose(net.v, net.params.v_rest)
    assert net.connectome.n_edges == edges_before
    assert np.allclose(net.efficacy, 1.0)


def test_plasticity_does_not_add_edges():
    g = synthetic_connectome(n=20, extra_edges=8, seed=3)
    net = LIFNetwork(g, seed=0)
    rule = RewardModulatedPlasticity(net)
    net.inject([0], 40.0)
    net.run_ms(15)
    rule.observe_activity()
    n_before = net.connectome.n_edges
    rule.apply_reward(1.0)
    assert net.connectome.n_edges == n_before
    assert net.efficacy.shape[0] == n_before

import numpy as np

from flybrain.mushroom_body import MushroomBodyLearning
from flybrain.network import LIFNetwork, LIFParams
from organism.toy import miniature_connectome


def test_mushroom_body_finds_kc_dan_mbon():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    mb = MushroomBodyLearning(net)
    assert mb.kenyon.size >= 1
    assert mb.mbon.size >= 1
    assert mb.dan.size >= 1
    assert mb.edge_mask.sum() >= 1
    assert not mb.notes["missing_circuit"]


def test_learning_does_not_overwrite_anatomy():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    mb = MushroomBodyLearning(net)
    anatomy = net.anatomical.copy()
    net.last_spikes[:] = False
    net.last_spikes[mb.kenyon] = True
    net.post_trace[mb.mbon] = 1.0
    mb.observe_activity()
    mb.apply_dopamine(1.0)
    assert np.allclose(net.anatomical, anatomy)
    assert not np.allclose(net.plastic_component, 0)


def test_global_reward_is_not_the_default_on_virtual_fly():
    from organism.fly import VirtualFly
    from flybrain.mushroom_body import MushroomBodyLearning

    fly = VirtualFly.hatch(seed=0, connectome="synthetic")
    assert isinstance(fly.plasticity, MushroomBodyLearning)

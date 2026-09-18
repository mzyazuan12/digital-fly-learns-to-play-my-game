"""Neuromodulators change circuits. They do not call walk()."""

import inspect

from organism.neuromodulation import Neuromodulation
from organism.physiology import Physiology
from organism.toy import miniature_connectome
from organism.bridge import MotorBridge
from flybrain.network import LIFNetwork, LIFParams


def test_physiology_has_no_action_if():
    src = inspect.getsource(Physiology)
    assert "if hunger" not in src
    assert "walk_to_food" not in src
    assert "start_walk" not in src
    mod = inspect.getsource(Neuromodulation)
    assert "walk()" not in mod


def test_octopamine_does_not_write_a_gait_command():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    nm = Neuromodulation(seed=0)
    nm.state.octopamine = 0.8
    nm.state.dopamine = 0.6
    net.clear_drive()
    sources = nm.modulate(net, bridge)
    assert "neuromod.octopamine.DNp09" in sources
    cmd = bridge.last_command
    assert cmd.mode == "rest"


def test_hunger_raises_dopamine_not_a_walk_flag():
    phys = Physiology(seed=0, legacy_scaffold=False)
    phys.state.hunger = 0.9
    phys.neuromodulation.update_from_metabolic(
        0.5, hunger=0.9, arousal=0.2, fatigue=0.1, sleep_pressure=0.0, odor=0.0
    )
    assert phys.neuromodulation.state.dopamine > 0.2
    assert phys.state.walking_bout_s == 0.0

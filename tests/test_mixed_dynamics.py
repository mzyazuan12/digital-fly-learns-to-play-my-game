"""Tiny mixed-dynamics invariants. No MaleCNS. If these fail, do not run lesions."""

import inspect

import numpy as np
import pytest

from flybrain.loader import Connectome
from flybrain.network import (
    GRADED_RELEASE_EPS,
    GRADED_RELEASE_SCALE_MV,
    LIFNetwork,
    MIN_PLASTIC_FACTOR,
    MixedDynamicsNetwork,
)
from flybrain.neuron_model import NeuronKind, NeuronModelTable, ParameterProvenance
from flybrain.neurons import LIFParams, nt_sign


def _chain(kinds: list[str], anatomical: float = 80.0, transmitters: list[str] | None = None):
    n = len(kinds)
    n_edges = n - 1
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.minimum(np.arange(1, n + 1), n_edges)
    post = np.arange(1, n, dtype=np.uint32)
    weight = np.full(n_edges, anatomical, dtype=np.uint32)
    transmitters = transmitters or ["acetylcholine"] * n
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    sign = np.empty(n_edges, dtype=np.int8)
    for i in range(n_edges):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    graph = Connectome(
        neuron_ids=np.arange(1, n + 1, dtype=np.uint64),
        pre_ptr=ptr,
        post=post,
        anatomical=weight,
        sign=sign,
        superclass=np.array([""] * n, dtype=object),
        cell_type=np.array([f"unit{i}" for i in range(n)], dtype=object),
        cell_class=np.array([""] * n, dtype=object),
        side=np.array([""] * n, dtype=object),
        neurotransmitter=np.array(transmitters, dtype=object),
        report={"dataset_id": "unit_mixed"},
    )
    kind = np.array(
        [
            NeuronKind.GRADED_RATE.value if k == "graded" else NeuronKind.SPIKING_LIF.value
            for k in kinds
        ],
        dtype=object,
    )
    models = NeuronModelTable(
        kind=kind,
        provenance=np.array([ParameterProvenance.ASSUMED.value] * n, dtype=object),
        notes={"unit": True},
    )
    params = LIFParams(dt=1.0, delay=1.0, t_ref=1.0, contact_gain=1.0)
    net = MixedDynamicsNetwork(graph, params=params, seed=0, models=models)
    net.intrinsic_noise_std = 0.0
    net.v.fill(net.v_rest)
    net.g_spike.fill(0)
    net.g_graded.fill(0)
    net.g.fill(0)
    return net


def test_lifnetwork_is_mixed_dynamics_alias():
    assert LIFNetwork is MixedDynamicsNetwork
    source = inspect.getsource(MixedDynamicsNetwork)
    assert source.count("def _deliver_graded") == 1
    sig = inspect.signature(MixedDynamicsNetwork._deliver_graded)
    assert "cells" in sig.parameters


def test_graded_a_drives_spiking_b_every_tick_without_accumulating():
    net = _chain(["graded", "spiking"], anatomical=5.0)
    held = float(net.v_rest + GRADED_RELEASE_SCALE_MV)
    currents = []
    voltages = []
    for _ in range(8):
        net.v[0] = held
        net._tick()
        assert bool(net.last_spikes[0]) is False
        assert bool(net.last_spikes[1]) is False
        assert float(net.graded_release[0]) > GRADED_RELEASE_EPS
        assert float(net.g_graded[1]) > 0.0
        currents.append(float(net.g_graded[1]))
        voltages.append(float(net.v[1]))
    assert int(net.counts[0]) == 0
    assert int(net.counts[1]) == 0
    assert np.allclose(currents, currents[0], rtol=0, atol=1e-6)
    assert voltages[-1] > voltages[0]
    assert voltages[-1] > float(net.v_rest)

    for _ in range(4):
        net.v[0] = net.v_rest
        net._tick()
    assert float(net.graded_release[0]) < GRADED_RELEASE_EPS
    assert abs(float(net.g_graded[1])) < 1e-6


def test_graded_cell_is_not_gated_by_refractory_live_mask():
    net = _chain(["graded", "spiking"], anatomical=5.0)
    net.refractory[0] = 9
    net.v[0] = float(net.v_rest + GRADED_RELEASE_SCALE_MV)
    net._tick()
    assert float(net.graded_release[0]) > GRADED_RELEASE_EPS
    assert float(net.g_graded[1]) > 0.0
    assert bool(net.last_spikes[0]) is False


def test_information_crosses_spiking_graded_spiking():
    net = _chain(["spiking", "graded", "spiking"], anatomical=200.0)
    net.inject([0], 40.0, source="unit.hold")
    max_release = 0.0
    max_graded_current = 0.0
    for _ in range(40):
        net._tick()
        max_release = max(max_release, float(net.graded_release[1]))
        max_graded_current = max(max_graded_current, float(abs(net.g_graded[2])))
    assert int(net.counts[0]) > 0
    assert int(net.counts[1]) == 0
    assert bool(net.is_graded[1])
    assert max_release > GRADED_RELEASE_EPS
    assert max_graded_current > 0.0
    assert int(net.counts[2]) > 0


def test_plastic_factor_cannot_reverse_synaptic_effect_sign():
    net = _chain(["spiking", "spiking"], anatomical=10.0, transmitters=["acetylcholine", "gaba"])
    assert MIN_PLASTIC_FACTOR > 0
    anatomy = net.anatomical.copy()
    signed = np.sign(net.synaptic_effect_sign)
    net.plastic_component[:] = -8.0
    net._rebuild_weights()
    plastic_factor = np.clip(1.0 + net.plastic_component, MIN_PLASTIC_FACTOR, 5.0)
    assert np.all(plastic_factor > 0)
    live = anatomy > 0
    assert np.all(np.sign(net.weight[live]) == signed[live])
    assert np.allclose(net.anatomical, anatomy)
    with pytest.raises(RuntimeError, match="functional_gain"):
        net.functional_gain[:] = -1.0
        net._rebuild_weights()

"""NO_SCAFFOLD walking initiation: no timer, no fallback, no named gait command."""

import pytest
from organism.config import (
    NO_SCAFFOLD,
    LEGACY_POLICY,
    ScaffoldViolation,
    assert_neural_drive_source,
    format_policy_banner,
    motor_fidelity_level,
    MotorMode,
)
from organism.bridge import MotorBridge, NonNeuralMotorAuthority, format_walk_trace
from organism.neuromodulation import MODULATORY_EFFECTS
from organism.toy import miniature_connectome
from experiment.walking_initiation_no_scaffold import run, probe_dnp09_on
from flybrain.network import LIFNetwork, LIFParams, MIN_PLASTIC_FACTOR, dataset_validation
from flybrain.neuron_model import NeuronKind
import numpy as np


def test_no_scaffold_disables_timers_and_fallbacks():
    assert NO_SCAFFOLD.allow_behavior_timers is False
    assert NO_SCAFFOLD.allow_motor_fallbacks is False
    assert NO_SCAFFOLD.allow_named_gait_commands is False
    assert NO_SCAFFOLD.allow_root_motion is False
    assert NO_SCAFFOLD.allow_privileged_world_state is False
    assert NO_SCAFFOLD.allow_pretrained_low_level_gait is True
    assert LEGACY_POLICY.allow_behavior_timers is True
    assert NO_SCAFFOLD.no_scaffold is True
    assert LEGACY_POLICY.no_scaffold is False
    banner = format_policy_banner(NO_SCAFFOLD)
    assert "NO_SCAFFOLD MODE" in banner
    assert "timers            OFF" in banner
    assert "named gait cmds   OFF" in banner
    assert "walk fallback     OFF" in banner
    assert "root motion       OFF" in banner
    assert "target coords     OFF" in banner
    assert "developer motor   OFF" in banner
    assert "low-level gait    ON" in banner
    assert issubclass(NonNeuralMotorAuthority, ScaffoldViolation)
    with pytest.raises(ScaffoldViolation):
        assert_neural_drive_source(NO_SCAFFOLD, "walking_timer")


def test_motor_fidelity_level_is_identified_dns_to_cpg():
    assert motor_fidelity_level(MotorMode.ENGINEERED_CPG) == 1
    assert motor_fidelity_level(MotorMode.NEURAL_CPG) == 2


def test_graded_cell_delivers_without_spiking():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    graded = np.flatnonzero(net.is_graded)
    assert graded.size >= 1
    i = int(graded[0])
    start = int(net.ptr[i])
    end = int(net.ptr[i + 1])
    assert end > start
    posts = net.post[start:end]
    net.drive.fill(0)
    net.v.fill(net.v_rest)
    net.v[i] = net.v_rest + 8.0
    net.step(1)
    assert bool(net.last_spikes[i]) is False
    assert net.models.kind[i] == NeuronKind.GRADED_RATE.value
    assert float(net.graded_release[i]) > 0.0
    assert float(net.analog_output[i]) > 0.0
    assert float(np.abs(net.g[posts]).sum()) > 0.0
    assert net.last_n_graded_deliveries >= 1
    assert net.last_n_graded_considered == int(net.is_graded.sum())


def test_graded_cells_are_considered_every_tick_even_without_spikes():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    net.drive.fill(0)
    net.v.fill(net.v_rest)
    net.step(1)
    assert net.last_n_spike_events == 0
    assert net.last_n_graded_considered == int(net.is_graded.sum())
    assert net.last_n_graded_considered >= 1


def test_graded_cell_at_rest_does_not_leak():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    graded = np.flatnonzero(net.is_graded)
    i = int(graded[0])
    net.drive.fill(0)
    net.v.fill(net.v_rest)
    net.g.fill(0)
    net.step(1)
    assert float(net.graded_release[i]) < 1e-3
    assert net.last_n_graded_deliveries == 0


def test_plastic_factor_cannot_flip_synapse_sign():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    anatomy = net.anatomical.copy()
    signed = np.sign(net.synaptic_effect_sign)
    net.plastic_component[:] = -8.0
    net._rebuild_weights()
    assert float(net.plastic_factor.min()) >= MIN_PLASTIC_FACTOR
    assert np.all(net.plastic_factor > 0)
    live = anatomy > 0
    assert np.all(np.sign(net.weight[live]) == signed[live])
    assert np.allclose(net.anatomical, anatomy)


def test_birth_membranes_are_not_identical():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=7)
    assert float(net.v.std()) > 0.2
    assert abs(float(net.v.mean()) - float(net.params.v_rest)) < 1.0


def test_modulatory_effects_have_provenance():
    assert MODULATORY_EFFECTS
    for item in MODULATORY_EFFECTS:
        assert item.confidence in {"MEASURED", "LITERATURE_DERIVED", "INFERRED", "ASSUMED"}
        assert item.source
        assert "walk()" not in item.effect


def test_walk_trace_lists_no_external_command_on_rest():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    counts = np.zeros(graph.n, dtype=np.int32)
    cmd = bridge.read(counts, 0.04)
    assert cmd.mode == "rest"
    assert cmd.scaffold_used is False
    text = format_walk_trace(bridge.last_trace)
    assert "WALK INITIATION TRACE" in text
    assert "TIMER AUTHORITY" in text
    assert "NONE ✓" in text
    assert "STATE: REST" in text
    assert "upstream input" in text
    assert bridge.last_trace["motor_fidelity_level"] == 1
    assert cmd.motor_interface == "ENGINEERED_NEURAL_MOTOR_INTERFACE"


def test_no_scaffold_rejects_bout_timer():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    counts = np.zeros(graph.n, dtype=np.int32)
    with pytest.raises(NonNeuralMotorAuthority):
        bridge.read(counts, 0.04, walking_drive=1.0, walking_bout_s=5.0)


def test_dataset_validation_exposes_assumption_layers():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    report = dataset_validation(graph, models=net.models, policy_name="NO_SCAFFOLD", net=net)
    assert "MaleCNS DATASET VALIDATION" in report["text"]
    assert report["no_scaffold"] is True
    assert report["layers"]["functional_gain"] == "ASSUMED"
    assert report["layers"]["membrane_model"] == "ASSUMED/LITERATURE_DERIVED"
    assert report["graded_cells"] >= 1
    assert report["spiking_cells"] >= 1


def test_no_scaffold_experiment_toy(tmp_path):
    result = run(
        connectome="synthetic",
        seed=1,
        spontaneous_steps=40,
        out=tmp_path / "walking_initiation_no_scaffold.json",
    )
    required = result["checklist_pass_required"]
    assert all(required.values()), required
    assert result["dnp09_probe"]["neural_authority"]
    assert result["trials"]["intact"]["scaffold_used"] is False
    assert result["standing"]["n_rest"] == result["standing"]["n"]
    assert result["statue_is_a_result"] is True
    assert result["policy"]["allow_behavior_timers"] is False
    assert "WALK INITIATION TRACE" in result["dnp09_probe"]["intact"]["trace_text"]
    assert result["birth_checkpoint"]
    assert {"intact", "DNp09_lesion", "upstream_lesion", "sham_lesion"} <= set(result["trials"])
    assert result["comparison"]["statue"] or result["trials"]["intact"]["n_walk"] >= 0
    assert result["synapse_coordinate_tables_loaded"] is False
    assert result["computational_graph"]["synapse_coordinate_tables_loaded"] is False


def test_optogenetic_dnp09_probe_still_causal():
    graph = miniature_connectome(1)
    result = probe_dnp09_on(graph, seed=1)
    assert result["neural_authority"]
    assert result["intact"]["scaffold_used"] is False
    assert result["lesion"]["mode"] == "rest"
    assert result["restored"]["mode"] == "walk"

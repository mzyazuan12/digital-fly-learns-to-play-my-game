"""SHIU_LIF_SANITY_MODEL isolated probes. No MaleCNS. Not PUGLIESE_CPG_MODEL."""

import numpy as np
import pytest

from experiment.dng100_cpg_rhythm import _population_metrics
from flybrain.lif_sanity import (
    ISOLATED_STIM_CURRENT,
    SCALE_RATIO_HI,
    SCALE_RATIO_LO,
    SCALE_SYNAPSE_COUNT,
    SYNAPSE_COUNT_SWEEP,
    assert_lif_sanity,
    cholinergic_depolarizes,
    gaba_hyperpolarizes,
    isolated_dng100_graph,
    isolated_positive_stimulus,
    isolated_rest,
    pugliese_stim_is_not_shiu_current,
    quiet_net,
    run_lif_sanity,
    run_single_synaptic_event,
    synapse_count_scaling,
    tiny_cpg_numerical_sanity,
    transmitter_sign_table,
    weight_orientation,
)
from flybrain.neurons import (
    LIFParams,
    PUGLIESE_CPG_MODEL,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    SHIU_LIF_SANITY_MODEL,
    SYNAPTIC_STEP_MV,
    V_REST_MV,
    V_THRESHOLD_MV,
    VOLTAGE_UNIT,
    WSYN_MV,
    nt_sign,
    valid_dynamics,
    voltage_is_physiological,
    voltages_finite,
)
from organism.cpg_rhythm import derive_rhythm_permissions, interpret_intact, rhythmicity_score


def test_model_labels_are_distinct():
    assert SHIU_LIF_SANITY_MODEL == "shiu_lif_sanity_v1"
    assert PUGLIESE_CPG_MODEL == "pugliese_cpg_reference_v1"
    assert SHIU_LIF_SANITY_MODEL != PUGLIESE_CPG_MODEL
    assert LIFParams().model_id == SHIU_LIF_SANITY_MODEL
    assert LIFParams().model_label == SHIU_LIF_SANITY_MODEL
    assert LIFParams().voltage_unit == VOLTAGE_UNIT == "mV"
    assert LIFParams().wsyn_mv == WSYN_MV == SYNAPTIC_STEP_MV == 0.275


def test_physiological_and_dynamics_are_not_the_same_check():
    rest = np.full(32, -52.0)
    mild = np.full(32, -120.0)
    boom = np.full(32, -400.0)
    nan = np.array([-52.0, np.nan])
    assert voltages_finite(rest)
    assert voltage_is_physiological(rest)
    assert valid_dynamics(rest)
    assert voltages_finite(mild)
    assert not voltage_is_physiological(mild)
    assert valid_dynamics(mild)
    assert voltages_finite(boom)
    assert not voltage_is_physiological(boom)
    assert not valid_dynamics(boom)
    assert not voltages_finite(nan)
    assert not voltage_is_physiological(nan)
    assert not valid_dynamics(nan)
    scored_mild = rhythmicity_score(mild, 1.0)
    assert scored_mild["voltage_physiological"] is False
    assert scored_mild["valid_dynamics"] is True
    assert scored_mild["exploding"] is True
    assert scored_mild["fft_executed"] is False
    assert scored_mild["allow_lesions"] is False
    scored_boom = rhythmicity_score(boom, 1.0)
    assert scored_boom["voltage_physiological"] is False
    assert scored_boom["valid_dynamics"] is False
    assert scored_boom["exploding"] is True
    assert scored_boom["fft_executed"] is False


def test_rhythm_permissions_are_derived_and_default_false():
    permissions = derive_rhythm_permissions(
        dng_voltage_physiological=True,
        dng_dynamics_valid=True,
        all_network_voltages_valid=True,
    )
    assert permissions["dng100_voltage_exploding"] is False
    assert permissions["valid_for_rhythm_analysis"] is True
    assert permissions["allow_lesions"] is True
    denied = derive_rhythm_permissions(
        dng_voltage_physiological=False,
        dng_dynamics_valid=True,
        all_network_voltages_valid=False,
    )
    assert denied["dng100_voltage_physiological"] is False
    assert denied["dng100_dynamics_valid"] is True
    assert denied["dng100_voltage_exploding"] is True
    assert denied["valid_for_rhythm_analysis"] is False
    assert denied["allow_lesions"] is False
    interpreted = interpret_intact(
        {
            "n_oscillatory_legs": 0,
            "any_leg_oscillatory": False,
            "any_exploding": False,
            "core_tonic_plateau": False,
            "legs": {},
        }
    )
    assert interpreted["valid_for_rhythm_analysis"] is False
    assert interpreted["allow_lesions"] is False


def test_solver_rejects_volt_scale_rest():
    with pytest.raises(ValueError, match="millivolts"):
        LIFParams(v_rest=-52e-3, v_threshold=-45e-3)


def test_pugliese_stim_250_is_not_a_shiu_lif_current():
    rec = pugliese_stim_is_not_shiu_current()
    assert rec["ok"], rec
    assert PUGLIESE_CPG_STIM_AMPLITUDE == 250.0
    assert PUGLIESE_CPG_STIM_AMPLITUDE != WSYN_MV
    assert PUGLIESE_CPG_STIM_AMPLITUDE != ISOLATED_STIM_CURRENT


def test_acetylcholine_is_plus_one_gaba_is_minus_one():
    signs = transmitter_sign_table()
    assert signs["ok"]
    assert nt_sign("acetylcholine") == 1
    assert nt_sign("gaba") == -1


def test_isolated_dng100_rests_without_stimulus():
    rec = isolated_rest()
    assert rec["ok"], rec
    assert rec["dynamics_model"] == SHIU_LIF_SANITY_MODEL
    assert rec["spikes"] == 0
    assert abs(rec["v_end"] - V_REST_MV) < 0.5
    assert rec["v_min"] > -80.0
    assert rec["v_max"] < V_THRESHOLD_MV


def test_isolated_dng100_positive_current_depolarizes_then_spikes():
    rec = isolated_positive_stimulus(current=ISOLATED_STIM_CURRENT)
    assert rec["ok"], rec
    assert rec["spikes"] > 0
    assert rec["hyperpolarized_runaway"] is False
    assert rec["depolarized_toward_threshold"] is True
    pre = rec["pre_spike_v"]
    assert len(pre) >= 2
    assert pre[0] > V_REST_MV
    assert pre[-1] > pre[0]
    assert rec["v_min"] > -100.0


def test_isolated_positive_current_does_not_run_to_minus_four_hundred():
    net = quiet_net(isolated_dng100_graph())
    net.add_drive([0], ISOLATED_STIM_CURRENT, source="sanity")
    voltages = []
    for _ in range(30):
        net.step(1)
        voltages.append(float(net.v[0]))
    v = np.asarray(voltages)
    assert np.all(np.isfinite(v))
    assert v.min() >= -100.0
    assert v.max() <= 40.0
    assert not np.any(v < -80.0)


def test_one_spike_psp_is_n_times_wsyn_not_hundreds_of_mv():
    rec = cholinergic_depolarizes()
    assert rec["ok"], rec
    assert rec["synapse_count"] == SCALE_SYNAPSE_COUNT == 20
    assert rec["wsyn_mv"] == WSYN_MV
    assert rec["g_post"] == pytest.approx(20 * 0.275, abs=1e-3)
    assert rec["expected_order_mv"] == pytest.approx(5.5, abs=1e-6)
    assert 0.01 <= rec["dv_post"] <= 50.0
    assert rec["v_post"] > -80.0
    assert rec["v_post"] - V_REST_MV < 50.0


def test_cholinergic_synapse_depolarizes_postsynaptic_cell():
    rec = cholinergic_depolarizes()
    assert rec["ok"], rec
    assert rec["dv_post"] > 0
    assert rec["g_post"] > 0
    assert rec["scale_ok"]


def test_synapse_count_scaling_is_monotonic_and_ratio_near_two():
    rec = synapse_count_scaling()
    assert rec["ok"], rec
    assert rec["model_id"] == SHIU_LIF_SANITY_MODEL
    assert rec["counts"] == list(SYNAPSE_COUNT_SWEEP)
    ach = rec["by_transmitter"]["acetylcholine"]
    gab = rec["by_transmitter"]["gaba"]
    assert all(v > 0 for v in ach["delta_v"])
    assert all(v < 0 for v in gab["delta_v"])
    assert SCALE_RATIO_LO < ach["ratio_20_over_10"] < SCALE_RATIO_HI
    assert SCALE_RATIO_LO < gab["ratio_20_over_10"] < SCALE_RATIO_HI
    one = run_single_synaptic_event(synapse_count=1, transmitter="acetylcholine")
    five = run_single_synaptic_event(synapse_count=5, transmitter="acetylcholine")
    ten = run_single_synaptic_event(synapse_count=10, transmitter="acetylcholine")
    twenty = run_single_synaptic_event(synapse_count=20, transmitter="acetylcholine")
    assert one["n_pre_spikes"] == 1
    assert one["measured"] < five["measured"] < ten["measured"] < twenty["measured"]
    ratio = abs(twenty["measured"] / ten["measured"])
    assert SCALE_RATIO_LO < ratio < SCALE_RATIO_HI
    gaba20 = run_single_synaptic_event(synapse_count=20, transmitter="gaba")
    assert gaba20["measured"] < 0
    assert gaba20["n_pre_spikes"] == 1
    rec = gaba_hyperpolarizes()
    assert rec["ok"], rec
    assert rec["dv_post"] < 0
    assert rec["g_post"] < 0
    assert rec["g_post"] == pytest.approx(-20 * 0.275, abs=1e-3)
    assert rec["v_post"] > -100.0
    assert rec["dv_post"] > -50.0


def test_weight_orientation_is_outgoing_csr_not_transposed():
    rec = weight_orientation()
    assert rec["ok"], rec
    assert rec["outgoing_from_A"] == [1]
    assert rec["outgoing_from_B"] == []
    assert rec["forward_g_post"] > 0
    assert abs(rec["forward_g_pre"]) < 1e-6


def test_zeroing_all_weights_disconnects_dng100_on_toy_graph():
    from flybrain.network import MixedDynamicsNetwork
    from flybrain.neurons import shiu_lif_params
    from organism.toy import miniature_connectome
    from organism.walking_pathways import WalkingCircuit

    graph = miniature_connectome(1)
    net = MixedDynamicsNetwork(graph, params=shiu_lif_params(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    net.anatomical[:] = 0
    net._rebuild_weights()
    assert float(np.abs(net.weight).sum()) == 0.0
    idx = int(WalkingCircuit(graph).indices("DNg100")[0])
    net.reset()
    net.v.fill(net.v_rest)
    net.add_drive([idx], ISOLATED_STIM_CURRENT, source="sanity.W0")
    voltages = []
    n_spikes = 0
    for _ in range(40):
        net.step(1)
        voltages.append(float(net.v[idx]))
        n_spikes += int(net.last_spikes[idx])
    v = np.asarray(voltages)
    assert n_spikes > 0
    assert np.all(np.isfinite(v))
    assert v.min() > -80.0
    assert v.max() <= 40.0


def test_all_lif_sanity_checks_pass():
    report = run_lif_sanity()
    assert report["ok"], report["failed"]
    assert report["dynamics_model"] == SHIU_LIF_SANITY_MODEL
    assert report["not_a_pugliese_reproduction"] is True
    assert report["physiological_bound_is_debug_guardrail"] is True
    assert_lif_sanity()
    assert report["v_rest"] == -52.0
    assert report["v_threshold"] == -45.0
    assert report["wsyn_mv"] == 0.275


def test_tiny_cpg_is_numerically_sane_and_not_a_pugliese_reproduction():
    rec = tiny_cpg_numerical_sanity()
    assert rec["ok"], rec
    assert rec["dynamics_model"] == SHIU_LIF_SANITY_MODEL
    assert rec["not_a_pugliese_reproduction"] is True
    assert rec["valid_dynamics"] is True
    assert rec["allow_lesions"] is False
    assert rec["dominant_frequency"] is None
    assert rec["rhythmicity_score"] is None
    assert rec["dng100_spikes"] > 0
    assert rec["e1_g_max"] > 0
    assert rec["v_min"] >= -100.0
    assert rec["v_max"] <= 40.0
    assert rec["n_neurons"] == 7


def test_exploding_voltage_nulls_dominant_frequency_and_scores():
    boom = np.linspace(-52.0, -400.0, 400)
    scored = rhythmicity_score(boom, 1.0)
    assert scored["exploding"]
    assert scored["valid_dynamics"] is False
    assert scored["valid_for_rhythm_analysis"] is False
    assert scored["allow_lesions"] is False
    assert scored["dominant_frequency"] is None
    assert scored["score"] is None
    assert scored["rhythmicity_score"] is None
    assert scored["oscillatory"] is False

    summary = {
        "any_exploding": True,
        "valid_dynamics": False,
        "valid_for_rhythm_analysis": False,
        "allow_lesions": False,
        "DNg100_v": {"mean": -421.0, "min": -589.0, "max": -379.0, "exploding": True},
        "legs": {
            "FL": {
                "E1": {"score": 0.0, "mean": 0.0, "oscillatory": False, "dominant_frequency": 17.4},
                "E2": {"score": 0.0, "mean": 0.0, "oscillatory": False, "peak_hz": 17.4},
                "I1": {"score": 0.14, "mean": 0.1, "oscillatory": False},
                "I2": {"score": 0.0, "mean": 0.0, "oscillatory": False},
                "MN": {"score": 0.46, "mean": 0.2, "oscillatory": True, "dominant_frequency": 17.4},
            }
        },
    }
    metrics = _population_metrics(summary)
    assert metrics["valid_dynamics"] is False
    assert metrics["allow_lesions"] is False
    assert metrics["dominant_frequency"] is None
    assert metrics["rhythmicity_score"] is None
    assert metrics["rhythmicity_score_E1"] is None
    interpreted = interpret_intact(
        {
            **summary,
            "n_oscillatory_legs": 1,
            "any_leg_oscillatory": True,
            "core_tonic_plateau": False,
        }
    )
    assert interpreted["oscillation_reproduced"] is False
    assert interpreted["allow_lesions"] is False
    assert interpreted["answer"] == "no"


def test_mn_oscillation_without_e1_e2_is_not_published_cpg():
    summary = {
        "n_oscillatory_legs": 1,
        "any_leg_oscillatory": True,
        "any_exploding": False,
        "valid_dynamics": True,
        "valid_for_rhythm_analysis": True,
        "core_tonic_plateau": False,
        "legs": {
            "FL": {
                "E1": {"oscillatory": False, "score": 0.0},
                "E2": {"oscillatory": False, "score": 0.0},
                "I1": {"oscillatory": False, "score": 0.14},
                "I2": {"oscillatory": False, "score": 0.0},
                "MN": {"oscillatory": True, "score": 0.46},
            }
        },
    }
    interpreted = interpret_intact(summary)
    assert interpreted["e1_rhythmic"] is False
    assert interpreted["e2_rhythmic"] is False
    assert interpreted["oscillation_reproduced"] is False
    assert "E1 and E2" in interpreted["next_step"]

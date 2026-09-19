"""SHIU_LIF_SANITY_MODEL probes. Not PUGLIESE_CPG_MODEL.

If these fail, do not run MaleCNS or lesions. Voltage is millivolts.
"""

from __future__ import annotations

import numpy as np

from flybrain.loader import Connectome
from flybrain.network import MixedDynamicsNetwork
from flybrain.neuron_model import NeuronKind, NeuronModelTable, ParameterProvenance
from flybrain.neurons import (
    DELAY_MS,
    PHYSIOLOGICAL_V_LOWER_MV,
    PHYSIOLOGICAL_V_UPPER_MV,
    PUGLIESE_CPG_MODEL,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    SHIU_LIF_SANITY_MODEL,
    SYNAPTIC_STEP_MV,
    TAU_M_MS,
    TAU_SYN_MS,
    V_RESET_MV,
    V_REST_MV,
    V_THRESHOLD_MV,
    VOLTAGE_UNIT,
    WSYN_MV,
    nt_sign,
    shiu_lif_params,
    valid_dynamics,
    voltage_is_physiological,
    voltages_finite,
)

# Isolated probe current in the Shiu millivolt drive convention.
# Not PUGLIESE_CPG_STIM_AMPLITUDE (250 in their size-normalized rate ODE).
ISOLATED_STIM_CURRENT = 40.0
SCALE_SYNAPSE_COUNT = 20
SYNAPSE_COUNT_SWEEP = (1, 5, 10, 20, 50)
SCALE_RATIO_COUNTS = (10, 20)
SCALE_RATIO_LO = 1.5
SCALE_RATIO_HI = 2.5
# First-order expected PSP scale: N × Wsyn. Kernel/integration may be smaller.
# A single spike must not move V by hundreds of mV.
SCALE_DV_MAX_MV = 50.0
SCALE_DV_MIN_MV = 0.01
DEBUG_HIERARCHY = (
    "numerical validity",
    "membrane validity",
    "synaptic sign validity",
    "synaptic scale validity",
    "graph orientation validity",
    "tiny CPG validity",
    "rhythm analysis",
    "lesions",
)

TINY_CPG_INDEX = {
    "DNg100": 0,
    "E1": 1,
    "E2": 2,
    "E3": 3,
    "I1": 4,
    "I2": 5,
    "MN": 6,
}


def _spiking_models(n: int) -> NeuronModelTable:
    return NeuronModelTable(
        kind=np.full(n, NeuronKind.SPIKING_LIF.value, dtype=object),
        provenance=np.full(n, ParameterProvenance.ASSUMED.value, dtype=object),
        notes={
            "lif_sanity": True,
            "model_id": SHIU_LIF_SANITY_MODEL,
            "dynamics_model": SHIU_LIF_SANITY_MODEL,
        },
    )


def _csr(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int):
    order = np.argsort(pre, kind="stable")
    pre, post, weight = pre[order], post[order], weight[order]
    counts = np.bincount(pre, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts, dtype=np.int64)
    return ptr, post, weight


def isolated_dng100_graph() -> Connectome:
    """One DNg100-labeled cell, W = 0. Completely disconnected."""
    return Connectome(
        neuron_ids=np.array([10045], dtype=np.uint64),
        pre_ptr=np.zeros(2, dtype=np.int64),
        post=np.zeros(0, dtype=np.uint32),
        anatomical=np.zeros(0, dtype=np.uint32),
        sign=np.zeros(0, dtype=np.int8),
        superclass=np.array(["descending_neuron"], dtype=object),
        cell_type=np.array(["DNg100"], dtype=object),
        cell_class=np.array([""], dtype=object),
        side=np.array(["L"], dtype=object),
        neurotransmitter=np.array(["acetylcholine"], dtype=object),
        report={
            "dataset_id": "lif_sanity_isolated_dng100",
            "model_id": SHIU_LIF_SANITY_MODEL,
            "dynamics_model": SHIU_LIF_SANITY_MODEL,
        },
    )


def directed_pair_graph(
    transmitter: str,
    *,
    reverse: bool = False,
    synapse_count: int = SCALE_SYNAPSE_COUNT,
) -> Connectome:
    """Two spiking cells. Default anatomy is A → B only (not B → A)."""
    pre = np.array([1 if reverse else 0], dtype=np.uint32)
    post = np.array([0 if reverse else 1], dtype=np.uint32)
    anatomical = np.array([int(synapse_count)], dtype=np.uint32)
    transmitters = [transmitter, "acetylcholine"]
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    ptr = np.zeros(3, dtype=np.int64)
    ptr[int(pre[0]) + 1 :] = 1
    sign = np.array([neuron_sign[int(pre[0])]], dtype=np.int8)
    return Connectome(
        neuron_ids=np.array([1, 2], dtype=np.uint64),
        pre_ptr=ptr,
        post=post,
        anatomical=anatomical,
        sign=sign,
        superclass=np.array(["descending_neuron", "descending_neuron"], dtype=object),
        cell_type=np.array(["A", "B"], dtype=object),
        cell_class=np.array(["", ""], dtype=object),
        side=np.array(["L", "L"], dtype=object),
        neurotransmitter=np.array(transmitters, dtype=object),
        report={
            "dataset_id": "lif_sanity_pair",
            "transmitter": transmitter,
            "reverse": reverse,
            "synapse_count": int(synapse_count),
            "wsyn_mv": WSYN_MV,
            "synaptic_step_mv": SYNAPTIC_STEP_MV,
            "model_id": SHIU_LIF_SANITY_MODEL,
            "dynamics_model": SHIU_LIF_SANITY_MODEL,
        },
    )


def quiet_net(graph: Connectome, *, models: NeuronModelTable | None = None) -> MixedDynamicsNetwork:
    params = shiu_lif_params(dt=1.0)
    net = MixedDynamicsNetwork(
        graph,
        params=params,
        seed=0,
        models=models or _spiking_models(graph.n),
    )
    net.intrinsic_noise_std = 0.0
    net.reset()
    return net


def _record(net: MixedDynamicsNetwork, steps: int) -> dict:
    v = np.zeros((steps, net.n), dtype=np.float64)
    g = np.zeros((steps, net.n), dtype=np.float64)
    spikes = np.zeros((steps, net.n), dtype=np.int8)
    for t in range(steps):
        net.step(1)
        v[t] = net.v
        g[t] = net.g
        spikes[t] = net.last_spikes.astype(np.int8)
    return {"v": v, "g": g, "spikes": spikes}


def isolated_rest(steps: int = 80) -> dict:
    """TEST 1: W = 0, stimulus = 0 → V → rest, no spikes, no explosion."""
    net = quiet_net(isolated_dng100_graph())
    net.v[0] = np.float32(-48.0)
    rec = _record(net, steps)
    v = rec["v"][:, 0]
    ok = (
        valid_dynamics(v)
        and int(rec["spikes"][:, 0].sum()) == 0
        and abs(float(v[-1]) - V_REST_MV) < 0.5
        and float(v[-1]) < float(v[0])
        and abs(float(net.weight.sum())) == 0.0
    )
    return {
        "ok": bool(ok),
        "name": "isolated_rest",
        "model_id": SHIU_LIF_SANITY_MODEL,
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "voltage_unit": VOLTAGE_UNIT,
        "v_start": float(v[0]),
        "v_end": float(v[-1]),
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "spikes": int(rec["spikes"][:, 0].sum()),
        "v_rest": V_REST_MV,
    }


def isolated_positive_stimulus(current: float = ISOLATED_STIM_CURRENT, steps: int = 80) -> dict:
    """TEST 2: W = 0, stimulus > 0 → V becomes less negative, then spikes and resets."""
    net = quiet_net(isolated_dng100_graph())
    net.add_drive([0], float(current), source="sanity.isolated.DNg100")
    pre = []
    v_hist = []
    n_spikes = 0
    first_spike = None
    for t in range(steps):
        net.step(1)
        voltage = float(net.v[0])
        v_hist.append(voltage)
        spiked = bool(net.last_spikes[0])
        if spiked:
            n_spikes += 1
            if first_spike is None:
                first_spike = t
        elif first_spike is None:
            pre.append(voltage)
    v = np.asarray(v_hist, dtype=np.float64)
    if first_spike == 0:
        depolarized = True
    else:
        depolarized = len(pre) >= 2 and pre[0] > V_REST_MV + 0.05 and pre[-1] > pre[0]
    hyperpolarized_runaway = bool(np.any(v < V_REST_MV - 5.0) and n_spikes == 0)
    ok = (
        valid_dynamics(v)
        and n_spikes > 0
        and depolarized
        and not hyperpolarized_runaway
    )
    return {
        "ok": bool(ok),
        "name": "isolated_positive_stimulus",
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "voltage_unit": VOLTAGE_UNIT,
        "current": float(current),
        "pre_spike_v": [float(x) for x in pre[:8]],
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "spikes": int(n_spikes),
        "first_spike_step": first_spike,
        "hyperpolarized_runaway": hyperpolarized_runaway,
        "depolarized_toward_threshold": depolarized,
    }


def _pair_psp(transmitter: str, *, reverse: bool = False, synapse_count: int = SCALE_SYNAPSE_COUNT) -> dict:
    net = quiet_net(directed_pair_graph(transmitter, reverse=reverse, synapse_count=synapse_count))
    rest_b = float(net.v[1])
    rest_a = float(net.v[0])
    sign = int(nt_sign(transmitter))
    expected_g = float(synapse_count) * float(net.params.wsyn_mv) * sign
    expected_order_mv = float(synapse_count) * float(net.params.wsyn_mv)
    net.v[0] = np.float32(float(net.v_th) + 1.0)
    net.step(1)
    a_spiked = bool(net.last_spikes[0])
    n_pre_spikes = int(a_spiked)
    delivered_g = 0.0
    v_b = rest_b
    v_a = float(net.v[0])
    for _ in range(int(net.delay_slots) + 2):
        net.step(1)
        n_pre_spikes += int(net.last_spikes[0])
        delivered_g = float(net.g[1])
        v_b = float(net.v[1])
        v_a = float(net.v[0])
        if abs(delivered_g) > 1e-9:
            break
    dv_post = float(v_b - rest_b)
    scale_ok = (
        abs(delivered_g - expected_g) <= 1e-3 * max(1.0, abs(expected_g))
        and SCALE_DV_MIN_MV <= abs(dv_post) <= SCALE_DV_MAX_MV
        and voltages_finite([v_a, v_b, rest_a, rest_b])
        and valid_dynamics([v_a, v_b, rest_a, rest_b])
        and voltage_is_physiological([v_a, v_b, rest_a, rest_b])
        and abs(dv_post) < 100.0
        and n_pre_spikes == 1
    )
    return {
        "transmitter": transmitter,
        "nt_sign": sign,
        "reverse_edge": bool(reverse),
        "synapse_count": int(synapse_count),
        "wsyn_mv": float(net.params.wsyn_mv),
        "synaptic_step_mv": SYNAPTIC_STEP_MV,
        "expected_g": expected_g,
        "expected_order_mv": expected_order_mv,
        "expected_scale_mv": expected_order_mv,
        "a_spiked": a_spiked,
        "n_pre_spikes": int(n_pre_spikes),
        "g_post": delivered_g,
        "g_pre": float(net.g[0]),
        "v_post": v_b,
        "v_pre": v_a,
        "dv_post": dv_post,
        "measured": dv_post,
        "dv_pre": float(v_a - rest_a),
        "weight": float(net.weight[0]) if net.weight.size else 0.0,
        "scale_ok": bool(scale_ok),
        "delay_ms": DELAY_MS,
        "tau_m_ms": TAU_M_MS,
        "tau_syn_ms": TAU_SYN_MS,
        "voltage_unit": VOLTAGE_UNIT,
        "model_id": SHIU_LIF_SANITY_MODEL,
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
    }


def cholinergic_depolarizes() -> dict:
    """A (acetylcholine) → B: forcing A to spike must depolarize B by ~N×Wsyn, not hundreds of mV."""
    rec = _pair_psp("acetylcholine")
    ok = (
        rec["a_spiked"]
        and rec["nt_sign"] == 1
        and rec["weight"] > 0
        and rec["g_post"] > 0
        and rec["dv_post"] > 0
        and rec["scale_ok"]
        and abs(rec["g_pre"]) < 1e-6
    )
    rec.update({"ok": bool(ok), "name": "cholinergic_depolarizes"})
    return rec


def gaba_hyperpolarizes() -> dict:
    """A (GABA) → B: forcing A to spike must hyperpolarize B by ~N×Wsyn, not −300 mV."""
    rec = _pair_psp("gaba")
    ok = (
        rec["a_spiked"]
        and rec["nt_sign"] == -1
        and rec["weight"] < 0
        and rec["g_post"] < 0
        and rec["dv_post"] < 0
        and rec["scale_ok"]
        and abs(rec["g_pre"]) < 1e-6
        and rec["v_post"] > PHYSIOLOGICAL_V_LOWER_MV
    )
    rec.update({"ok": bool(ok), "name": "gaba_hyperpolarizes"})
    return rec


def weight_orientation() -> dict:
    """TEST 4: CSR is outgoing. A → B exists, B → A does not."""
    forward = _pair_psp("acetylcholine", reverse=False)
    backward = _pair_psp("acetylcholine", reverse=True)
    net = quiet_net(directed_pair_graph("acetylcholine"))
    ptr = net.connectome.pre_ptr
    post = net.connectome.post
    outgoing_a = post[int(ptr[0]) : int(ptr[1])].tolist()
    outgoing_b = post[int(ptr[1]) : int(ptr[2])].tolist()
    ok = (
        outgoing_a == [1]
        and outgoing_b == []
        and forward["a_spiked"]
        and forward["g_post"] > 0
        and forward["scale_ok"]
        and abs(forward["g_pre"]) < 1e-6
        and forward["dv_post"] > 0
        and (not backward["g_post"] > 1e-9)
    )
    return {
        "ok": bool(ok),
        "name": "weight_orientation",
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "outgoing_from_A": outgoing_a,
        "outgoing_from_B": outgoing_b,
        "forward_g_post": forward["g_post"],
        "forward_g_pre": forward["g_pre"],
        "forward_dv_post": forward["dv_post"],
        "expected_g": forward["expected_g"],
        "transposed_graph_g_post": backward["g_post"],
        "convention": "pre_ptr is outgoing CSR; post[pre_ptr[i]:pre_ptr[i+1]] are targets of i",
    }


def transmitter_sign_table() -> dict:
    table = {
        "acetylcholine": int(nt_sign("acetylcholine")),
        "gaba": int(nt_sign("gaba")),
        "glutamate": int(nt_sign("glutamate")),
        "histamine": int(nt_sign("histamine")),
    }
    ok = table["acetylcholine"] == 1 and table["gaba"] == -1 and table["glutamate"] == -1 and table["histamine"] == -1
    return {"ok": bool(ok), "name": "transmitter_sign_table", "signs": table}


def pugliese_stim_is_not_shiu_current() -> dict:
    """Lock the two stimulus conventions apart. Do not inject 250 into this LIF."""
    ok = (
        PUGLIESE_CPG_STIM_AMPLITUDE == 250.0
        and PUGLIESE_CPG_STIM_AMPLITUDE != WSYN_MV
        and PUGLIESE_CPG_STIM_AMPLITUDE != ISOLATED_STIM_CURRENT
        and SHIU_LIF_SANITY_MODEL != PUGLIESE_CPG_MODEL
    )
    return {
        "ok": bool(ok),
        "name": "pugliese_stim_is_not_shiu_current",
        "shiu_model": SHIU_LIF_SANITY_MODEL,
        "pugliese_model": PUGLIESE_CPG_MODEL,
        "pugliese_cpg_stim": PUGLIESE_CPG_STIM_AMPLITUDE,
        "shiu_wsyn_mv": WSYN_MV,
        "shiu_isolated_current": ISOLATED_STIM_CURRENT,
        "note": (
            "Pugliese DNg100_Stim amplitude 250 is PUGLIESE_CPG_MODEL with "
            "cell-size normalization. Do not transfer it into SHIU_LIF_SANITY_MODEL."
        ),
    }


def tiny_cpg_connectome(*, synapse_count: int = SCALE_SYNAPSE_COUNT) -> Connectome:
    """DNg100 + E1/E2/E3/I1/I2 + one MN. Not MaleCNS. Not PUGLIESE_CPG_MODEL."""
    n = 7
    dng, e1, e2, e3, i1, i2, mn = 0, 1, 2, 3, 4, 5, 6
    w = int(synapse_count)
    pre = np.array(
        [
            dng, dng, dng,
            e1, e2,
            e1, e2,
            i1, i1,
            e2,
            i2, i2,
            e1, e2, e3,
        ],
        dtype=np.uint32,
    )
    post = np.array(
        [
            e1, e2, e3,
            e2, e1,
            i1, i1,
            e1, e2,
            i2,
            e1, e2,
            mn, mn, mn,
        ],
        dtype=np.uint32,
    )
    anatomical = np.full(pre.size, w, dtype=np.uint32)
    ptr, post, anatomical = _csr(pre, post, anatomical, n)
    transmitters = np.array(
        ["acetylcholine", "acetylcholine", "acetylcholine", "acetylcholine", "gaba", "gaba", "acetylcholine"],
        dtype=object,
    )
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    sign = np.empty(len(post), dtype=np.int8)
    for i in range(n):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    return Connectome(
        neuron_ids=np.arange(1, n + 1, dtype=np.uint64),
        pre_ptr=ptr,
        post=post.astype(np.uint32, copy=False),
        anatomical=anatomical,
        sign=sign,
        superclass=np.array(
            ["descending_neuron", "vnc_intrinsic", "vnc_intrinsic", "vnc_intrinsic", "vnc_intrinsic", "vnc_intrinsic", "vnc_motor"],
            dtype=object,
        ),
        cell_type=np.array(
            ["DNg100", "IN17A001", "INXXX466", "IN19B012", "IN16B036", "IN19A007", "MN"],
            dtype=object,
        ),
        cell_class=np.array(["", "premotor", "premotor", "premotor", "premotor", "premotor", "motor"], dtype=object),
        side=np.array(["L"] * n, dtype=object),
        neurotransmitter=transmitters,
        report={
            "dataset_id": "tiny_cpg",
            "dynamics_model": SHIU_LIF_SANITY_MODEL,
            "not_pugliese_cpg_model": True,
            "synapse_count": w,
            "wsyn_mv": WSYN_MV,
        },
    )


def tiny_cpg_numerical_sanity(*, steps: int = 80, current: float = ISOLATED_STIM_CURRENT) -> dict:
    """Reconnect the walking motif only after the four isolated tests pass.

    Success is finite millivolt traces and downstream current, not a published rhythm.
    """
    four = run_lif_sanity()
    if not four["ok"]:
        return {
            "ok": False,
            "name": "tiny_cpg_numerical_sanity",
            "failed": ["lif_sanity"] + four["failed"],
            "note": "Isolated SHIU_LIF tests failed. Do not interpret this CPG.",
        }
    graph = tiny_cpg_connectome()
    net = MixedDynamicsNetwork(graph, params=shiu_lif_params(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    net.reset()
    dng = TINY_CPG_INDEX["DNg100"]
    e1 = TINY_CPG_INDEX["E1"]
    i1 = TINY_CPG_INDEX["I1"]
    net.add_drive([dng], float(current), source="sanity.tiny_cpg.DNg100")
    v = np.zeros((steps, net.n), dtype=np.float64)
    g_e1 = np.zeros(steps, dtype=np.float64)
    dng_spikes = 0
    for t in range(steps):
        net.step(1)
        v[t] = net.v
        g_e1[t] = float(net.g[e1])
        dng_spikes += int(net.last_spikes[dng])
    dynamics_ok = bool(valid_dynamics(v))
    e1_depolarized = bool(float(np.max(v[:, e1])) > V_REST_MV + 0.05)
    e1_got_current = bool(float(np.max(g_e1)) > 0.0)
    i1_not_exploded = bool(valid_dynamics(v[:, i1]))
    ok = dynamics_ok and dng_spikes > 0 and e1_depolarized and e1_got_current and i1_not_exploded
    return {
        "ok": bool(ok),
        "name": "tiny_cpg_numerical_sanity",
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "not_a_pugliese_reproduction": True,
        "voltage_unit": VOLTAGE_UNIT,
        "valid_dynamics": dynamics_ok,
        "allow_lesions": False,
        "dominant_frequency": None,
        "rhythmicity_score": None,
        "dng100_spikes": int(dng_spikes),
        "e1_vmax": float(np.max(v[:, e1])),
        "e1_vmin": float(np.min(v[:, e1])),
        "e1_g_max": float(np.max(g_e1)),
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "n_neurons": int(graph.n),
        "wsyn_mv": WSYN_MV,
        "v_reset_mv": V_RESET_MV,
        "v_threshold_mv": V_THRESHOLD_MV,
        "next": (
            "Tiny CPG voltages are finite and DNg100 drives E1. Still not a "
            "Pugliese reproduction. Only then consider full MaleCNS."
            if ok
            else "Tiny CPG is not numerically sane. Do not load MaleCNS."
        ),
    }


def run_lif_sanity() -> dict:
    tests = [
        transmitter_sign_table(),
        pugliese_stim_is_not_shiu_current(),
        isolated_rest(),
        isolated_positive_stimulus(),
        cholinergic_depolarizes(),
        gaba_hyperpolarizes(),
        weight_orientation(),
    ]
    by_name = {row["name"]: row for row in tests}
    failed = [row["name"] for row in tests if not row["ok"]]
    return {
        "ok": not failed,
        "failed": failed,
        "tests": by_name,
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "not_a_pugliese_reproduction": True,
        "pugliese_cpg_model": PUGLIESE_CPG_MODEL,
        "voltage_unit": VOLTAGE_UNIT,
        "v_rest": V_REST_MV,
        "v_reset": V_RESET_MV,
        "v_threshold": V_THRESHOLD_MV,
        "tau_m_ms": TAU_M_MS,
        "tau_syn_ms": TAU_SYN_MS,
        "delay_ms": DELAY_MS,
        "wsyn_mv": WSYN_MV,
        "physiological_v_lower": PHYSIOLOGICAL_V_LOWER_MV,
        "physiological_v_upper": PHYSIOLOGICAL_V_UPPER_MV,
        "physiological_bound_is_debug_guardrail": True,
        "next_if_failed": (
            "Do not load MaleCNS. Do not run lesions. The sign, scale, or CSR "
            "orientation of MixedDynamicsNetwork is wrong."
        ),
        "next_if_passed": (
            "SHIU_LIF_SANITY_MODEL isolated tests passed. Next is the tiny CPG "
            "subgraph (DNg100+E1/E2/E3/I1/I2), still not PUGLIESE_CPG_MODEL, "
            "and not full MaleCNS."
        ),
    }


def assert_lif_sanity() -> dict:
    report = run_lif_sanity()
    if not report["ok"]:
        raise RuntimeError(
            "SHIU_LIF_SANITY_MODEL failed: "
            + ", ".join(report["failed"])
            + ". Do not run MaleCNS or lesions. "
            + report["next_if_failed"]
        )
    return report


def format_lif_sanity(report: dict | None = None) -> str:
    report = report or run_lif_sanity()
    lines = [
        f"{SHIU_LIF_SANITY_MODEL}  (not {PUGLIESE_CPG_MODEL})",
        f"ok={report['ok']}  unit={report['voltage_unit']}  "
        f"V_rest={report['v_rest']}  V_thresh={report['v_threshold']}  Wsyn={report['wsyn_mv']} mV/synapse",
        "",
    ]
    for name, row in report["tests"].items():
        mark = "PASS" if row["ok"] else "FAIL"
        extra = ""
        if "v_end" in row:
            extra = f"  V {row.get('v_start'):.2f} → {row['v_end']:.2f}"
        elif "pre_spike_v" in row:
            extra = f"  spikes={row.get('spikes')}  minV={row.get('v_min'):.2f}"
        elif "outgoing_from_A" in row:
            extra = f"  A→{row['outgoing_from_A']}  B→{row['outgoing_from_B']}"
        elif "expected_g" in row:
            extra = (
                f"  N={row.get('synapse_count')}×Wsyn={row.get('wsyn_mv')} "
                f"g={row.get('g_post'):.3f} (expect {row.get('expected_g'):.3f}) "
                f"dV={row.get('dv_post'):.3f} mV"
            )
        elif "signs" in row:
            extra = f"  {row['signs']}"
        elif "pugliese_cpg_stim" in row:
            extra = f"  Pugliese stim={row['pugliese_cpg_stim']} ≠ Shiu Wsyn/current"
        lines.append(f"  {mark}  {name}{extra}")
    lines.append("")
    lines.append(report["next_if_passed"] if report["ok"] else report["next_if_failed"])
    return "\n".join(lines) + "\n"


def format_tiny_cpg(report: dict | None = None) -> str:
    report = report or tiny_cpg_numerical_sanity()
    return (
        f"{SHIU_LIF_SANITY_MODEL} tiny CPG  ok={report['ok']}  "
        f"valid_dynamics={report.get('valid_dynamics')}  "
        f"DNg100 spikes={report.get('dng100_spikes')}  "
        f"E1 Vmax={report.get('e1_vmax')}  "
        f"V[{report.get('v_min')}, {report.get('v_max')}] mV\n"
        f"{report.get('next')}\n"
    )

"""Sustained DNg100 current → per-leg VNC CPG traces.

Question: does MixedDynamicsNetwork on this intact MaleCNS graph reproduce
the bioRxiv DNg100 → E1/E2/inhibition → motor rhythm (Pugliese et al. 2025)?

Stimulate the single DNg100 that innervates the left VNC. Let ordinary
MaleCNS edges propagate. Record each E1/E2/I1 copy in its T1/T2/T3 × L/R
slot from per-bodyId LegNp innervation. Do not hand-wire pairwise gains.
Do not retune the graph. Do not drive FlyBody.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.lif_sanity import (
    assert_lif_sanity,
    format_lif_sanity,
    format_tiny_cpg,
    run_lif_sanity,
    tiny_cpg_connectome,
    tiny_cpg_numerical_sanity,
)
from flybrain.loader import DEFAULT_DATA, computational_graph_manifest, shuffled_connectome
from flybrain.network import MixedDynamicsNetwork, dataset_validation
from flybrain.neurons import (
    PUGLIESE_CPG_MODEL,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    SHIU_LIF_SANITY_MODEL,
    VOLTAGE_UNIT,
    WSYN_MV,
    shiu_lif_params,
)
from organism.config import MODEL_VERSION, MotorMode, NO_SCAFFOLD, format_policy_banner
from organism.cpg_rhythm import (
    RHYTHMICITY_THRESHOLD,
    allocate_traces,
    interpret_intact,
    record_tick,
)
from organism.fly import VirtualFly, _json_ready
from organism.roi_innervation import (
    INTERNAL_TO_PAPER,
    CORE_CPG_TYPES,
    MALECNS_DNG100_BODY_IDS,
    PUGLIESE_DNG100_STIM,
    PUGLIESE_MANC_STIM_BODY,
    PUGLIESE_MANC_STIM_INDEX,
    load_cpg_mapping,
    left_vnc_dng100_malecns_body_id,
    malecns_body_id_of,
    malecns_dng100_by_vnc_innervation,
    malecns_dng100_matching_manc_body,
)
from organism.toy import miniature_connectome
from organism.walking_pathways import (
    E5_TYPE_PROVENANCE,
    I2_TYPE_PROVENANCE,
    LEG_SLOTS,
    NEURAL_CPG_DRIVES_JOINTS,
    WALKING_CIRCUIT_TYPES,
    WalkingCircuit,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dng100_cpg_rhythm.json"
TRACES = ROOT / "outputs" / "dng100_cpg_rhythm_traces.npz"


def stimulated_dng100(graph, circuit: WalkingCircuit, stim: str = "left_vnc") -> dict:
    """Choose which MaleCNS DNg100 to inject. Default: left-VNC innervation.

    Pugliese DNg100_Stim is the same biological TYPE on a different animal:
    MANC_T1 matrix index 31, MANC body 10093. In that MANC table, body 10056
    is vMS16. MaleCNS DNg100 is resolved from annotations[type == DNg100].
    """
    all_idx = circuit.indices("DNg100")
    mapping = load_cpg_mapping() if graph.n >= 10_000 else {}
    malecns_block = (mapping.get("DNg100") or {}).get("malecns") or {}
    left_vnc = left_vnc_dng100_malecns_body_id(mapping) if mapping else None
    right_vnc = malecns_dng100_by_vnc_innervation(mapping, "R") if mapping else None
    by_side = {"L": [], "R": []}
    for i in all_idx.tolist():
        side = str(graph.side[int(i)]).upper()[:1]
        if side in by_side:
            by_side[side].append(int(i))
    chosen = all_idx
    note = "both MaleCNS DNg100 neurons"
    if stim in {"both", "all"}:
        chosen = all_idx
        note = "both MaleCNS DNg100 neurons"
    elif stim in {"pugliese", "manc_correspondent"}:
        correspondent = (
            malecns_dng100_matching_manc_body(int(PUGLIESE_DNG100_STIM["source_body_id"]), mapping)
            if mapping
            else None
        )
        if correspondent is not None:
            chosen = np.asarray([graph.index_of(correspondent)], dtype=np.int32)
            note = (
                f"MaleCNS DNg100 whose curated mancBodyid equals Pugliese MANC body "
                f"{PUGLIESE_DNG100_STIM['source_body_id']} (malecns_body_id={correspondent}). "
                "Same annotated type, different specimen. Not integer identity."
            )
        elif by_side["R"]:
            chosen = np.asarray(by_side["R"], dtype=np.int32)
            note = "toy fallback: annotation-right DNg100"
        else:
            chosen = all_idx[:1]
            note = "first DNg100 (mancBodyid correspondence unavailable)"
    elif stim in {"left_vnc", "DNg100_left_vnc"}:
        if left_vnc is not None:
            chosen = np.asarray([graph.index_of(left_vnc)], dtype=np.int32)
            note = (
                f"MaleCNS DNg100 that innervates the left VNC "
                f"(body_id={left_vnc}, annotation side is not this key). "
                "Pugliese reference is MANC_T1 matrix_index=31 body_id=10093 "
                "(same type, different animal). MANC body 10056 is vMS16."
            )
        elif by_side["L"]:
            chosen = np.asarray(by_side["L"], dtype=np.int32)
            note = "toy/annotation-left DNg100 (no ROI mapping)"
        else:
            chosen = all_idx[:1]
            note = "first DNg100 (left-VNC mapping unavailable)"
    elif stim in {"right_vnc"}:
        if right_vnc is not None:
            chosen = np.asarray([graph.index_of(right_vnc)], dtype=np.int32)
            note = f"MaleCNS DNg100 that innervates the right VNC (body_id={right_vnc})"
        elif by_side["R"]:
            chosen = np.asarray(by_side["R"], dtype=np.int32)
            note = "annotation-right DNg100"
    elif stim in {"soma_left", "DNg100_L"}:
        if by_side["L"]:
            chosen = np.asarray(by_side["L"], dtype=np.int32)
            note = "annotation-left instance DNg100_L"
    elif stim in {"soma_right", "DNg100_R"}:
        if by_side["R"]:
            chosen = np.asarray(by_side["R"], dtype=np.int32)
            note = "annotation-right instance DNg100_R"
    bodies = [int(graph.neuron_ids[i]) for i in np.asarray(chosen).tolist()]
    instances = []
    for entry in (malecns_block.get("left"), malecns_block.get("right")):
        if malecns_body_id_of(entry) in bodies:
            instances.append((entry or {}).get("instance") or "")
    return {
        "dataset": "MaleCNS_v1" if graph.n >= 10_000 else "toy",
        "malecns_body_ids": bodies,
        "body_ids": bodies,
        "indices": np.asarray(chosen, dtype=np.int32),
        "instances": instances,
        "stim": stim,
        "note": note,
        "pugliese_reference": dict(PUGLIESE_DNG100_STIM),
        "all_n": int(all_idx.size),
    }


def malecns_available() -> bool:
    cache = DEFAULT_DATA / "normalized" / "graph.npz"
    edges = DEFAULT_DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    return cache.exists() or edges.exists()


def load_graph(connectome: str, seed: int):
    if connectome in {"synthetic", "toy", "miniature"}:
        return miniature_connectome(seed)
    if connectome in {"tiny_cpg", "cpg_subgraph"}:
        return tiny_cpg_connectome()
    if connectome in {"malecns", "malecns_v1", "full"}:
        return VirtualFly._load_graph("malecns", seed=seed)
    raise ValueError(f"Unknown connectome '{connectome}'")


def _silence(net, indices: np.ndarray, silent: bool) -> None:
    idx = np.asarray(indices, dtype=np.int32)
    if idx.size:
        net.lesion(idx, silent=silent)


def run_condition(
    net,
    circuit: WalkingCircuit,
    *,
    dng100: np.ndarray,
    current: float,
    steps: int,
    warmup: int,
    source: str,
    lesion: np.ndarray | None = None,
) -> dict:
    net.reset()
    net.clear_drive()
    _silence(net, lesion if lesion is not None else np.zeros(0, dtype=np.int32), False)
    if lesion is not None and lesion.size:
        _silence(net, lesion, True)
    rec = allocate_traces(circuit, steps)
    dt = float(net.params.dt)
    for t in range(steps):
        if t == warmup and dng100.size:
            net.add_drive(dng100, current, source=source)
        net.step(1)
        record_tick(rec, net, circuit, t, (t + 1) * dt)
    if lesion is not None and lesion.size:
        _silence(net, lesion, False)
    stim_onset = warmup * dt
    summary = rec.summary(dt, stim_onset)
    return {
        "summary": summary,
        "preview": rec.previews(),
        "recording": rec,
        "stim_onset_ms": stim_onset,
        "steps": steps,
        "warmup": warmup,
    }


def _lesion_effect(intact: dict, lesioned: dict) -> dict:
    def _core_score(summary: dict) -> float:
        scores = []
        for slot, row in (summary.get("legs") or {}).items():
            for role in ("E1", "E2", "I1", "MN"):
                scored = row.get(role) or {}
                scores.append(float(scored.get("score") or 0.0))
        return float(np.mean(scores)) if scores else 0.0

    before = _core_score(intact)
    after = _core_score(lesioned)
    return {
        "intact_mean_score": before,
        "lesion_mean_score": after,
        "collapsed": bool(
            intact.get("any_leg_oscillatory") and not lesioned.get("any_leg_oscillatory")
        ),
        "score_drop": float(before - after),
        "interpretable": bool(
            intact.get("any_leg_oscillatory")
            and not intact.get("any_exploding")
            and intact.get("allow_lesions", False)
        ),
    }


def _population_metrics(summary: dict) -> dict:
    legs = summary.get("legs") or {}
    e1_scores, e2_scores, i1_scores, i2_scores, mn_scores = [], [], [], [], []
    freqs = []
    mn_means = []
    n_active = 0
    exploding = bool(summary.get("any_exploding") or summary.get("dng100_voltage_exploding"))
    valid = bool(summary.get("valid_for_rhythm_analysis", False)) and not exploding
    for row in legs.values():
        for role, bucket in (("E1", e1_scores), ("E2", e2_scores), ("I1", i1_scores), ("I2", i2_scores), ("MN", mn_scores)):
            scored = row.get(role) or {}
            raw = scored.get("score")
            bucket.append(None if raw is None else float(raw or 0.0))
            hz = scored.get("dominant_frequency") or scored.get("peak_hz")
            if hz and valid:
                freqs.append(float(hz))
            if float(scored.get("mean") or 0.0) > 0.05 or scored.get("oscillatory"):
                n_active += 1
            if role == "MN":
                mn_means.append(float(scored.get("mean") or 0.0))
    mean_freq = float(np.mean(freqs)) if freqs else None
    if not valid:
        return {
            "model_id": SHIU_LIF_SANITY_MODEL,
            "valid_dynamics": False,
            "valid_for_rhythm_analysis": False,
            "allow_lesions": False,
            "dng100_voltage_physiological": bool(summary.get("dng100_voltage_physiological", False)),
            "dng100_dynamics_valid": bool(summary.get("dng100_dynamics_valid", False)),
            "dng100_voltage_exploding": bool(summary.get("dng100_voltage_exploding", True)),
            "fft_executed": False,
            "dominant_frequency": None,
            "rhythmicity_score": None,
            "rhythmicity_score_E1": None,
            "rhythmicity_score_E2": None,
            "rhythmicity_score_I1": None,
            "rhythmicity_score_I2": None,
            "rhythmicity_score_MN": None,
            "spectral_peak_power_mean": None,
            "autocorrelation_peak_mean": None,
            "mean_motor_firing_rate": float(np.mean(mn_means)) if mn_means else 0.0,
            "active_neuron_count": int(n_active),
            "frequency_below_published_band": False,
            "invalid_reason": "non-physiological or non-finite voltage; FFT/autocorr peaks are not a CPG rhythm",
        }

    def _mean(values: list) -> float:
        nums = [float(v) for v in values if v is not None]
        return float(np.mean(nums)) if nums else 0.0

    return {
        "valid_dynamics": True,
        "valid_for_rhythm_analysis": True,
        "allow_lesions": True,
        "dominant_frequency": mean_freq,
        "rhythmicity_score": _mean(e1_scores + e2_scores + i1_scores + mn_scores),
        "spectral_peak_power_mean": float(
            np.mean(
                [
                    float((row.get(role) or {}).get("spectral_peak_power") or 0.0)
                    for row in legs.values()
                    for role in ("E1", "E2", "I1", "MN")
                ]
            )
        )
        if legs
        else 0.0,
        "autocorrelation_peak_mean": _mean(e1_scores + e2_scores + i1_scores + mn_scores),
        "rhythmicity_score_E1": _mean(e1_scores),
        "rhythmicity_score_E2": _mean(e2_scores),
        "rhythmicity_score_I1": _mean(i1_scores),
        "rhythmicity_score_I2": _mean(i2_scores),
        "rhythmicity_score_MN": _mean(mn_scores),
        "mean_motor_firing_rate": float(np.mean(mn_means)) if mn_means else 0.0,
        "active_neuron_count": int(n_active),
        "frequency_below_published_band": bool(mean_freq is not None and mean_freq < 7.0),
        "invalid_reason": None,
    }


def _plot_traces(recording, path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    t = np.asarray(recording.t_ms)
    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    series = [
        ("DNg100 stimulated", recording.dng100),
        ("E1 LF", recording.legs.get("FL", {}).get("E1")),
        ("E2 LF", recording.legs.get("FL", {}).get("E2")),
        ("I1 LF", recording.legs.get("FL", {}).get("I1")),
        ("MN LF", recording.legs.get("FL", {}).get("MN")),
        ("I2 LF", recording.legs.get("FL", {}).get("I2")),
    ]
    for ax, (title, y) in zip(axes, series):
        y = np.asarray(y) if y is not None else np.zeros_like(t)
        ax.plot(t, y, color="black", lw=0.8)
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle("MaleCNS DNg100 stim (no FlyBody)", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run(
    *,
    connectome: str = "toy",
    seed: int = 1,
    current: float = 40.0,
    steps: int | None = None,
    warmup: int | None = None,
    lesions: bool = True,
    scramble: bool = True,
    stim: str = "left_vnc",
    out: Path = OUT,
) -> dict:
    graph = load_graph(connectome, seed)
    assert_lif_sanity()
    tiny = connectome in {"tiny_cpg", "cpg_subgraph"}
    if connectome in {"malecns", "malecns_v1", "full"}:
        cpg = tiny_cpg_numerical_sanity()
        if not cpg["ok"]:
            raise RuntimeError(
                "Tiny CPG is not numerically sane. Refusing MaleCNS. "
                + str(cpg.get("next") or "")
            )
    if tiny:
        lesions = False
        scramble = False
    if steps is None:
        steps = 120 if tiny else (300 if graph.n < 1000 else 500)
    if warmup is None:
        warmup = min(50, max(10, steps // 8))
    params = shiu_lif_params(dt=1.0)
    net = MixedDynamicsNetwork(graph, params=params, seed=seed)
    circuit = WalkingCircuit(graph)
    stim_info = stimulated_dng100(graph, circuit, stim=stim)
    dng100 = stim_info["indices"]
    source = "experiment.optogenetic.DNg100"
    intact = run_condition(
        net,
        circuit,
        dng100=dng100,
        current=current,
        steps=steps,
        warmup=warmup,
        source=source,
    )
    interpretation = interpret_intact(intact["summary"])
    conditions = {
        "intact": {
            "summary": intact["summary"],
            "preview": intact["preview"],
            "stim_onset_ms": intact["stim_onset_ms"],
            "metrics": _population_metrics(intact["summary"]),
        }
    }
    lesion_report = {}
    if lesions or scramble:
        allow = bool(
            intact["summary"].get("allow_lesions", intact["summary"].get("valid_dynamics", True))
        ) and not intact["summary"].get("any_exploding")
        if not allow:
            lesion_report["skipped"] = {
                "reason": (
                    "Intact voltage is non-physiological or non-finite. "
                    "Lesions and scramble are uninterpretable. Fix isolated LIF sanity first."
                ),
                "requested_lesions": bool(lesions),
                "requested_scramble": bool(scramble),
            }
            lesions = False
            scramble = False
    if lesions:
        expected = {
            "E1": "rhythm should collapse strongly (Pugliese: E1 necessary)",
            "E2": "rhythm should collapse strongly (Pugliese: E2 necessary)",
            "I1": "rhythm may persist (Pugliese: I1 and I2 appear partly redundant)",
            "I2": "rhythm may persist (Pugliese: I1 and I2 appear partly redundant)",
            "I1_I2": "especially interesting; both inhibitory cells silenced",
        }
        i1_idx = circuit.indices("I1")
        i2_idx = circuit.indices("I2")
        both_inh = (
            np.unique(np.concatenate([i1_idx, i2_idx]))
            if i1_idx.size or i2_idx.size
            else np.zeros(0, dtype=np.int32)
        )
        for name, idx in (("E1", circuit.indices("E1")), ("E2", circuit.indices("E2")), ("I1", i1_idx), ("I2", i2_idx), ("I1_I2", both_inh)):
            result = run_condition(
                net,
                circuit,
                dng100=dng100,
                current=current,
                steps=steps,
                warmup=warmup,
                source=source,
                lesion=idx,
            )
            conditions[f"lesion_{name}"] = {
                "summary": result["summary"],
                "preview": result["preview"],
                "n_silenced": int(idx.size),
                "metrics": _population_metrics(result["summary"]),
            }
            lesion_report[name] = _lesion_effect(intact["summary"], result["summary"])
            lesion_report[name]["published_expectation"] = expected[name]
            if not lesion_report[name]["interpretable"]:
                lesion_report[name]["note"] = (
                    "Intact network did not oscillate; lesion comparison is "
                    "not a replication of the preprint necessity result."
                )
        if scramble:
            rng = np.random.default_rng(seed + 17)
            shuffled = shuffled_connectome(graph, rng)
            shuffled_net = MixedDynamicsNetwork(shuffled, params=params, seed=seed)
            shuffled_circuit = WalkingCircuit(shuffled)
            shuffled_stim = stimulated_dng100(shuffled, shuffled_circuit, stim=stim)
            result = run_condition(
                shuffled_net,
                shuffled_circuit,
                dng100=shuffled_stim["indices"],
                current=current,
                steps=steps,
                warmup=warmup,
                source=source,
            )
            conditions["scrambled_connectome"] = {
                "summary": result["summary"],
                "preview": result["preview"],
                "control": "shuffled_post_indices",
                "metrics": _population_metrics(result["summary"]),
            }
            lesion_report["scrambled"] = _lesion_effect(intact["summary"], result["summary"])
            lesion_report["scrambled"]["published_expectation"] = (
                "Real topology should support rhythm more than a degree-matched shuffle"
            )

    anatomy = circuit.anatomy()
    catalog = circuit.catalog()
    mapping = load_cpg_mapping() if graph.n >= 10_000 else {}
    metrics = _population_metrics(intact["summary"])
    if metrics.get("frequency_below_published_band"):
        interpretation["frequency_note"] = (
            f"Dominant frequency {metrics.get('dominant_frequency')} Hz is below "
            "the published 7–15 Hz locomotor band. Logged, not clamped."
        )
    payload = {
        "model_version": MODEL_VERSION,
        "policy": NO_SCAFFOLD.name,
        "connectome": connectome,
        "n_neurons": int(graph.n),
        "seed": seed,
        "dt_ms": float(params.dt),
        "steps": steps,
        "warmup": warmup,
        "current": current,
        "motor_mode": MotorMode.NEURAL_CPG.value,
        "joints_from_neural_cpg": NEURAL_CPG_DRIVES_JOINTS,
        "engineered_cpg_used": False,
        "walk_api_called": False,
        "graph_modified": False,
        "dynamics_retuned": False,
        "dynamics_model": SHIU_LIF_SANITY_MODEL,
        "pugliese_cpg_model": PUGLIESE_CPG_MODEL,
        "is_pugliese_reproduction": False,
        "voltage_unit": VOLTAGE_UNIT,
        "wsyn_mv": WSYN_MV,
        "walking_circuit_types": dict(WALKING_CIRCUIT_TYPES),
        "identity_lock": {
            "CORE_CPG_TYPES": dict(CORE_CPG_TYPES),
            "MALECNS_DNG100_BODY_IDS": sorted(MALECNS_DNG100_BODY_IDS),
            "PUGLIESE_MANC_STIM_INDEX": PUGLIESE_MANC_STIM_INDEX,
            "PUGLIESE_MANC_STIM_BODY": PUGLIESE_MANC_STIM_BODY,
            "I2": CORE_CPG_TYPES["I2"],
        },
        "e5_type_provenance": dict(E5_TYPE_PROVENANCE),
        "i2_type_provenance": dict(I2_TYPE_PROVENANCE),
        "rhythmicity_threshold": RHYTHMICITY_THRESHOLD,
        "published_walk_hz": [7.0, 15.0],
        "source_status": "bioRxiv preprint (Pugliese et al. 2025), not a peer-reviewed article",
        "dng100_n": int(circuit.indices("DNg100").size),
        "dng100_is_six_neurons": False,
        "malecns_dng100_body_ids": [int(graph.neuron_ids[i]) for i in circuit.indices("DNg100").tolist()[:8]],
        "dng100_sides": [str(graph.side[i]) for i in circuit.indices("DNg100").tolist()[:8]],
        "stimulated": stim_info,
        "cpg_mapping_path": str(DEFAULT_DATA / "cpg_mapping.json"),
        "cpg_mapping_roles": {
            name: {
                slot: malecns_body_id_of(entry)
                for slot, entry in ((mapping.get(WALKING_CIRCUIT_TYPES[name]) or {}).get("neurons") or {}).items()
            }
            for name in ("E1", "E2", "I1", "I2", "E3", "E4", "E5")
            if mapping
        },
        "paper_slots": {slot: INTERNAL_TO_PAPER.get(slot, slot) for slot in LEG_SLOTS},
        "catalog": {
            name: catalog[name]
            for name in ("DNg100", "DNb08", "E1", "E2", "I1", "I2", "E3", "E4", "E5")
            if name in catalog
        },
        "leg_copies": {slot: circuit.legs[slot].as_dict() for slot in LEG_SLOTS},
        "anatomy_core": {
            "dng100_to_E1": anatomy["dng100_to_E1"],
            "dng100_to_each_E1": anatomy.get("dng100_to_each_E1"),
            "E1_to_E2": anatomy["E1_to_E2"],
            "I1_to_E1": anatomy["I1_to_E1"],
            "n_leg_copies_with_E1": anatomy["n_leg_copies_with_E1"],
        },
        "metrics": metrics,
        "valid_dynamics": bool(metrics.get("valid_dynamics")),
        "valid_for_rhythm_analysis": bool(metrics.get("valid_for_rhythm_analysis")),
        "allow_lesions": bool(metrics.get("allow_lesions")),
        "discarded_prior_malecns_run": {
            "scientific_status": "invalid",
            "i2_type": "IN19B007",
            "src": "toy_single_ipsilateral_as_T1",
            "dng100_voltage": {"mean": -421.47, "min": -589.32, "max": -379.24},
            "dng100_spikes": 0,
            "dominant_frequency": 17.427,
            "reason": (
                "Wrong I2 identity and non-physiological DNg100 voltage. "
                "Do not interpret the 17.4 Hz peak. Canonical I2 is IN19A007; "
                "MaleCNS DNg100 is {10045, 10056}."
            ),
        },
        "authors_vs_ours": {
            "authors": {
                "graph": "MANC T1 DN-to-MN subgraph, 4604 neurons",
                "dynamics_model": PUGLIESE_CPG_MODEL,
                "equations": "rate ODE dR/dt = (half-tanh(WR+I) - R)/tau with cell-size normalization",
                "tau_s": 0.02,
                "threshold": 7.5,
                "weight_multiplier": 0.03,
                "stim_current": PUGLIESE_CPG_STIM_AMPLITUDE,
                "stim_index_in_their_W": 31,
                "stim_manc_body_id": 10093,
                "note": (
                    "stimNeurons [[31]] is MANC T1 matrix index 31 = MANC body 10093 "
                    "(type DNg100). Same biological type as MaleCNS DNg100, not the "
                    "same body ID. MANC body 10056 is vMS16 in that table. "
                    "Amplitude 250 is this rate-ODE convention; size normalization "
                    "changes the required DNg100 stim. Do not copy 250 into SHIU_LIF."
                ),
                "T_s": 2.0,
                "oscillation_threshold": 0.5,
            },
            "ours": {
                "graph": (
                    "tiny CPG subgraph"
                    if tiny
                    else "full MaleCNS sparse graph, ordinary edges, no extra CPG wiring"
                    if graph.n >= 10_000
                    else "toy/miniature graph"
                ),
                "dynamics_model": SHIU_LIF_SANITY_MODEL,
                "equations": "MixedDynamicsNetwork current-based LIF (mV) + graded VNC premotor",
                "not_a_pugliese_reproduction": True,
                "dt_ms": float(params.dt),
                "wsyn_mv": float(params.wsyn_mv),
                "contact_gain": float(params.contact_gain),
                "voltage_unit": VOLTAGE_UNIT,
                "stim_current": current,
                "stimulated": stim_info["note"],
                "malecns_body_ids": stim_info.get("malecns_body_ids") or stim_info.get("body_ids"),
                "pugliese_reference": stim_info.get("pugliese_reference"),
                "retuned": False,
            },
        },
        "dataset_validation": dataset_validation(graph, policy_name=NO_SCAFFOLD.name, net=net),
        "conditions": conditions,
        "lesions": lesion_report,
        **interpretation,
        "computational_graph": computational_graph_manifest(),
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(payload), indent=2))
    arrays = {
        "t_ms": intact["recording"].t_ms,
        "DNg100": intact["recording"].dng100,
        "DNg100_v": intact["recording"].dng100_v,
        "DNg100_L": intact["recording"].dng100_l,
        "DNg100_R": intact["recording"].dng100_r,
        "DNg100_L_v": intact["recording"].dng100_l_v,
        "DNg100_R_v": intact["recording"].dng100_r_v,
    }
    for slot, traces in intact["recording"].legs.items():
        for role, tr in traces.items():
            arrays[f"{slot}.{role}"] = tr
    trace_path = TRACES if out == OUT else out.with_suffix(".npz")
    np.savez_compressed(trace_path, **arrays)
    plot_path = _plot_traces(intact["recording"], trace_path.with_suffix(".png"))
    payload["output"] = str(out)
    payload["traces"] = str(trace_path)
    payload["trace_plot"] = str(plot_path) if plot_path else None
    out.write_text(json.dumps(_json_ready(payload), indent=2))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome", default="toy", choices=("toy", "synthetic", "tiny_cpg", "malecns"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--current", type=float, default=40.0)
    parser.add_argument("--steps", type=int, default=0, help="0 = toy 300 / tiny CPG 120 / MaleCNS 500")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-lesions", action="store_true")
    parser.add_argument("--no-scramble", action="store_true")
    parser.add_argument("--stim", default="left_vnc", help="left_vnc | right_vnc | both | soma_left | soma_right")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument(
        "--sanity",
        action="store_true",
        help="SHIU_LIF_SANITY_MODEL isolated tests. Does not load MaleCNS.",
    )
    parser.add_argument(
        "--tiny-cpg",
        action="store_true",
        help="After isolated tests, run DNg100+E1/E2/E3/I1/I2 numerical sanity. Not MaleCNS.",
    )
    args = parser.parse_args(argv)
    if args.sanity:
        print(format_lif_sanity(run_lif_sanity()))
        return 0
    if args.tiny_cpg:
        print(format_lif_sanity(run_lif_sanity()))
        print(format_tiny_cpg(tiny_cpg_numerical_sanity()))
        return 0
    if args.connectome == "malecns" and not malecns_available():
        raise SystemExit("MaleCNS files are not in data/malecns_v1")
    print(format_policy_banner(NO_SCAFFOLD))
    print()
    print("DNg100 CPG rhythm (MODE_NEURAL_CPG, joints not actuated)")
    print(f"connectome={args.connectome} current={args.current} stim={args.stim}")
    result = run(
        connectome=args.connectome,
        seed=args.seed,
        current=args.current,
        steps=args.steps or None,
        warmup=args.warmup or None,
        lesions=not args.no_lesions,
        scramble=not args.no_scramble,
        stim=args.stim,
        out=Path(args.out),
    )
    print(f"n={result['n_neurons']} steps={result['steps']} DNg100 n={result['dng100_n']}")
    print(f"stimulated: {result.get('stimulated', {})}")
    print(f"metrics: {result.get('metrics')}")
    print(f"dynamics_model={result.get('dynamics_model')} is_pugliese_reproduction={result.get('is_pugliese_reproduction')}")
    if not result.get("valid_dynamics", result.get("valid_for_rhythm_analysis", True)):
        print("DYNAMICS INVALID — dominant_frequency and rhythmicity_score are None; lesions forbidden")
    print(f"E5 canonical type: {result['walking_circuit_types']['E5']}")
    print(f"leg copies with E1: {result['anatomy_core']['n_leg_copies_with_E1']}")
    print(f"question: {result['question']}")
    print(f"answer: {result['answer']}")
    print(f"oscillation_reproduced: {result['oscillation_reproduced']}")
    print(f"graph_modified: {result['graph_modified']} dynamics_retuned: {result['dynamics_retuned']}")
    intact = result["conditions"]["intact"]["summary"]
    dng_v = intact.get("DNg100_v") or {}
    print(
        f"DNg100 spikes mean={intact['DNg100'].get('mean'):.3f} "
        f"V mean={dng_v.get('mean')} min={dng_v.get('min')} max={dng_v.get('max')} "
        f"exploding={dng_v.get('exploding') or intact.get('any_exploding')}"
    )
    print(
        f"oscillatory legs: {intact.get('n_oscillatory_legs')} "
        f"tonic_core: {intact.get('core_tonic_plateau')} exploding={intact.get('any_exploding')}"
    )
    for slot in LEG_SLOTS:
        row = (intact.get("legs") or {}).get(slot) or {}
        e1 = row.get("E1") or {}
        if not e1:
            continue
        score = e1.get("score")
        print(
            f"  {slot} E1 score={score} hz={e1.get('peak_hz')} "
            f"tonic={e1.get('tonic_plateau')} osc={e1.get('oscillatory')}"
        )
    if result.get("lesions"):
        for name, row in result["lesions"].items():
            print(
                f"lesion {name}: collapsed={row.get('collapsed')} "
                f"interpretable={row.get('interpretable')} drop={row.get('score_drop'):.3f}"
            )
    print(result["next_step"])
    print(f"wrote {result['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sustained DNg100 current → per-leg VNC CPG traces.

Question: does MixedDynamicsNetwork on this MaleCNS graph reproduce the
bioRxiv DNg100 → E1/E2/inhibition → motor rhythm (Pugliese et al. 2025)?

Stimulate the two real DNg100 body IDs. Let ordinary MaleCNS edges
propagate. Record each E1/E2/I1 copy in its T1/T2/T3 × L/R slot.
Do not hand-wire pairwise gains. Do not retune the graph. Do not drive
FlyBody.

Use the project interpreter (`.venv/bin/python` or
`python -m experiment.dng100_cpg_rhythm`), not macOS system Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.loader import DEFAULT_DATA, computational_graph_manifest
from flybrain.network import MixedDynamicsNetwork, LIFParams, dataset_validation
from organism.config import MODEL_VERSION, MotorMode, NO_SCAFFOLD, format_policy_banner
from organism.cpg_rhythm import (
    RHYTHMICITY_THRESHOLD,
    allocate_traces,
    interpret_intact,
    record_tick,
)
from organism.fly import VirtualFly, _json_ready
from organism.toy import miniature_connectome
from organism.walking_pathways import (
    E5_TYPE_PROVENANCE,
    LEG_SLOTS,
    NEURAL_CPG_DRIVES_JOINTS,
    WALKING_CIRCUIT_TYPES,
    WalkingCircuit,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dng100_cpg_rhythm.json"
TRACES = ROOT / "outputs" / "dng100_cpg_rhythm_traces.npz"


def malecns_available() -> bool:
    cache = DEFAULT_DATA / "normalized" / "graph.npz"
    edges = DEFAULT_DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    return cache.exists() or edges.exists()


def load_graph(connectome: str, seed: int):
    if connectome in {"synthetic", "toy", "miniature"}:
        return miniature_connectome(seed)
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
            intact.get("any_leg_oscillatory") and not intact.get("any_exploding")
        ),
    }


def run(
    *,
    connectome: str = "toy",
    seed: int = 1,
    current: float = 40.0,
    steps: int | None = None,
    warmup: int | None = None,
    lesions: bool = True,
    out: Path = OUT,
) -> dict:
    graph = load_graph(connectome, seed)
    if steps is None:
        steps = 300 if graph.n < 1000 else 500
    if warmup is None:
        warmup = min(50, max(10, steps // 8))
    params = LIFParams(dt=1.0)
    net = MixedDynamicsNetwork(graph, params=params, seed=seed)
    circuit = WalkingCircuit(graph)
    dng100 = circuit.indices("DNg100")
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
        }
    }
    lesion_report = {}
    if lesions:
        for name, key in (("E1", "E1"), ("E2", "E2"), ("I1", "I1")):
            idx = circuit.indices(key)
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
            }
            lesion_report[name] = _lesion_effect(intact["summary"], result["summary"])
            expected = {
                "E1": "rhythm should collapse strongly (preprint full-network necessity)",
                "E2": "rhythm should collapse strongly (preprint full-network necessity)",
                "I1": "effect may be weaker; other inhibitory cells can compensate",
            }[name]
            lesion_report[name]["published_expectation"] = expected
            if not lesion_report[name]["interpretable"]:
                lesion_report[name]["note"] = (
                    "Intact network did not oscillate; lesion comparison is "
                    "not a replication of the preprint necessity result."
                )

    anatomy = circuit.anatomy()
    catalog = circuit.catalog()
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
        "walking_circuit_types": dict(WALKING_CIRCUIT_TYPES),
        "e5_type_provenance": dict(E5_TYPE_PROVENANCE),
        "rhythmicity_threshold": RHYTHMICITY_THRESHOLD,
        "published_walk_hz": [7.0, 15.0],
        "source_status": "bioRxiv preprint (Pugliese et al. 2025), not a peer-reviewed article",
        "dng100_n": int(dng100.size),
        "dng100_is_six_neurons": False,
        "dng100_body_ids": [int(graph.neuron_ids[i]) for i in dng100.tolist()[:8]],
        "dng100_sides": [str(graph.side[i]) for i in dng100.tolist()[:8]],
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
        "authors_vs_ours": {
            "authors": {
                "graph": "MANC T1 DN-to-MN subgraph, 4604 neurons",
                "equations": "rate ODE dR/dt = (half-tanh(WR+I) - R)/tau",
                "tau_s": 0.02,
                "threshold": 7.5,
                "weight_multiplier": 0.03,
                "stim_current": 250,
                "stim_index_in_their_W": 31,
                "T_s": 2.0,
                "oscillation_threshold": 0.5,
            },
            "ours": {
                "graph": "full MaleCNS sparse graph, ordinary edges, no extra CPG wiring",
                "equations": "MixedDynamicsNetwork LIF + graded VNC premotor",
                "dt_ms": float(params.dt),
                "contact_gain": float(params.contact_gain),
                "stim_current": current,
                "stimulated": "both DNg100 body IDs",
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
    np.savez_compressed(TRACES if out == OUT else out.with_suffix(".npz"), **arrays)
    payload["output"] = str(out)
    payload["traces"] = str(TRACES if out == OUT else out.with_suffix(".npz"))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome", default="toy", choices=("toy", "synthetic", "malecns"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--current", type=float, default=40.0)
    parser.add_argument("--steps", type=int, default=0, help="0 = toy 300 / MaleCNS 500")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--no-lesions", action="store_true")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    if args.connectome == "malecns" and not malecns_available():
        raise SystemExit("MaleCNS files are not in data/malecns_v1")
    print(format_policy_banner(NO_SCAFFOLD))
    print()
    print("DNg100 CPG rhythm (MODE_NEURAL_CPG, joints not actuated)")
    print(f"connectome={args.connectome} current={args.current}")
    result = run(
        connectome=args.connectome,
        seed=args.seed,
        current=args.current,
        steps=args.steps or None,
        warmup=args.warmup or None,
        lesions=not args.no_lesions,
        out=Path(args.out),
    )
    print(f"n={result['n_neurons']} steps={result['steps']} DNg100 n={result['dng100_n']}")
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
        print(
            f"  {slot} E1 score={e1.get('score'):.3f} hz={e1.get('peak_hz')} "
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

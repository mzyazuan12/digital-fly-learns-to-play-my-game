"""experiment/walking_dn_investigator

Keep the DNp09 experiment. This one asks a different question:

    Do identified walking DNs and the published VNC CPG exist in this
    connectome, and does driving them change CPG / motor-neuron activity?

It does not add walk(), stand(), or a timer. Legs still move through the
engineered FlyGym CPG until MODE_NEURAL_MOTOR exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.loader import DEFAULT_DATA, Connectome, computational_graph_manifest
from flybrain.network import LIFNetwork, LIFParams, dataset_validation
from organism.bridge import MotorBridge
from organism.config import NO_SCAFFOLD, MODEL_VERSION, format_policy_banner
from organism.fly import VirtualFly, _json_ready
from organism.toy import miniature_connectome
from organism.walking_pathways import WalkingCircuit, ENGINEERED_CPG_STILL_EXECUTES

ROOT = Path(__file__).resolve().parents[1]

PROBE_DNS = (
    "DNp09",
    "DNg100",
    "DNb08",
    "oDN1",
    "DNa01",
    "DNa02",
    "MDN",
    "bluebell",
    "brake",
    "foxglove",
)


def malecns_available() -> bool:
    cache = DEFAULT_DATA / "normalized" / "graph.npz"
    edges = DEFAULT_DATA / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    return cache.exists() or edges.exists()


def load_graph(connectome: str, seed: int) -> Connectome:
    if connectome in {"synthetic", "toy", "miniature"}:
        return miniature_connectome(seed)
    if connectome in {"malecns", "malecns_v1", "full"}:
        return VirtualFly._load_graph("malecns", seed=seed)
    raise ValueError(f"Unknown connectome '{connectome}'")


def probe_pathway(
    graph: Connectome,
    indices: np.ndarray,
    *,
    source: str,
    seed: int = 1,
    current: float = 40.0,
    steps: int = 10,
) -> dict:
    """Optogenetic-style current. Experimental probe, not a timer or gait command."""
    params = LIFParams(dt=1.0)
    net = LIFNetwork(graph, params=params, seed=seed)
    bridge = MotorBridge(graph, legacy_scaffold=False, policy=NO_SCAFFOLD)
    circuit = bridge.circuit
    analog = getattr(net, "graded_release", net.graded_output)
    baseline_counts = np.zeros(graph.n, dtype=np.int32)
    baseline = circuit.rates(baseline_counts, 0.01, analog)

    def phase() -> dict:
        net.reset()
        net.clear_drive()
        if indices.size:
            net.add_drive(indices, current, source=source)
        bridge.reset_traces()
        counts = net.step(steps)
        cmd = bridge.read(
            counts,
            steps * 0.001,
            net=net,
            external_command=source,
            motor_mode="MODE_ENGINEERED_CPG",
        )
        analog_now = getattr(net, "graded_release", net.graded_output)
        duration_s = steps * 0.001
        rates = circuit.rates(counts, duration_s, analog_now)
        return {
            "mode": cmd.mode,
            "walk_hz": cmd.walk_hz,
            "locomotor_drive": cmd.locomotor_drive,
            "steering_drive": cmd.steering_drive,
            "left": cmd.left,
            "right": cmd.right,
            "scaffold_used": cmd.scaffold_used,
            "neural_only": cmd.neural_only,
            "n_driven": int(indices.size),
            "spikes_driven": int(counts[indices].sum()) if indices.size else 0,
            "n_spike_events": int(net.last_n_spike_events),
            "n_graded_deliveries": int(net.last_n_graded_deliveries),
            "rates": rates,
        }

    if indices.size == 0:
        return {
            "present": False,
            "driven": {
                "mode": "rest",
                "walk_hz": 0.0,
                "locomotor_drive": 0.0,
                "scaffold_used": False,
                "n_driven": 0,
                "spikes_driven": 0,
                "rates": baseline,
            },
            "engineered_cpg_engaged": False,
            "cpg_or_mn_responded": False,
        }
    driven = phase()
    e1 = float(driven["rates"]["E1"]["analog"])
    e2 = float(driven["rates"]["E2"]["analog"])
    mn = float(driven["rates"]["vnc_motor"]["hz"])
    return {
        "present": True,
        "driven": driven,
        "engineered_cpg_engaged": driven["mode"] in {"walk", "reverse"} and not driven["scaffold_used"],
        "cpg_or_mn_responded": e1 > 1e-4 or e2 > 1e-4 or mn > 0.0,
        "source": source,
    }


def evaluate_anatomy(anatomy: dict, catalog: dict) -> dict[str, bool]:
    return {
        "DNg100 present": bool(catalog["DNg100"]["resolved"]),
        "core CPG types present": bool(anatomy["core_cpg_present"]),
        "DNg100 contacts E1": int(anatomy["dng100_to_E1"]["contacts"]) > 0,
        "E1-E2 recurrence present": int(anatomy["E1_to_E2"]["contacts"]) + int(anatomy["E2_to_E1"]["contacts"])
        > 0,
        "I1 contacts E1 or E2": int(anatomy["I1_to_E1"]["contacts"]) + int(anatomy["I1_to_E2"]["contacts"]) > 0,
        "CPG contacts motor neurons": int(anatomy["cpg_core_to_vnc_motor"]["contacts"]) > 0,
        "FlyGym CPG still executes joints": bool(anatomy["engineered_cpg_still_executes_joints"]),
        "neural VNC CPG does not drive joints yet": anatomy["neural_vnc_cpg_drives_joints"] is False,
    }


def run(
    *,
    connectome: str = "malecns",
    seed: int = 1,
    out: Path | None = None,
    probe_steps: int | None = None,
) -> dict:
    used = connectome
    if connectome in {"malecns", "malecns_v1", "full"} and not malecns_available():
        used = "synthetic"
    if probe_steps is None:
        probe_steps = 8 if used.startswith("male") else 12
    manifest = computational_graph_manifest()
    graph = load_graph(used, seed)
    bridge = MotorBridge(graph, legacy_scaffold=False, policy=NO_SCAFFOLD)
    circuit = bridge.circuit
    print(format_policy_banner(NO_SCAFFOLD), flush=True)
    validation = dataset_validation(graph, models=None, policy_name=NO_SCAFFOLD.name)
    catalog = circuit.catalog()
    anatomy = circuit.anatomy()
    probes = {}
    for name in PROBE_DNS:
        idx = circuit.indices(name)
        probes[name] = probe_pathway(
            graph,
            idx,
            source=f"experiment.optogenetic.{name}",
            seed=seed,
            steps=probe_steps,
        )
        print(
            f"probe {name}: n={int(idx.size)} mode={probes[name]['driven']['mode']} "
            f"cpg/mn={probes[name]['cpg_or_mn_responded']}",
            flush=True,
        )
    missing = [name for name, row in catalog.items() if not row["resolved"]]
    result = {
        "experiment": "walking_dn_investigator",
        "model_version": MODEL_VERSION,
        "connectome_requested": connectome,
        "connectome_used": used,
        "full_malecns": used in {"malecns", "malecns_v1", "full"} and graph.n >= 100_000,
        "neurons": graph.n,
        "edges": graph.n_edges,
        "seed": seed,
        "policy": NO_SCAFFOLD.as_dict(),
        "consciousness_claimed": False,
        "dataset_validation": validation,
        "computational_graph": manifest,
        "synapse_coordinate_tables_loaded": False,
        "catalog": catalog,
        "missing_types": missing,
        "anatomy": anatomy,
        "probes": probes,
        "engineered_cpg_still_executes": ENGINEERED_CPG_STILL_EXECUTES,
        "neural_vnc_cpg_drives_joints": False,
        "no_walk_function": True,
        "target_pathway": (
            "MaleCNS brain → DNg100/DNb08/other walking DNs → VNC E1/E2/I1 "
            "→ leg motor neurons → muscles → FlyBody"
        ),
        "current_pathway": (
            "MaleCNS identified walking DNs → ENGINEERED_NEURAL_MOTOR_INTERFACE "
            "→ FlyGym HybridTurningController → FlyBody"
        ),
        "honesty": {
            "body_can_walk_if_driven": True,
            "male_cns_currently_generates_walking_rhythm": None,
            "hybrid_turning_controller_still_used": True,
            "preprogrammed_steps_still_used": True,
            "foxglove_cb0890_may_be_unlabeled": not catalog["foxglove"]["resolved"],
            "gap_junctions": "ABSENT",
            "functional_gain": "ASSUMED",
        },
        "success_criterion": (
            "This experiment succeeds if it reports which walking DNs and CPG "
            "cells exist, whether DNg100 contacts E1, and whether driving those "
            "cells changes CPG/MN activity — without a timer or named gait command. "
            "Silent CPG dynamics are a result."
        ),
    }
    result["anatomy_checks"] = evaluate_anatomy(anatomy, catalog)
    out = out or (ROOT / "outputs" / "walking_dn_investigator.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome", default="malecns", choices=("malecns", "synthetic"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    result = run(
        connectome=args.connectome,
        seed=args.seed,
        out=args.out,
        probe_steps=args.steps,
    )
    anatomy = result["anatomy"]
    print(
        json.dumps(
            {
                "connectome_used": result["connectome_used"],
                "neurons": result["neurons"],
                "missing_types": result["missing_types"],
                "DNg100_n": result["catalog"]["DNg100"]["n"],
                "E1_n": result["catalog"]["E1"]["n"],
                "dng100_to_E1_contacts": anatomy["dng100_to_E1"]["contacts"],
                "cpg_to_mn_contacts": anatomy["cpg_core_to_vnc_motor"]["contacts"],
                "DNg100_probe_mode": result["probes"]["DNg100"]["driven"]["mode"],
                "DNg100_cpg_or_mn": result["probes"]["DNg100"]["cpg_or_mn_responded"],
                "engineered_cpg_still_executes": result["engineered_cpg_still_executes"],
                "anatomy_checks": result["anatomy_checks"],
            },
            indent=2,
        )
    )
    required = (
        result["anatomy_checks"]["FlyGym CPG still executes joints"],
        result["anatomy_checks"]["neural VNC CPG does not drive joints yet"],
        result["probes"]["DNp09"]["driven"]["scaffold_used"] is False,
        result["probes"]["DNg100"]["driven"]["scaffold_used"] is False,
    )
    return 0 if all(required) else 1


if __name__ == "__main__":
    raise SystemExit(main())

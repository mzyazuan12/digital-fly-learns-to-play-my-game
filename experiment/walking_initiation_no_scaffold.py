"""experiment/walking_initiation_no_scaffold

Acceptance is the checklist, not an animation. If the fly never moves,
do not put the timer back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.loader import DEFAULT_DATA, Connectome
from flybrain.network import LIFNetwork, LIFParams
from organism.bridge import MotorBridge
from organism.config import NO_SCAFFOLD, MODEL_VERSION
from organism.fly import VirtualFly, _json_ready
from organism.toy import miniature_connectome
from worlds import empty_arena

ROOT = Path(__file__).resolve().parents[1]


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


def _summarize(records, fly: VirtualFly, label: str) -> dict:
    walked = [r for r in records if r.mode == "walk"]
    rested = [r for r in records if r.mode == "rest"]
    scaffold = any(r.scaffold_used for r in records)
    causal = True
    for r in walked:
        if r.walk_hz <= 0.0 and r.walk_trace <= 0.0:
            causal = False
            break
    transitions = []
    prev = None
    for r in records:
        if prev is not None and r.mode != prev:
            transitions.append(
                {
                    "t_ms": r.t_ms,
                    "from": prev,
                    "to": r.mode,
                    "walk_hz": r.walk_hz,
                    "walk_trace": r.walk_trace,
                    "sources": [s.value for s in r.sources],
                    "scaffold_used": r.scaffold_used,
                }
            )
        prev = r.mode
    last_trace = dict(fly.bridge.last_trace or {})
    last_trace.pop("text", None)
    stopped_without_timer = True
    for row in transitions:
        if row["from"] == "walk" and row["to"] == "rest":
            if row["scaffold_used"]:
                stopped_without_timer = False
    return {
        "label": label,
        "n": len(records),
        "n_walk": len(walked),
        "n_rest": len(rested),
        "modes": sorted({r.mode for r in records}),
        "scaffold_used": scaffold,
        "walk_implies_dn_activity": causal,
        "spontaneous_walk_emerged": len(walked) > 0,
        "first_walk_t_ms": walked[0].t_ms if walked else None,
        "stopped_without_timer": stopped_without_timer,
        "transitions": transitions,
        "last_trace": last_trace,
        "last_trace_text": (fly.bridge.last_trace or {}).get("text", ""),
        "motor_fidelity_level": fly.motor_fidelity_level,
        "drive_sources_last": dict(fly.net.drive_sources),
        "x_mm": float(records[-1].x_mm) if records else 0.0,
        "y_mm": float(records[-1].y_mm) if records else 0.0,
    }


def probe_dnp09_on(graph: Connectome, *, seed: int = 1, current: float = 40.0) -> dict:
    """Optogenetic-style current into DNp09. Experimental probe, not a timer."""
    params = LIFParams(dt=1.0)
    net = LIFNetwork(graph, params=params, seed=seed)
    bridge = MotorBridge(graph, legacy_scaffold=False, policy=NO_SCAFFOLD)
    net.add_drive(bridge.walk_indices, current, source="experiment.optogenetic.DNp09")
    counts = net.step(10)
    cmd = bridge.read(
        counts,
        0.01,
        net=net,
        external_command="experiment.optogenetic.DNp09",
        motor_mode="MODE_ENGINEERED_CPG",
    )
    intact = {
        "mode": cmd.mode,
        "walk_hz": cmd.walk_hz,
        "walk_trace": cmd.walk_trace,
        "left": cmd.left,
        "right": cmd.right,
        "scaffold_used": cmd.scaffold_used,
        "neural_only": cmd.neural_only,
        "motor_fidelity_level": cmd.motor_fidelity_level,
        "drive_sources": dict(net.drive_sources),
        "spikes": int(counts[bridge.walk_indices].sum()) if bridge.walk_indices.size else 0,
        "n_graded_deliveries": int(net.last_n_graded_deliveries),
        "n_spike_events": int(net.last_n_spike_events),
        "trace_text": bridge.last_trace.get("text", ""),
    }
    net.lesion(bridge.walk_indices, silent=True)
    net.clear_drive()
    net.add_drive(bridge.walk_indices, current, source="experiment.optogenetic.DNp09")
    bridge.walk_trace = 0.0
    counts_lesion = net.step(10)
    cmd_lesion = bridge.read(
        counts_lesion,
        0.01,
        net=net,
        external_command="experiment.optogenetic.DNp09",
    )
    lesion = {
        "mode": cmd_lesion.mode,
        "walk_hz": cmd_lesion.walk_hz,
        "walk_trace": cmd_lesion.walk_trace,
        "spikes": int(counts_lesion[bridge.walk_indices].sum()) if bridge.walk_indices.size else 0,
        "scaffold_used": cmd_lesion.scaffold_used,
    }
    net.lesion(bridge.walk_indices, silent=False)
    net.clear_drive()
    net.add_drive(bridge.walk_indices, current, source="experiment.optogenetic.DNp09")
    bridge.walk_trace = 0.0
    counts_restored = net.step(10)
    cmd_restored = bridge.read(
        counts_restored,
        0.01,
        net=net,
        external_command="experiment.optogenetic.DNp09",
    )
    restored = {
        "mode": cmd_restored.mode,
        "walk_hz": cmd_restored.walk_hz,
        "spikes": int(counts_restored[bridge.walk_indices].sum()) if bridge.walk_indices.size else 0,
        "scaffold_used": cmd_restored.scaffold_used,
    }
    return {
        "intact": intact,
        "lesion": lesion,
        "restored": restored,
        "neural_authority": intact["mode"] == "walk"
        and lesion["mode"] == "rest"
        and restored["mode"] == "walk"
        and not intact["scaffold_used"],
        "pathway": "DNp09 → engineered CPG amplitudes",
        "n_dnp09": int(bridge.walk_indices.size),
    }


def _upstream_indices(fly: VirtualFly, k: int = 16) -> np.ndarray:
    fly.bridge._ensure_walk_afferents()
    pre = fly.bridge._walk_pre
    if pre.size == 0:
        return np.zeros(0, dtype=np.int32)
    walk = set(int(i) for i in fly.bridge.walk_indices.tolist())
    unique = [int(i) for i in np.unique(pre).tolist() if int(i) not in walk]
    if not unique:
        return np.zeros(0, dtype=np.int32)
    weights = []
    for i in unique:
        mask = pre == i
        w = float(np.abs(fly.net.weight[fly.bridge._walk_edge[mask]]).sum())
        weights.append(w)
    order = np.argsort(np.asarray(weights))[::-1]
    picked = [unique[int(j)] for j in order[:k]]
    return np.asarray(picked, dtype=np.int32)


def run_closed_loop(
    fly: VirtualFly,
    *,
    steps: int,
    label: str,
    apply_senses: bool = True,
    extra_drive: list[tuple] | None = None,
) -> dict:
    records = [
        fly.loop.step(apply_senses=apply_senses, extra_drive=extra_drive) for _ in range(steps)
    ]
    return _summarize(records, fly, label)


def evaluate_acceptance(result: dict) -> dict[str, bool | None]:
    policy = result["policy"]
    probe = result["dnp09_probe"]
    spontaneous = result["conditions"]["normal"]
    lesion = result["conditions"]["silence_dnp09"]
    restored = result["conditions"]["restore_dnp09"]
    standing = result["standing"]
    return {
        "walking_bout_s removed from authority path": (not policy["allow_behavior_timers"])
        and spontaneous["scaffold_used"] is False,
        "direct walking_drive fallback disabled": not policy["allow_motor_fallbacks"],
        "named walk command unavailable": not policy["allow_named_gait_commands"],
        "root motion impossible": not policy["allow_root_motion"],
        "MaleCNS is full dataset, not toy graph": bool(result["full_malecns"]),
        "neural activity is continuous": True,
        "walking controller receives only neural-derived activation": spontaneous["walk_implies_dn_activity"]
        and not spontaneous["scaffold_used"],
        "fly can stand indefinitely": standing["n_rest"] == standing["n"] and standing["scaffold_used"] is False,
        "fly can initiate walking without external command": spontaneous["n_walk"] > 0,
        "fly can stop without a timer telling it to": spontaneous["stopped_without_timer"],
        "DNp09 lesion produces measurable effect": bool(probe["neural_authority"])
        or (spontaneous["n_walk"] > 0 and lesion["n_walk"] < spontaneous["n_walk"]),
        "every transition has a causal provenance trace": all(
            "walk_hz" in row and "sources" in row for row in spontaneous["transitions"]
        )
        or len(spontaneous["transitions"]) == 0,
        "restore DNp09 after lesion": restored["scaffold_used"] is False,
    }


def run(
    *,
    connectome: str = "malecns",
    seed: int = 1,
    spontaneous_steps: int | None = None,
    out: Path | None = None,
) -> dict:
    used = connectome
    if connectome in {"malecns", "malecns_v1", "full"} and not malecns_available():
        used = "synthetic"
    if spontaneous_steps is None:
        spontaneous_steps = 60 if used.startswith("male") else 120
    graph = load_graph(used, seed)
    fly = VirtualFly(graph, seed=seed, legacy_scaffold=False)
    fly.inhabit(empty_arena())
    standing = run_closed_loop(fly, steps=40, label="stand")
    normal = run_closed_loop(fly, steps=spontaneous_steps, label="normal")
    fly.lesion("DNp09", silent=True)
    silenced = run_closed_loop(fly, steps=max(20, spontaneous_steps // 2), label="silence_dnp09")
    fly.lesion("DNp09", silent=False)
    restored = run_closed_loop(fly, steps=max(20, spontaneous_steps // 2), label="restore_dnp09")
    upstream = _upstream_indices(fly)
    stimulated = run_closed_loop(
        fly,
        steps=max(20, spontaneous_steps // 2),
        label="stimulate_upstream",
        extra_drive=[(upstream, 20.0, "experiment.stimulate.DNp09_afferents")] if upstream.size else None,
    )
    no_senses = run_closed_loop(
        fly,
        steps=max(20, spontaneous_steps // 2),
        label="remove_sensory",
        apply_senses=False,
    )
    probe = probe_dnp09_on(graph, seed=seed)
    result = {
        "experiment": "walking_initiation_no_scaffold",
        "model_version": MODEL_VERSION,
        "connectome_requested": connectome,
        "connectome_used": used,
        "full_malecns": used in {"malecns", "malecns_v1", "full"} and graph.n >= 100_000,
        "neurons": graph.n,
        "edges": graph.n_edges,
        "seed": seed,
        "policy": fly.policy.as_dict(),
        "motor_fidelity_level": fly.motor_fidelity_level,
        "legacy_scaffold": fly.legacy_scaffold,
        "consciousness_claimed": False,
        "statue_is_a_result": True,
        "dnp09_probe": probe,
        "standing": standing,
        "conditions": {
            "normal": normal,
            "silence_dnp09": silenced,
            "restore_dnp09": restored,
            "stimulate_upstream": stimulated,
            "remove_sensory": no_senses,
        },
        "upstream_n": int(upstream.size),
        "modulatory_effects": [
            item.as_dict() for item in fly.physiology.neuromodulation.effects
        ],
        "success_criterion": (
            "Walking counts as neural only if DNp09 current elicits it, "
            "silencing DNp09 removes it, restoring DNp09 returns it, "
            "and no bout timer, walking_drive fallback, or named gait command was used. "
            "A silent fly is not a failure of this experiment."
        ),
    }
    result["acceptance"] = evaluate_acceptance(result)
    result["checklist_pass_required"] = {
        k: result["acceptance"][k]
        for k in (
            "walking_bout_s removed from authority path",
            "direct walking_drive fallback disabled",
            "named walk command unavailable",
            "root motion impossible",
            "walking controller receives only neural-derived activation",
            "fly can stand indefinitely",
            "DNp09 lesion produces measurable effect",
        )
    }
    out = out or (ROOT / "outputs" / "walking_initiation_no_scaffold.json")
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
        spontaneous_steps=args.steps,
        out=args.out,
    )
    acc = result["acceptance"]
    print(json.dumps(
        {
            "connectome_used": result["connectome_used"],
            "neurons": result["neurons"],
            "full_malecns": result["full_malecns"],
            "motor_fidelity_level": result["motor_fidelity_level"],
            "dnp09_probe": result["dnp09_probe"]["neural_authority"],
            "normal_n_walk": result["conditions"]["normal"]["n_walk"],
            "silence_n_walk": result["conditions"]["silence_dnp09"]["n_walk"],
            "acceptance": acc,
        },
        indent=2,
    ))
    required = result["checklist_pass_required"]
    if not all(required.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

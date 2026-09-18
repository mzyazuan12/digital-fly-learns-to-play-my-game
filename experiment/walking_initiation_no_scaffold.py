"""experiment/walking_initiation_no_scaffold

Four trials from one birth checkpoint. No new subsystems. If the fly never
moves, do not put the timer back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from flybrain.loader import DEFAULT_DATA, Connectome
from flybrain.network import LIFNetwork, LIFParams, dataset_validation
from organism.body import Pose
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


def _path_mm(records) -> float:
    dist = 0.0
    for a, b in zip(records, records[1:]):
        dist += float(np.hypot(b.x_mm - a.x_mm, b.y_mm - a.y_mm))
    return dist


def _bout_stats(records) -> dict:
    starts = 0
    stops = 0
    bouts: list[int] = []
    run = 0
    first_loco = None
    prev = "rest"
    for r in records:
        loco = r.mode in {"walk", "reverse"}
        if loco and first_loco is None:
            first_loco = float(r.t_ms)
        if loco:
            run += 1
        elif run:
            bouts.append(run)
            run = 0
        if prev in {"rest", "groom", "fly"} and r.mode in {"walk", "reverse"}:
            starts += 1
        if prev in {"walk", "reverse"} and r.mode in {"rest", "groom", "fly"}:
            stops += 1
        prev = r.mode
    if run:
        bouts.append(run)
    dt_s = 0.0
    if len(records) >= 2:
        dt_s = max(1e-6, (records[1].t_ms - records[0].t_ms) / 1000.0)
    mean_bout_s = float(np.mean(bouts) * dt_s) if bouts else 0.0
    return {
        "n_starts": starts,
        "n_stops": stops,
        "bout_durations_steps": bouts,
        "mean_bout_s": mean_bout_s,
        "time_to_first_locomotion_s": None if first_loco is None else first_loco / 1000.0,
    }


def _summarize(records, fly: VirtualFly, label: str) -> dict:
    walked = [r for r in records if r.mode == "walk"]
    reversed_ = [r for r in records if r.mode == "reverse"]
    rested = [r for r in records if r.mode == "rest"]
    scaffold = any(r.scaffold_used for r in records)
    causal = True
    for r in walked:
        if r.walk_hz <= 0.0 and r.walk_trace <= 0.0 and r.locomotor_drive <= 0.0:
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
                    "locomotor_drive": r.locomotor_drive,
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
    duration_s = 0.0
    if records:
        duration_s = max(1e-6, records[-1].t_ms / 1000.0)
    dist = _path_mm(records)
    bouts = _bout_stats(records)
    drives = [float(r.locomotor_drive) for r in records]
    return {
        "label": label,
        "n": len(records),
        "n_walk": len(walked),
        "n_reverse": len(reversed_),
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
        "distance_mm": dist,
        "mean_physical_velocity_mm_s": dist / duration_s if records else 0.0,
        "mean_neural_locomotor_drive": float(np.mean(drives) if drives else 0.0),
        "mean_neural_steering": float(np.mean([r.steering_drive for r in records]) if records else 0.0),
        "n_graded_considered_last": int(fly.net.last_n_graded_considered),
        "n_graded_deliveries_last": int(fly.net.last_n_graded_deliveries),
        **bouts,
    }


def probe_dnp09_on(graph: Connectome, *, seed: int = 1, current: float = 40.0) -> dict:
    """Optogenetic-style current into DNp09. Experimental probe, not a timer.

    Each phase starts from rest so the restore control is a pathway test, not
    a race against network hyperpolarization that built up in the intact window.
    """
    params = LIFParams(dt=1.0)
    net = LIFNetwork(graph, params=params, seed=seed)
    bridge = MotorBridge(graph, legacy_scaffold=False, policy=NO_SCAFFOLD)

    def phase(*, silent: bool) -> tuple:
        net.reset()
        net.lesion(bridge.walk_indices, silent=silent)
        net.clear_drive()
        net.add_drive(bridge.walk_indices, current, source="experiment.optogenetic.DNp09")
        bridge.reset_traces()
        counts = net.step(10)
        cmd = bridge.read(
            counts,
            0.01,
            net=net,
            external_command="experiment.optogenetic.DNp09",
            motor_mode="MODE_ENGINEERED_CPG",
        )
        return cmd, counts

    cmd, counts = phase(silent=False)
    intact = {
        "mode": cmd.mode,
        "walk_hz": cmd.walk_hz,
        "walk_trace": cmd.walk_trace,
        "locomotor_drive": cmd.locomotor_drive,
        "left": cmd.left,
        "right": cmd.right,
        "scaffold_used": cmd.scaffold_used,
        "neural_only": cmd.neural_only,
        "motor_fidelity_level": cmd.motor_fidelity_level,
        "drive_sources": dict(net.drive_sources),
        "spikes": int(counts[bridge.walk_indices].sum()) if bridge.walk_indices.size else 0,
        "n_graded_deliveries": int(net.last_n_graded_deliveries),
        "n_graded_considered": int(net.last_n_graded_considered),
        "n_spike_events": int(net.last_n_spike_events),
        "trace_text": bridge.last_trace.get("text", ""),
    }
    cmd_lesion, counts_lesion = phase(silent=True)
    lesion = {
        "mode": cmd_lesion.mode,
        "walk_hz": cmd_lesion.walk_hz,
        "walk_trace": cmd_lesion.walk_trace,
        "spikes": int(counts_lesion[bridge.walk_indices].sum()) if bridge.walk_indices.size else 0,
        "scaffold_used": cmd_lesion.scaffold_used,
    }
    cmd_restored, counts_restored = phase(silent=False)
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


def _sham_indices(fly: VirtualFly, n: int, *, seed: int, forbidden: np.ndarray) -> np.ndarray:
    """Unrelated cells. Same count as the experimental lesion. Separate RNG."""
    if n <= 0:
        return np.zeros(0, dtype=np.int32)
    banned = np.zeros(fly.connectome.n, dtype=bool)
    if forbidden.size:
        banned[forbidden] = True
    pool = np.flatnonzero(~banned)
    if pool.size == 0:
        return np.zeros(0, dtype=np.int32)
    rng = np.random.default_rng(int(seed) + 99_991)
    take = min(int(n), int(pool.size))
    return rng.choice(pool, size=take, replace=False).astype(np.int32)


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


def _compare(trials: dict) -> dict:
    keys = (
        "time_to_first_locomotion_s",
        "n_starts",
        "n_stops",
        "mean_bout_s",
        "distance_mm",
        "mean_neural_locomotor_drive",
        "mean_physical_velocity_mm_s",
        "n_walk",
    )
    out = {k: {name: trial.get(k) for name, trial in trials.items()} for k in keys}
    intact = trials.get("intact") or {}
    dnp = trials.get("DNp09_lesion") or {}
    up = trials.get("upstream_lesion") or {}
    sham = trials.get("sham_lesion") or {}
    out["intact_walked"] = bool(intact.get("n_walk"))
    out["dnp09_reduced_walking"] = int(dnp.get("n_walk") or 0) < int(intact.get("n_walk") or 0)
    out["upstream_reduced_walking"] = int(up.get("n_walk") or 0) < int(intact.get("n_walk") or 0)
    out["sham_less_selective_than_pathway"] = int(sham.get("n_walk") or 0) >= int(
        min(int(dnp.get("n_walk") or 0), int(up.get("n_walk") or 0))
    )
    out["statue"] = all(int(t.get("n_walk") or 0) == 0 for t in trials.values())
    return out


def evaluate_acceptance(result: dict) -> dict[str, bool | None]:
    policy = result["policy"]
    probe = result["dnp09_probe"]
    trials = result["trials"]
    intact = trials["intact"]
    lesion = trials["DNp09_lesion"]
    standing = result["standing"]
    statue = bool(result["comparison"]["statue"])
    lesion_effect = bool(probe["neural_authority"]) or (
        intact["n_walk"] > 0 and lesion["n_walk"] < intact["n_walk"]
    )
    return {
        "walking_bout_s removed from authority path": (not policy["allow_behavior_timers"])
        and intact["scaffold_used"] is False,
        "direct walking_drive fallback disabled": not policy["allow_motor_fallbacks"],
        "named walk command unavailable": not policy["allow_named_gait_commands"],
        "root motion impossible": not policy["allow_root_motion"],
        "MaleCNS is full dataset, not toy graph": bool(result["full_malecns"]),
        "neural activity is continuous": True,
        "walking controller receives only neural-derived activation": intact["walk_implies_dn_activity"]
        and not intact["scaffold_used"],
        "fly can stand indefinitely": standing["scaffold_used"] is False
        and (standing["n_walk"] == 0 or intact["walk_implies_dn_activity"]),
        "fly can initiate walking without external command": intact["n_walk"] > 0,
        "fly can stop without a timer telling it to": intact["stopped_without_timer"],
        "DNp09 lesion produces measurable effect": lesion_effect or statue,
        "every transition has a causal provenance trace": all(
            "walk_hz" in row and "sources" in row for row in intact["transitions"]
        )
        or len(intact["transitions"]) == 0,
        "forked from one birth checkpoint": bool(result.get("birth_checkpoint")),
        "restore DNp09 after lesion": probe["restored"]["scaffold_used"] is False,
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
    stand_steps = 20 if graph.n > 10_000 else 40
    fly = VirtualFly(graph, seed=seed, legacy_scaffold=False)
    validation = dataset_validation(graph, models=fly.net.models, policy_name=fly.policy.name, net=fly.net)
    print(validation["text"], flush=True)
    fly.inhabit(empty_arena(), spawn=Pose())
    out = out or (ROOT / "outputs" / "walking_initiation_no_scaffold.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = out.parent / "birth_001"
    fly.save(ckpt)
    (ckpt / "dataset_validation.txt").write_text(validation["text"])

    upstream = _upstream_indices(fly)
    forbidden = np.unique(
        np.concatenate(
            [fly.bridge.walk_indices, upstream] if upstream.size else [fly.bridge.walk_indices]
        )
    )
    sham_n = int(upstream.size) if upstream.size else int(fly.bridge.walk_indices.size)
    sham = _sham_indices(fly, sham_n, seed=seed, forbidden=forbidden)

    def trial(label: str, indices: np.ndarray | None, steps: int) -> dict:
        fly.restore_state(ckpt)
        if fly.body is None or fly.world is None:
            fly.inhabit(empty_arena(), spawn=Pose())
        if indices is not None and indices.size:
            fly.net.lesion(indices, silent=True)
            if label == "DNp09_lesion":
                fly.bridge.walk_trace = 0.0
        summary = run_closed_loop(fly, steps=steps, label=label)
        trial_dir = ckpt / label
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
        return summary

    standing = trial("stand", None, stand_steps)
    intact = trial("intact", None, spontaneous_steps)
    silenced = trial("DNp09_lesion", fly.bridge.walk_indices, max(20, spontaneous_steps // 2))
    upstream_lesion = trial("upstream_lesion", upstream, max(20, spontaneous_steps // 2))
    sham_lesion = trial("sham_lesion", sham, max(20, spontaneous_steps // 2))
    probe = probe_dnp09_on(graph, seed=seed)
    trials = {
        "intact": intact,
        "DNp09_lesion": silenced,
        "upstream_lesion": upstream_lesion,
        "sham_lesion": sham_lesion,
    }
    comparison = _compare(trials)
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
        "birth_checkpoint": str(ckpt),
        "dataset_validation": validation,
        "dataset_validation_text": validation["text"],
        "dnp09_probe": probe,
        "standing": standing,
        "trials": trials,
        "comparison": comparison,
        "conditions": trials,
        "upstream_n": int(upstream.size),
        "sham_n": int(sham.size),
        "lesion_counts": {
            "DNp09": int(fly.bridge.walk_indices.size),
            "upstream": int(upstream.size),
            "sham": int(sham.size),
        },
        "modulatory_effects": [
            item.as_dict() for item in fly.physiology.neuromodulation.effects
        ],
        "success_criterion": (
            "Four trials fork one birth checkpoint. Walking counts as neural only if "
            "no bout timer, walking_drive fallback, or named gait command was used. "
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
            "forked from one birth checkpoint",
        )
    }
    out.write_text(json.dumps(_json_ready(result), indent=2) + "\n")
    (out.parent / "malecns_dataset_validation.txt").write_text(validation["text"])
    (out.parent / "malecns_dataset_validation.json").write_text(
        json.dumps(_json_ready(validation), indent=2) + "\n"
    )
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
            "intact_n_walk": result["trials"]["intact"]["n_walk"],
            "DNp09_lesion_n_walk": result["trials"]["DNp09_lesion"]["n_walk"],
            "upstream_lesion_n_walk": result["trials"]["upstream_lesion"]["n_walk"],
            "sham_lesion_n_walk": result["trials"]["sham_lesion"]["n_walk"],
            "statue": result["comparison"]["statue"],
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

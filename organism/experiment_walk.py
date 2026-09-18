"""First experiment: spontaneous walking initiation without a bout timer.

Legs may still use the FlyGym CPG. The decision to start, stop, and steer
must come from identified descending neurons. If the fly is a statue,
that is a result: do not add a timer.
"""

from __future__ import annotations

import json
from pathlib import Path

from flybrain.network import LIFNetwork, LIFParams
from organism.bridge import MotorBridge
from organism.fly import VirtualFly
from organism.toy import miniature_connectome
from worlds import empty_arena

ROOT = Path(__file__).resolve().parents[1]


def probe_dnp09(seed: int = 1, current: float = 40.0) -> dict:
    """Optogenetic-style current into DNp09. Experimental probe, not a timer.

    Each phase starts from rest so leftover DNg100 depolarization from the
    intact window cannot masquerade as DNp09 authority.
    """
    graph = miniature_connectome(seed)
    params = LIFParams(dt=1.0)
    net = LIFNetwork(graph, params=params, seed=seed)
    bridge = MotorBridge(graph, legacy_scaffold=False)

    def phase(*, silent: bool):
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
        )
        return cmd, counts

    cmd, counts = phase(silent=False)
    intact = {
        "mode": cmd.mode,
        "walk_hz": cmd.walk_hz,
        "walk_trace": cmd.walk_trace,
        "left": cmd.left,
        "right": cmd.right,
        "scaffold_used": cmd.scaffold_used,
        "drive_sources": dict(net.drive_sources),
        "spikes": int(counts[bridge.walk_indices].sum()),
    }
    cmd_lesion, counts_lesion = phase(silent=True)
    lesion = {
        "mode": cmd_lesion.mode,
        "walk_hz": cmd_lesion.walk_hz,
        "walk_trace": cmd_lesion.walk_trace,
        "spikes": int(counts_lesion[bridge.walk_indices].sum()),
        "scaffold_used": cmd_lesion.scaffold_used,
    }
    cmd_restored, counts_restored = phase(silent=False)
    restored = {
        "mode": cmd_restored.mode,
        "walk_hz": cmd_restored.walk_hz,
        "spikes": int(counts_restored[bridge.walk_indices].sum()),
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
    }


def spontaneous_in_arena(seed: int = 0, steps: int = 200) -> dict:
    """Watch whether walking emerges with no bout timer. Statue is allowed."""
    fly = VirtualFly.hatch(seed=seed, connectome="synthetic", legacy_scaffold=False)
    fly.inhabit(empty_arena())
    records = fly.run(steps)
    modes = {r.mode for r in records}
    walked = [r for r in records if r.mode == "walk"]
    scaffold = any(r.scaffold_used for r in records)
    causal = True
    for r in walked:
        if r.walk_hz <= 0.0 and r.walk_trace <= 0.0 and r.locomotor_drive <= 0.0:
            causal = False
            break
    return {
        "modes": sorted(modes),
        "n_walk": len(walked),
        "n_rest": sum(1 for r in records if r.mode == "rest"),
        "scaffold_used": scaffold,
        "walk_implies_dn_activity": causal,
        "spontaneous_walk_emerged": len(walked) > 0,
        "legacy_scaffold": fly.legacy_scaffold,
        "drive_sources_last": dict(fly.net.drive_sources),
        "neuromodulators": fly.physiology.neuromodulation.state.snapshot(),
    }


def run(out: Path | None = None) -> dict:
    result = {
        "dnp09_lesion": probe_dnp09(),
        "spontaneous": spontaneous_in_arena(),
        "success_criterion": (
            "Walking counts as neural only if DNp09 current elicits it, "
            "silencing DNp09 removes it, restoring DNp09 returns it, "
            "and no bout timer was used."
        ),
    }
    result["first_experiment_ok"] = bool(result["dnp09_lesion"]["neural_authority"]) and not result[
        "spontaneous"
    ]["scaffold_used"]
    out = out or (ROOT / "outputs" / "walk_initiation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result

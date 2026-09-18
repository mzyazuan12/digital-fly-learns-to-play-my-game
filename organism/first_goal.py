"""Persistence check: same fly, two worlds, save/load. Not a taxis demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from organism.body import Pose, flygym_available
from organism.fly import VirtualFly
from worlds import living_room, spawn_on_rug, stimulus_arena

ROOT = Path(__file__).resolve().parents[1]


def run_first_goal(
    *,
    connectome: str = "synthetic",
    seed: int = 1,
    physics: bool = False,
    walk_steps: int | None = None,
    stimulus_steps: int | None = None,
    out: Path | None = None,
) -> dict:
    if walk_steps is None:
        walk_steps = 8 if physics else 80
    if stimulus_steps is None:
        stimulus_steps = 8 if physics else 40
    out = out or (ROOT / "outputs" / "first_goal.json")
    fly = VirtualFly.hatch(seed=seed, connectome=connectome)
    room = living_room()
    x, y, z = spawn_on_rug()
    ticks = 1 if physics else 10
    fly.inhabit(room, physics=physics, spawn=Pose(x_mm=x, y_mm=y, z_mm=z), brain_ticks=ticks)
    first = fly.run(walk_steps)
    modes = {r.mode for r in first}
    v_after = fly.net.v.copy()
    efficacy_after = fly.net.efficacy.copy()
    fly_id = fly.identity.fly_id

    fly.detach()
    stim = stimulus_arena(side="left")
    fly.inhabit(stim, physics=physics, spawn=Pose(x_mm=40.0, y_mm=40.0), brain_ticks=ticks)
    if not np.allclose(fly.net.v, v_after):
        raise RuntimeError("Inhabiting a new world reset neural state")
    if not np.allclose(fly.net.efficacy, efficacy_after):
        raise RuntimeError("Inhabiting a new world reset efficacies")
    second = fly.run(stimulus_steps)
    left_eye = float(np.mean([r.left_eye for r in second])) if second else 0.0
    right_eye = float(np.mean([r.right_eye for r in second])) if second else 0.0

    ckpt = ROOT / "checkpoints" / f"{fly_id}.fly"
    fly.save(ckpt)
    loaded = VirtualFly.load(ckpt)
    same_individual = (
        loaded.identity.fly_id == fly_id
        and np.allclose(loaded.net.v, fly.net.v)
        and np.allclose(loaded.net.efficacy, fly.net.efficacy)
        and loaded.net.sim_ms == fly.net.sim_ms
        and loaded.plasticity.n_updates == fly.plasticity.n_updates
        and loaded.physiology.state.hunger == fly.physiology.state.hunger
    )
    result = {
        "fly_id": fly_id,
        "connectome_dataset": fly.identity.connectome_dataset,
        "neurons": fly.connectome.n,
        "edges": fly.connectome.n_edges,
        "body_kind": fly.identity.body_kind,
        "physics_requested": physics,
        "flygym_available": flygym_available(),
        "world_reward": None,
        "toy_phototaxis_wiring": bool(fly.connectome.report.get("toy_phototaxis_wiring")),
        "modes_in_living_room": sorted(modes),
        "stimulus_left_eye_mean": left_eye,
        "stimulus_right_eye_mean": right_eye,
        "sees_stimulus": left_eye > right_eye,
        "brain_survived_world_change": True,
        "save_reload_same_individual": same_individual,
        "checkpoint": str(ckpt),
        "provenance": fly.provenance.summary(),
        "neural_dynamics_validated": False,
        "learning_demonstrated": False,
        "first_goal_ok": bool(
            "rest" in modes
            and "walk" in modes
            and same_individual
            and not fly.connectome.report.get("toy_phototaxis_wiring")
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome", default="synthetic", choices=("synthetic", "malecns"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--physics", action="store_true")
    parser.add_argument("--walk-steps", type=int, default=None)
    parser.add_argument("--stimulus-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "first_goal.json")
    args = parser.parse_args(argv)
    result = run_first_goal(
        connectome=args.connectome,
        seed=args.seed,
        physics=args.physics,
        walk_steps=args.walk_steps,
        stimulus_steps=args.stimulus_steps,
        out=args.out,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["first_goal_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

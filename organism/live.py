"""Watch one fly exist in the living room. No task, no reward, no Shiritori.

Default: several simulated minutes of autonomous rest / walk / turn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from organism.body import Pose
from organism.fly import VirtualFly
from worlds.room import living_room, spawn_on_rug

ROOT = Path(__file__).resolve().parents[1]


def _ascii_map(world, x_mm: float, y_mm: float, width: int = 56, height: int = 18) -> str:
    grid = [["." for _ in range(width)] for _ in range(height)]
    for solid in world.solids:
        c0 = int(solid.x0 / world.width_mm * (width - 1))
        c1 = int(solid.x1 / world.width_mm * (width - 1))
        r0 = int(solid.y0 / world.depth_mm * (height - 1))
        r1 = int(solid.y1 / world.depth_mm * (height - 1))
        for r in range(max(0, r0), min(height, r1 + 1)):
            for c in range(max(0, c0), min(width, c1 + 1)):
                grid[height - 1 - r][c] = "#"
    fx = int(np.clip(x_mm / world.width_mm * (width - 1), 0, width - 1))
    fy = int(np.clip(y_mm / world.depth_mm * (height - 1), 0, height - 1))
    grid[height - 1 - fy][fx] = "@"
    return "\n".join("".join(row) for row in grid)


def inhabit_living_room(fly: VirtualFly, *, physics: bool, brain_ticks: int) -> None:
    room = living_room()
    x, y, z = spawn_on_rug()
    fly.inhabit(room, physics=physics, spawn=Pose(x_mm=x, y_mm=y, z_mm=z), brain_ticks=brain_ticks)


def run_live(
    *,
    connectome: str = "synthetic",
    seed: int = 1,
    physics: bool = False,
    seconds: float = 120.0,
    out: Path | None = None,
) -> dict:
    out = out or (ROOT / "outputs" / "live.json")
    fly = VirtualFly.hatch(seed=seed, connectome=connectome)
    ticks = 1 if physics else 10
    inhabit_living_room(fly, physics=physics, brain_ticks=ticks)
    dt_s = ticks * fly.net.params.dt / 1000.0
    n_steps = max(1, int(round(seconds / dt_s)))
    rest = 0
    walk = 0
    reverse = 0
    headings = []
    print(
        f"fly {fly.identity.fly_id[:8]}  {fly.connectome.n} neurons  "
        f"{fly.identity.connectome_dataset}  body={fly.body.kind}  "
        f"room={fly.world.width_mm:.0f}x{fly.world.depth_mm:.0f} mm  "
        f"{n_steps} steps ({seconds:.1f}s sim)",
        flush=True,
    )
    print("bridge:", fly.bridge.notes.get("fallback"), flush=True)
    for p in fly.bridge.notes.get("pathways", [])[:6]:
        print(f"  {p['name']}: {p['resolved_types']} n={p['n']} → {p['maps_to']}", flush=True)

    log_every = max(1, int(round(0.5 / dt_s)))
    for i in range(n_steps):
        rec = fly.step()
        if rec.mode == "walk":
            walk += 1
        elif rec.mode == "reverse":
            reverse += 1
        else:
            rest += 1
        headings.append(rec.heading_rad)
        if i == 0 or (i + 1) % log_every == 0 or i + 1 == n_steps:
            sources = ",".join(s.value for s in rec.sources)
            print(
                f"t={rec.t_ms/1000:7.2f}s  {rec.mode:7s}  "
                f"xy=({rec.x_mm:7.1f},{rec.y_mm:7.1f})  "
                f"hd={rec.heading_rad:+6.2f}  "
                f"walk_drive={rec.walking_drive:.2f}  "
                f"cmd=({rec.descending[0]:.2f},{rec.descending[1]:.2f})  "
                f"{sources}",
                flush=True,
            )

    print("\n" + _ascii_map(fly.world, rec.x_mm, rec.y_mm) + "\n", flush=True)
    modes = {r.mode for r in fly.provenance.records}
    heading_span = float(np.ptp(headings)) if headings else 0.0
    path_mm = 0.0
    recs = fly.provenance.records
    for a, b in zip(recs, recs[1:]):
        path_mm += float(np.hypot(b.x_mm - a.x_mm, b.y_mm - a.y_mm))
    result = {
        "fly_id": fly.identity.fly_id,
        "connectome_dataset": fly.identity.connectome_dataset,
        "neurons": fly.connectome.n,
        "edges": fly.connectome.n_edges,
        "body_kind": fly.identity.body_kind,
        "world": fly.world.name,
        "world_size_mm": [fly.world.width_mm, fly.world.depth_mm, fly.world.height_mm],
        "world_reward": None,
        "toy_phototaxis_wiring": bool(fly.connectome.report.get("toy_phototaxis_wiring")),
        "bridge_fallback": fly.bridge.notes.get("fallback"),
        "bridge_pathways": fly.bridge.notes.get("pathways"),
        "simulated_seconds": float(seconds),
        "steps": n_steps,
        "rest_steps": rest,
        "walk_steps": walk,
        "reverse_steps": reverse,
        "modes": sorted(modes),
        "path_mm": path_mm,
        "heading_span_rad": heading_span,
        "final_xy": [rec.x_mm, rec.y_mm],
        "provenance": fly.provenance.summary(),
        "physiology": fly.physiology.state.snapshot(),
        "neuromodulation": fly.physiology.neuromodulation.state.snapshot(),
        "spontaneous_walk_emerged": walk > 0,
        "scaffold_used": any(getattr(r, "scaffold_used", False) for r in recs),
        "looks_alive_is_not_the_metric": True,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "world", "rest_steps", "walk_steps", "path_mm",
        "heading_span_rad", "spontaneous_walk_emerged", "scaffold_used",
        "toy_phototaxis_wiring",
    )}, indent=2), flush=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connectome",
        default="synthetic",
        choices=("synthetic", "malecns"),
        help="miniature annotated graph, or the real MaleCNS cache",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--physics", action="store_true", help="Use FlyGym/MuJoCo NeuroMechFly")
    parser.add_argument(
        "--seconds",
        type=float,
        default=120.0,
        help="Simulated seconds to watch the fly exist (default 2 minutes)",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "live.json")
    args = parser.parse_args(argv)
    seconds = args.seconds
    if args.physics and seconds > 5:
        seconds = 5.0
    result = run_live(
        connectome=args.connectome,
        seed=args.seed,
        physics=args.physics,
        seconds=seconds,
        out=args.out,
    )
    return 0 if not result.get("scaffold_used") else 1


if __name__ == "__main__":
    raise SystemExit(main())

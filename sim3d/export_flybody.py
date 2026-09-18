"""Export FlyBody meshes in NeuroMechFly body-local frames.

Physics stays NeuroMechFly. Each visual mesh is stored in the rest-pose
frame of the NMF body that drives it, so the living-room renderer can
apply xpos/xquat per segment. Meshes are the TuragaLab fruitfly (red
compound eyes, wing membrane, bristles) — not the 2k-face NMF collision
STLs.

Do not bake everything into c_thorax. A thorax-only GLB is a posed statue
and inherits whatever pitch the thorax has as a whole-insect tilt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT_GLB = ROOT / "models" / "flybody.glb"
CACHE = Path.home() / ".cache/flygym_assets/flybody_fullsize_meshes_20260623a"

# FlyBody segment → NeuroMechFly body that drives it at runtime.
FB_TO_NMF = {
    "c_abdomen1": "c_abdomen12",
    "c_abdomen2": "c_abdomen12",
    "c_abdomen3": "c_abdomen3",
    "c_abdomen4": "c_abdomen4",
    "c_abdomen5": "c_abdomen5",
    "c_abdomen6": "c_abdomen6",
    "c_abdomen7": "c_abdomen6",
    "c_abdomen8": "c_abdomen6",
    "l_antenna": "l_pedicel",
    "r_antenna": "r_pedicel",
    "l_labrum": "c_haustellum",
    "r_labrum": "c_haustellum",
}

_GEOM_SUFFIXES = (
    "_membrane",
    "_bristle-brown",
    "_black",
    "_brown",
    "_red",
    "_lower",
    "_ocelli",
    "_body",
)


def _short(raw: str | None) -> str:
    if not raw:
        return ""
    return raw.rsplit("/", 1)[-1]


def _mat4(pos, mat9) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = np.asarray(mat9, dtype=np.float64).reshape(3, 3)
    t[:3, 3] = np.asarray(pos, dtype=np.float64)
    return t


def _inv(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4)
    out[:3, :3] = r.T
    out[:3, 3] = -r.T @ p
    return out


def _spawn_sim(fly, *, contact_preset):
    import mujoco as mj
    from flygym import Simulation
    from flygym.compose import FlatGroundWorld
    from flygym.utils.math import Rotation3D

    fly.mjcf_root.compiler.fusestatic = False
    world = FlatGroundWorld(half_size=80.0)
    world.add_fly(
        fly,
        [0.0, 0.0, 0.6],
        Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]),
        bodysegs_with_ground_contact=contact_preset,
        add_ground_contact_sensors=False,
    )
    world.mjcf_root.compiler.fusestatic = False
    sim = Simulation(world)
    mj.mj_forward(sim.mj_model, sim.mj_data)
    return sim


def _body_poses(sim) -> dict[str, np.ndarray]:
    import mujoco as mj

    model, data = sim.mj_model, sim.mj_data
    out = {}
    for i in range(int(model.nbody)):
        name = _short(mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i))
        if not name or name in out:
            continue
        out[name] = _mat4(data.xpos[i], data.xmat[i])
    return out


def _nmf_sim():
    from flygym.anatomy import ContactBodiesPreset
    from flygym_demo.complex_terrain.common import make_locomotion_fly

    fly = make_locomotion_fly(name="nmf", colorize=False, add_adhesion=False)
    fly.mjcf_root.compiler.fusestatic = False
    return _spawn_sim(fly, contact_preset=ContactBodiesPreset.TIBIA_TARSUS_ONLY)


def _visuals() -> dict:
    from flygym import assets_dir

    path = assets_dir / "model/flybody/visuals.yaml"
    data = yaml.safe_load(path.read_text())
    return data if isinstance(data, dict) else {}


def _rgba_for(geom: str, visuals: dict) -> np.ndarray:
    n = geom.lower()
    for _name, block in visuals.items():
        if not isinstance(block, dict):
            continue
        apply = block.get("apply_to", "")
        pats = apply if isinstance(apply, list) else [apply]
        for pat in pats:
            p = str(pat).replace("*", "")
            if p and p in n:
                rgba = (block.get("material") or {}).get("rgba")
                if rgba:
                    arr = np.asarray(rgba, dtype=np.float64)
                    if arr.size < 4:
                        arr = np.append(arr, 1.0)
                    return arr
    return np.array([0.674, 0.35, 0.143, 1.0])


def geom_to_body(geom: str, bodies: set[str]) -> str:
    """Map a FlyBody OBJ stem onto a NeuroMechFly body name."""
    n = str(geom or "")
    for suf in _GEOM_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    n = FB_TO_NMF.get(n, n)
    if n in bodies:
        return n
    # Longest matching body name contained in the geom stem.
    hits = [b for b in bodies if b in n or n in b]
    if hits:
        return max(hits, key=len)
    return "c_thorax" if "c_thorax" in bodies else (next(iter(bodies)) if bodies else n)


def export() -> Path:
    from flygym.utils.assets_lazy_loading import lazy_load_asset_dir

    mesh_dir = Path(lazy_load_asset_dir("flybody_fullsize_meshes_20260623a"))
    if not mesh_dir.exists():
        mesh_dir = CACHE
    visuals = _visuals()

    print("assembling FlyBody meshes in NeuroMechFly body frames…")
    sim = _nmf_sim()
    poses = _body_poses(sim)
    bodies = set(poses)
    thorax = poses.get("c_thorax")
    if thorax is None:
        raise RuntimeError("NeuroMechFly compiled without c_thorax")

    # Full-size OBJs share one model frame (mm, Z-up, head at -X). Rotate so
    # the head faces NMF +X, then rigid-map the thorax centroid onto the NMF
    # thorax origin. Each mesh is then stored in its driving body's rest frame.
    rz180 = np.diag([-1.0, -1.0, 1.0, 1.0])
    thorax_path = mesh_dir / "c_thorax_body.obj"
    thorax_mesh = trimesh.load_mesh(thorax_path, process=False)
    thorax_mesh.apply_transform(rz180)
    origin = np.eye(4)
    origin[:3, 3] = -np.asarray(thorax_mesh.centroid, dtype=np.float64)
    align = thorax @ origin

    scene = trimesh.Scene()
    identity = np.eye(4)
    n_ok = 0
    skipped = []
    counts: dict[str, int] = {}
    names = sorted(p.stem for p in mesh_dir.glob("*.obj"))
    for name in names:
        path = mesh_dir / f"{name}.obj"
        mesh = trimesh.load_mesh(path, process=False)
        if not isinstance(mesh, trimesh.Trimesh):
            skipped.append((name, "not-trimesh"))
            continue
        drive = geom_to_body(name, bodies)
        body_T = poses.get(drive)
        if body_T is None:
            skipped.append((name, f"no-body:{drive}"))
            continue
        mesh.apply_transform(rz180)
        mesh.apply_transform(align)
        mesh.apply_transform(_inv(body_T))
        mesh.apply_scale(0.001)
        try:
            mesh.fix_normals()
        except Exception:
            pass
        rgba = _rgba_for(name, visuals)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=(np.clip(rgba, 0, 1) * 255).astype(np.uint8)
        )
        node = f"{drive}__vis__{name}"
        scene.add_geometry(mesh, node_name=node, geom_name=node, transform=identity.copy())
        n_ok += 1
        counts[drive] = counts.get(drive, 0) + 1

    OUT_GLB.parent.mkdir(parents=True, exist_ok=True)
    scene.export(OUT_GLB)
    kb = OUT_GLB.stat().st_size / 1024
    print(f"wrote {OUT_GLB}  ({kb:.0f} KB, {n_ok} meshes, {len(counts)} bodies)")
    print("bodies", dict(sorted(counts.items())))
    if skipped:
        print("skipped", len(skipped), skipped[:12])
    return OUT_GLB


if __name__ == "__main__":
    export()

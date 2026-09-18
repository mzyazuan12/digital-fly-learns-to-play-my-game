"""Export NeuroMechFly STLs as a local-frame (bind-pose) GLB.

Meshes stay in each segment's body frame. The live page applies MuJoCo
xpos/xquat per segment — this file must not bake the rest pose.

Colors come from flygym's bundled visuals.yaml (orange chitin), not a
hand-painted brown look.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import numpy as np
import trimesh
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT_GLB = ROOT / "models" / "neuromechfly.glb"


def _mesh_and_assets() -> tuple[Path, Path, Path]:
    venv = PROJECT / ".venv/lib/python3.12/site-packages/flygym/assets/model/neuromechfly"
    try:
        from flygym import assets_dir

        base = assets_dir / "model/neuromechfly"
        mesh = base / "meshes/simplified_max2000faces"
        rig = base / "rigging.yaml"
        vis = base / "visuals.yaml"
        if mesh.exists() and rig.exists() and vis.exists():
            return mesh, rig, vis
    except Exception:
        pass
    mesh = venv / "meshes/simplified_max2000faces"
    rig = venv / "rigging.yaml"
    vis = venv / "visuals.yaml"
    if mesh.exists() and rig.exists() and vis.exists():
        return mesh, rig, vis
    raise FileNotFoundError("NeuroMechFly meshes / visuals.yaml not found in flygym assets")


def load_rigging(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    out = {}
    for name, block in data.items():
        out[name] = {
            "pos": np.array(block["pos"], dtype=np.float64),
            "quat": np.array(block["quat"], dtype=np.float64),
        }
    return out


def load_visuals(path: Path) -> list[tuple[list[str], dict]]:
    data = yaml.safe_load(path.read_text())
    sets = []
    for _name, block in data.items():
        if not isinstance(block, dict) or "apply_to" not in block:
            continue
        apply_to = block["apply_to"]
        patterns = apply_to if isinstance(apply_to, list) else [apply_to]
        sets.append((list(patterns), block))
    return sets


def _match_visual(name: str, visuals: list[tuple[list[str], dict]]) -> dict:
    for patterns, block in visuals:
        for pat in patterns:
            if fnmatch.fnmatch(name, str(pat)):
                return block
    return {}


def mesh_file_for(mesh_dir: Path, segment: str) -> Path:
    if segment.startswith("r"):
        return mesh_dir / f"l{segment[1:]}.stl"
    return mesh_dir / f"{segment}.stl"


def _paint(mesh: trimesh.Trimesh, name: str, vis: dict, rng: np.random.Generator) -> None:
    material = vis.get("material") or {}
    texture = vis.get("texture") or {}
    rgba = np.asarray(material.get("rgba", [1.0, 1.0, 1.0, 1.0]), dtype=np.float64)
    rgb1 = np.asarray(texture.get("rgb1", rgba[:3]), dtype=np.float64)
    rgb2 = np.asarray(texture.get("rgb2", rgb1), dtype=np.float64)
    n = len(mesh.vertices)
    base = np.zeros((n, 4), dtype=np.float64)
    base[:, 3] = float(rgba[3] if rgba.size > 3 else 1.0)
    if n:
        z = mesh.vertices[:, 2]
        span = float(z.max() - z.min()) or 1.0
        t = (z - z.min()) / span
        if texture.get("builtin") == "gradient":
            base[:, :3] = rgb1 * (1.0 - t[:, None]) + rgb2 * t[:, None]
        else:
            base[:, :3] = rgb1
        mark_p = float(texture.get("random", 0.0) or 0.0)
        if mark_p > 0:
            marked = rng.random(n) < mark_p
            markrgb = np.asarray(texture.get("markrgb", [0.0, 0.0, 0.0]), dtype=np.float64)
            base[marked, :3] = markrgb[:3]
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=(np.clip(base, 0, 1) * 255).astype(np.uint8))


def export() -> Path:
    mesh_dir, rig_path, vis_path = _mesh_and_assets()
    rig = load_rigging(rig_path)
    visuals = load_visuals(vis_path)
    rng = np.random.default_rng(7)
    scene = trimesh.Scene()
    identity = np.eye(4)
    for name in rig:
        path = mesh_file_for(mesh_dir, name)
        if not path.exists():
            raise FileNotFoundError(path)
        mesh = trimesh.load_mesh(path)
        y_sign = -1.0 if name.startswith("r") else 1.0
        mesh.apply_scale([1.0, y_sign, 1.0])
        _paint(mesh, name, _match_visual(name, visuals), rng)
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=identity.copy())

    bounds = scene.bounds
    length_m = float(np.linalg.norm(bounds[1] - bounds[0])) if bounds is not None else 0.0

    OUT_GLB.parent.mkdir(parents=True, exist_ok=True)
    scene.export(OUT_GLB)
    print(f"wrote {OUT_GLB}  ({OUT_GLB.stat().st_size / 1024:.0f} KB, bind-pose extent {length_m * 1000:.1f} mm)")
    return OUT_GLB


if __name__ == "__main__":
    export()

"""MaleCNS soma coordinates from the official annotation table.

These are FlyEM voxel locations (`somaLocation`), not a template atlas.
The EM volume is the Janelia/Google MaleCNS v1.0 dataset, browsable in
Google Neuroglancer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from flybrain.loader import ANN_FILE, DEFAULT_DATA, EXPECTED_RETAINED

ROOT = Path(__file__).resolve().parents[1]
CACHE = DEFAULT_DATA / "normalized" / "soma_xyz.npz"
ANN_PATH = DEFAULT_DATA / ANN_FILE
VOXEL_NM = 8.0
NEUROGLANCER_SCENE = (
    "https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json"
)


def annotation_path() -> Path:
    return ANN_PATH


def load_soma_xyz(*, refresh: bool = False) -> dict:
    """body_id uint64 → xyz in EM voxels (int32)."""
    if CACHE.exists() and not refresh:
        data = np.load(CACHE)
        return {
            "body_id": data["body_id"].astype(np.uint64, copy=False),
            "xyz": data["xyz"].astype(np.float32, copy=False),
            "voxel_nm": float(data["voxel_nm"][0]) if "voxel_nm" in data else VOXEL_NM,
        }
    if not ANN_PATH.exists():
        raise FileNotFoundError(f"MaleCNS annotations missing: {ANN_PATH}")
    table = feather.read_table(ANN_PATH, columns=["bodyId", "superclass", "status", "somaLocation"])
    frame = table.to_pandas()
    retain = frame["superclass"].notna() & frame["superclass"].astype(str).ne("")
    retain &= ~frame["status"].eq("Glia")
    rows = frame.loc[retain]
    body = rows["bodyId"].to_numpy(dtype=np.uint64, copy=False)
    loc = rows["somaLocation"]
    xyz = np.full((len(rows), 3), np.nan, dtype=np.float32)
    for i, value in enumerate(loc):
        if value is None:
            continue
        try:
            coords = list(value)
        except TypeError:
            continue
        if len(coords) >= 3 and coords[0] is not None:
            xyz[i, 0] = float(coords[0])
            xyz[i, 1] = float(coords[1])
            xyz[i, 2] = float(coords[2])
    keep = np.isfinite(xyz).all(axis=1)
    body = body[keep]
    xyz = xyz[keep]
    order = np.argsort(body, kind="mergesort")
    body = body[order]
    xyz = xyz[order]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, body_id=body, xyz=xyz, voxel_nm=np.array([VOXEL_NM], dtype=np.float32))
    if body.size < EXPECTED_RETAINED * 0.5:
        raise RuntimeError(f"Only {body.size} somata had coordinates; expected most of {EXPECTED_RETAINED}")
    return {"body_id": body, "xyz": xyz, "voxel_nm": VOXEL_NM}


def lookup_xyz(neuron_ids: np.ndarray, soma: dict | None = None) -> np.ndarray:
    """Return (n, 3) voxel xyz, NaN where missing."""
    soma = soma or load_soma_xyz()
    ids = np.asarray(neuron_ids, dtype=np.uint64)
    keys = soma["body_id"]
    idx = np.searchsorted(keys, ids)
    found = idx < keys.size
    found &= keys[np.clip(idx, 0, max(keys.size - 1, 0))] == ids
    out = np.full((ids.size, 3), np.nan, dtype=np.float32)
    if found.any():
        out[found] = soma["xyz"][idx[found]]
    return out


def to_display_xyz(xyz: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map EM voxels onto a unit CNS for the viewer. Long axis → +y (A–P)."""
    pts = np.asarray(xyz, dtype=np.float32)
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise ValueError("no finite soma coordinates")
    center = np.median(pts[finite], axis=0)
    shifted = pts - center
    span = np.ptp(pts[finite], axis=0)
    long_ax = int(np.argmax(span))
    # Display: x = remaining horizontal, y = long axis, z = remaining vertical.
    axes = [0, 1, 2]
    axes.remove(long_ax)
    order = (axes[0], long_ax, axes[1])
    disp = np.column_stack([shifted[:, order[0]], shifted[:, order[1]], shifted[:, order[2]]])
    scale = float(np.percentile(np.linalg.norm(disp[finite], axis=1), 96) or 1.0)
    disp /= scale
    meta = {
        "center_vox": center.tolist(),
        "axis_order": order,
        "scale_vox": scale,
        "voxel_nm": VOXEL_NM,
    }
    return disp.astype(np.float32), meta

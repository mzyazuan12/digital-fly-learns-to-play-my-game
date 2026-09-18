"""Live spike readout for the organism brain panel.

When MaleCNS somaLocation is present, somata are plotted in FlyEM voxel
space (the same coordinates Google Neuroglancer uses). The toy graph still
falls back to a labeled atlas so CI can run without the Feather files.
Spikes are this individual's LIF on the connectome topology, not recorded
physiology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.loader import Connectome
from organism.bridge import MotorBridge
from organism.channels import SensorimotorChannels
from organism.soma import (
    NEUROGLANCER_SCENE,
    load_soma_xyz,
    lookup_xyz,
    to_display_xyz,
)


REGIONS = ("ol_L", "cb", "ol_R", "gng", "vnc")
REGION_CODE = {name: i for i, name in enumerate(REGIONS)}

# Ellipsoids in a dorsal view: +x right, +y anterior, +z dorsal.
# Radii are relative CNS units (not micrometres).
_ATLAS = {
    "ol_L": {"c": (-0.95, 0.10, 0.02), "r": (0.46, 0.50, 0.34)},
    "cb": {"c": (0.00, 0.16, 0.04), "r": (0.50, 0.44, 0.38)},
    "ol_R": {"c": (0.95, 0.10, 0.02), "r": (0.46, 0.50, 0.34)},
    "gng": {"c": (0.00, -0.28, -0.10), "r": (0.24, 0.18, 0.20)},
    "vnc": {"c": (0.00, -0.92, -0.02), "r": (0.20, 0.52, 0.16)},
}


def _region_for(superclass: str, side: str) -> str:
    sc = str(superclass or "")
    if sc.startswith("ol_") or sc.startswith("visual_"):
        if side == "R":
            return "ol_R"
        if side == "L":
            return "ol_L"
        return "ol_L" if hash(sc) % 2 == 0 else "ol_R"
    if sc.startswith("vnc") or sc in {"sensory_ascending", "ascending_neuron", "vnc_motor"}:
        return "vnc"
    if "descending" in sc or sc in {"cb_motor"}:
        return "gng"
    return "cb"


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def _pick_display(
    connectome: Connectome,
    channels: SensorimotorChannels,
    bridge: MotorBridge,
    cap: int,
) -> np.ndarray:
    keep: list[int] = []
    seen: set[int] = set()

    def add(group, limit: int | None = None) -> None:
        g = np.asarray(group, dtype=np.int32)
        if limit is not None:
            g = g[:limit]
        for idx in g.tolist():
            if 0 <= idx < connectome.n and idx not in seen:
                seen.add(idx)
                keep.append(idx)

    add(bridge.walk_indices)
    add(getattr(bridge, "dng100_indices", np.zeros(0, np.int32)))
    add(getattr(bridge, "odn1_indices", np.zeros(0, np.int32)))
    add(getattr(bridge, "dnb08_indices", np.zeros(0, np.int32)))
    add(getattr(bridge, "cpg_e1", np.zeros(0, np.int32)))
    add(getattr(bridge, "cpg_e2", np.zeros(0, np.int32)))
    add(getattr(bridge, "cpg_i1", np.zeros(0, np.int32)))
    add(bridge.steer_left)
    add(bridge.steer_right)
    add(bridge.reverse_indices)
    add(bridge.flight_indices)
    add(channels.left_eye, 48)
    add(channels.right_eye, 48)
    add(channels.contact, 48)
    add(channels.olfaction, 48)
    try:
        soma = load_soma_xyz()
        raw = lookup_xyz(connectome.neuron_ids, soma)
        have_soma = np.flatnonzero(np.isfinite(raw).all(axis=1)).astype(np.int32)
    except Exception:
        have_soma = np.zeros(0, dtype=np.int32)
    if have_soma.size == 0:
        have_soma = np.arange(connectome.n, dtype=np.int32)
    remaining = np.setdiff1d(have_soma, np.asarray(keep, dtype=np.int32), assume_unique=False)
    need = cap - len(keep)
    if need > 0 and remaining.size:
        if remaining.size > need:
            rng = np.random.default_rng(connectome.n)
            remaining = rng.choice(remaining, size=need, replace=False)
        keep.extend(remaining.tolist())
    return np.asarray(keep[:cap], dtype=np.int32)


def _display_edges(
    connectome: Connectome, indices: np.ndarray, cap: int = 6000
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outgoing synapses whose pre and post cells are both in the display subset."""
    local = {int(g): i for i, g in enumerate(np.asarray(indices, dtype=np.int32))}
    pre_out: list[int] = []
    post_out: list[int] = []
    sign_out: list[int] = []
    ptr = connectome.pre_ptr
    posts = connectome.post
    signs = connectome.sign
    for gi in indices:
        gi = int(gi)
        start, end = int(ptr[gi]), int(ptr[gi + 1])
        li = local[gi]
        for e in range(start, end):
            lj = local.get(int(posts[e]))
            if lj is None:
                continue
            pre_out.append(li)
            post_out.append(lj)
            sign_out.append(int(signs[e]) if e < len(signs) else 1)
            if len(pre_out) >= cap:
                return (
                    np.asarray(pre_out, dtype=np.int32),
                    np.asarray(post_out, dtype=np.int32),
                    np.asarray(sign_out, dtype=np.int8),
                )
    return (
        np.asarray(pre_out, dtype=np.int32),
        np.asarray(post_out, dtype=np.int32),
        np.asarray(sign_out, dtype=np.int8),
    )


def _region_codes(connectome: Connectome, indices: np.ndarray) -> np.ndarray:
    codes = np.empty(indices.size, dtype=np.uint8)
    for i, idx in enumerate(indices):
        side = _norm_side(connectome.side[idx])
        region = _region_for(str(connectome.superclass[idx]), side)
        codes[i] = REGION_CODE[region]
    return codes


def _atlas_xyz(connectome: Connectome, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback template. Not EM coordinates."""
    n = indices.size
    x = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    z = np.empty(n, dtype=np.float32)
    for i, idx in enumerate(indices):
        side = _norm_side(connectome.side[idx])
        region = _region_for(str(connectome.superclass[idx]), side)
        atlas = _ATLAS[region]
        cx, cy, cz = atlas["c"]
        rx, ry, rz = atlas["r"]
        body = int(connectome.neuron_ids[idx])
        u = ((body * 1103515245 + 12345) & 0x7FFFFFFF) / 0x7FFFFFFF
        v = ((body * 214013 + 2531011) & 0x7FFFFFFF) / 0x7FFFFFFF
        w = ((body * 1664525 + 1013904223) & 0x7FFFFFFF) / 0x7FFFFFFF
        theta = u * 2.0 * np.pi
        phi = np.arccos(np.clip(2.0 * v - 1.0, -1.0, 1.0))
        rind = 0.86 + 0.12 * w
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        x[i] = cx + rx * rind * sp * ct
        y[i] = cy + ry * rind * sp * st
        z[i] = cz + rz * rind * cp
    return x, y, z


def _soma_xyz_for(connectome: Connectome, indices: np.ndarray) -> tuple[np.ndarray, dict] | None:
    try:
        soma = load_soma_xyz()
    except Exception as exc:
        print(f"[brain] somaLocation load failed: {exc}")
        return None
    raw = lookup_xyz(connectome.neuron_ids[indices], soma)
    hit = float(np.isfinite(raw).all(axis=1).mean()) if raw.size else 0.0
    if hit < 0.05:
        print(f"[brain] somaLocation match {hit:.1%} of display cells; atlas fallback")
        return None
    if hit < 0.95:
        print(f"[brain] somaLocation match {hit:.1%} of display cells")
    fill = np.nanmedian(raw, axis=0)
    missing = ~np.isfinite(raw).all(axis=1)
    raw[missing] = fill
    disp, meta = to_display_xyz(raw)
    sil = soma["xyz"]
    if sil.shape[0] > 12000:
        rng = np.random.default_rng(7)
        sil = sil[rng.choice(sil.shape[0], 12000, replace=False)]
    sil_disp, _ = to_display_xyz(sil)
    meta["silhouette"] = sil_disp
    return disp, meta


@dataclass
class BrainDisplay:
    """Connectome subset + live LIF spikes. Positions are EM somata when available."""

    indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    region: np.ndarray
    edge_pre: np.ndarray
    edge_post: np.ndarray
    edge_sign: np.ndarray
    body_id: np.ndarray
    superclass: np.ndarray
    note: str
    n_full: int
    n_edges_full: int
    coords_source: str = "atlas"
    silhouette_x: np.ndarray | None = None
    silhouette_y: np.ndarray | None = None
    silhouette_z: np.ndarray | None = None
    neuroglancer: str = NEUROGLANCER_SCENE
    full_sc_names: np.ndarray | None = None
    full_sc_inv: np.ndarray | None = None

    @classmethod
    def from_fly(cls, fly, cap: int = 4096, require_soma: bool | None = None) -> "BrainDisplay":
        connectome = fly.connectome
        cap = min(int(cap), connectome.n)
        if require_soma is None:
            require_soma = str(fly.identity.connectome_dataset) in {"malecns_v1", "malecns"}
        indices = _pick_display(connectome, fly.channels, fly.bridge, cap)
        region = _region_codes(connectome, indices)
        soma = _soma_xyz_for(connectome, indices)
        if soma is not None:
            disp, meta = soma
            x, y, z = disp[:, 0], disp[:, 1], disp[:, 2]
            sil = meta["silhouette"]
            coords_source = "somaLocation"
            note = (
                "MaleCNS v1.0 somaLocation in FlyEM voxels (8 nm). Same space as "
                "Google Neuroglancer / Janelia FlyEM. Spikes are this fly's LIF "
                "on the MaleCNS graph, not recorded physiology."
            )
            sil_x, sil_y, sil_z = sil[:, 0], sil[:, 1], sil[:, 2]
        else:
            if require_soma:
                raise RuntimeError(
                    "MaleCNS somaLocation is required for the live brain panel; "
                    "atlas fallback is not the connectome."
                )
            x, y, z = _atlas_xyz(connectome, indices)
            coords_source = "atlas"
            note = (
                "Atlas-mapped onto a Drosophila CNS template from superclass/side. "
                "Not EM skeleton coordinates. Used when MaleCNS somaLocation is unavailable."
            )
            sil_x = sil_y = sil_z = None
        print(f"[brain] coords_source={coords_source} display={indices.size}/{connectome.n}")
        edge_pre, edge_post, edge_sign = _display_edges(connectome, indices)
        sc_full = np.asarray(connectome.superclass, dtype=object)
        full_names, full_inv = np.unique(sc_full, return_inverse=True)
        return cls(
            indices=indices,
            x=x,
            y=y,
            z=z,
            region=region,
            edge_pre=edge_pre,
            edge_post=edge_post,
            edge_sign=edge_sign,
            body_id=connectome.neuron_ids[indices].astype(np.uint64),
            superclass=np.asarray(connectome.superclass[indices], dtype=object),
            note=note,
            n_full=connectome.n,
            n_edges_full=connectome.n_edges,
            coords_source=coords_source,
            silhouette_x=sil_x,
            silhouette_y=sil_y,
            silhouette_z=sil_z,
            full_sc_names=full_names,
            full_sc_inv=full_inv.astype(np.int32, copy=False),
        )

    def layout(self) -> dict:
        payload = {
            "n_full": self.n_full,
            "n_display": int(self.indices.size),
            "n_edges_full": self.n_edges_full,
            "n_edges": int(self.edge_pre.size),
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "z": self.z.tolist(),
            "region": self.region.tolist(),
            "edge_pre": self.edge_pre.tolist(),
            "edge_post": self.edge_post.tolist(),
            "edge_sign": self.edge_sign.tolist(),
            "ids": [str(int(i)) for i in self.body_id],
            "superclass": [str(s) for s in self.superclass],
            "regions": list(REGIONS),
            "atlas": {k: {"c": list(v["c"]), "r": list(v["r"])} for k, v in _ATLAS.items()},
            "note": self.note,
            "coords_source": self.coords_source,
            "neuroglancer": self.neuroglancer,
        }
        if self.silhouette_x is not None:
            payload["silhouette_x"] = self.silhouette_x.tolist()
            payload["silhouette_y"] = self.silhouette_y.tolist()
            payload["silhouette_z"] = self.silhouette_z.tolist()
        return payload

    def snapshot(self, fly, counts: np.ndarray, duration_s: float) -> dict:
        idx = self.indices
        window = np.asarray(counts, dtype=np.int32)
        shown = window[idx] if idx.size else np.zeros(0, dtype=np.int32)
        spiked = np.flatnonzero(shown > 0).astype(np.int32)
        active = int((window > 0).sum())
        total = int(window.sum())
        cmd = fly.bridge.last_command
        ch = fly.channels

        def rate(group) -> float:
            g = np.asarray(group, dtype=np.int32)
            if g.size == 0 or duration_s <= 0:
                return 0.0
            return float(window[g].sum()) / (duration_s * g.size)

        walk = float(cmd.walk_hz)
        rest = 1.0 if cmd.mode == "rest" else max(0.0, 1.0 - min(1.0, walk / 8.0))
        groom = 1.0 if cmd.mode == "groom" else float(min(1.0, fly.physiology.state.grooming_drive))
        fly_bar = 1.0 if cmd.mode == "fly" else float(min(1.0, fly.physiology.state.flight_drive))
        region_spikes = {name: 0 for name in REGIONS}
        for code, nspk in zip(self.region[spiked], shown[spiked]):
            region_spikes[REGIONS[int(code)]] += int(nspk)

        order = np.argsort(-shown[spiked])[:16] if spiked.size else []
        cells = []
        for k in order:
            li = int(spiked[k])
            cells.append(
                {
                    "i": li,
                    "id": str(int(self.body_id[li])),
                    "class": str(self.superclass[li]),
                    "region": REGIONS[int(self.region[li])],
                    "spikes": int(shown[li]),
                }
            )
        by_class: dict[str, int] = {}
        shown_classes = self.superclass[spiked] if spiked.size else []
        shown_counts = shown[spiked] if spiked.size else []
        for name, nspk in zip(shown_classes, shown_counts):
            key = str(name or "unknown")
            by_class[key] = by_class.get(key, 0) + int(nspk)
        full_class: dict[str, int] = {}
        if (
            total
            and self.full_sc_names is not None
            and self.full_sc_inv is not None
            and window.size == self.full_sc_inv.size
        ):
            sums = np.bincount(
                self.full_sc_inv, weights=window.astype(np.float64), minlength=self.full_sc_names.size
            )
            full_class = {
                str(name): int(s)
                for name, s in zip(self.full_sc_names, sums)
                if s
            }
        return {
            "window_ms": duration_s * 1000.0,
            "total_spikes": total,
            "active": active,
            "n_display": int(idx.size),
            "n_full": self.n_full,
            "display_spikes": int(shown.sum()),
            "spiked": spiked.tolist(),
            "cells": cells,
            "regions": region_spikes,
            "pathways": {
                "walk": {"hz": walk, "n": int(fly.bridge.walk_indices.size)},
                "steer_l": {"hz": float(cmd.steer_l_hz), "n": int(fly.bridge.steer_left.size)},
                "steer_r": {"hz": float(cmd.steer_r_hz), "n": int(fly.bridge.steer_right.size)},
                "visual_l": {"hz": rate(ch.left_eye), "n": int(ch.left_eye.size)},
                "visual_r": {"hz": rate(ch.right_eye), "n": int(ch.right_eye.size)},
                "contact": {"hz": rate(ch.contact), "n": int(ch.contact.size)},
            },
            "readout": {
                "walk": float(min(1.0, walk / 12.0)),
                "steer_l": float(min(1.0, cmd.steer_l_hz / 20.0)),
                "steer_r": float(min(1.0, cmd.steer_r_hz / 20.0)),
                "rest": float(rest),
                "groom": float(groom),
                "fly": float(fly_bar),
            },
            "mode": cmd.mode,
            "n_edges": int(self.edge_pre.size),
            "n_edges_full": self.n_edges_full,
            "fired_edges": _fired_edges(self.edge_pre, spiked),
            "note": self.note,
            "coords_source": self.coords_source,
            "by_class": by_class,
            "by_superclass": full_class,
        }


def _fired_edges(edge_pre: np.ndarray, spiked: np.ndarray, cap: int = 1800) -> list[int]:
    if edge_pre.size == 0 or spiked.size == 0:
        return []
    hit = np.zeros(int(np.max(edge_pre, initial=0)) + 1, dtype=bool)
    hit[spiked[spiked < hit.size]] = True
    fired = np.flatnonzero(hit[edge_pre]).astype(np.int32)
    if fired.size > cap:
        fired = fired[:cap]
    return fired.tolist()

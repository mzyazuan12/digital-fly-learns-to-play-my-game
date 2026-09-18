"""Stable sensory and descending populations.

These bundles are annotation-based interfaces so the same fly keeps the same
cells across environments. They are not a claim that those cells are a
keyboard, a phototaxis circuit, or a word decoder.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.loader import Connectome


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS", "RHS"}:
        return "R"
    return ""


def _split_lr(indices: np.ndarray, sides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        empty = np.zeros(0, dtype=np.int32)
        return empty, empty
    labels = np.array([_norm_side(sides[i]) for i in indices])
    left = indices[labels == "L"]
    right = indices[labels == "R"]
    if left.size == 0 or right.size == 0:
        half = max(1, indices.size // 2)
        left = indices[:half]
        right = indices[half:]
        if right.size == 0:
            right = indices[-1:]
        if left.size == 0:
            left = indices[:1]
    return left.astype(np.int32), right.astype(np.int32)


def _or_lookup(connectome: Connectome, names: tuple[str, ...]) -> np.ndarray:
    chunks = [connectome.lookup(superclass=name) for name in names]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32)


@dataclass
class SensorimotorChannels:
    left_eye: np.ndarray
    right_eye: np.ndarray
    proprioception: np.ndarray
    contact: np.ndarray
    antenna_left: np.ndarray
    antenna_right: np.ndarray
    olfaction: np.ndarray
    gustation: np.ndarray
    left_descending: np.ndarray
    right_descending: np.ndarray
    notes: dict

    def index_groups(self) -> dict[str, np.ndarray]:
        return {
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "proprioception": self.proprioception,
            "contact": self.contact,
            "antenna_left": self.antenna_left,
            "antenna_right": self.antenna_right,
            "olfaction": self.olfaction,
            "gustation": self.gustation,
            "left_descending": self.left_descending,
            "right_descending": self.right_descending,
        }

    def to_jsonable(self) -> dict:
        payload = {key: value.tolist() for key, value in self.index_groups().items()}
        payload["notes"] = self.notes
        return payload

    @classmethod
    def from_jsonable(cls, payload: dict) -> "SensorimotorChannels":
        notes = payload.get("notes", {})
        arrays = {
            key: np.asarray(payload[key], dtype=np.int32)
            for key in (
                "left_eye",
                "right_eye",
                "proprioception",
                "contact",
                "antenna_left",
                "antenna_right",
                "olfaction",
                "gustation",
                "left_descending",
                "right_descending",
            )
        }
        return cls(notes=notes, **arrays)

    @classmethod
    def from_connectome(cls, connectome: Connectome, seed: int = 0) -> "SensorimotorChannels":
        rng = np.random.default_rng(seed)
        visual = _or_lookup(connectome, ("ol_sensory",))
        if visual.size == 0:
            visual = _or_lookup(connectome, ("ol_intrinsic",))
        if visual.size == 0:
            visual = connectome.lookup(superclass="cb_sensory")
        if visual.size == 0:
            visual = np.arange(min(32, connectome.n), dtype=np.int32)
            vis_src = "first_n_neurons_no_visual_annotation"
        else:
            vis_src = "annotated_ol_or_cb_sensory"
        left_eye, right_eye = _split_lr(visual, connectome.side)

        vnc_s = _or_lookup(
            connectome, ("vnc_sensory", "vnc_sensory_tbc", "sensory_ascending")
        )
        if vnc_s.size == 0:
            vnc_s = np.arange(min(64, connectome.n), dtype=np.int32)
            proprio_src = "first_n_neurons_no_vnc_sensory"
        else:
            proprio_src = "annotated_vnc_sensory"
        vnc_s = rng.permutation(vnc_s)
        half = max(1, vnc_s.size // 2)
        proprioception = vnc_s[:half]
        contact = vnc_s[half:] if vnc_s.size > half else vnc_s

        antenna = _or_lookup(connectome, ("cb_sensory", "cb_sensory_tbc"))
        if antenna.size == 0:
            antenna = visual
            ant_src = "reused_visual_pool"
        else:
            ant_src = "annotated_cb_sensory"
        antenna_left, antenna_right = _split_lr(antenna, connectome.side)
        olfaction = antenna_left
        gustation = antenna_right

        dns = _or_lookup(connectome, ("descending_neuron", "descending_neuron_tbc"))
        if dns.size < 4:
            dns = np.arange(max(0, connectome.n - 32), connectome.n, dtype=np.int32)
            dn_src = "last_n_neurons_insufficient_DNs"
        else:
            dn_src = "annotated_descending_neurons"
        left_dn, right_dn = _split_lr(dns, connectome.side)

        notes = {
            "visual_source": vis_src,
            "proprio_source": proprio_src,
            "antenna_source": ant_src,
            "descending_source": dn_src,
            "engineered_interface": True,
            "not_a_biological_keyboard": True,
            "not_identified_phototaxis_circuit": True,
            "n_left_eye": int(left_eye.size),
            "n_right_eye": int(right_eye.size),
            "n_left_dn": int(left_dn.size),
            "n_right_dn": int(right_dn.size),
        }
        return cls(
            left_eye=left_eye,
            right_eye=right_eye,
            proprioception=proprioception.astype(np.int32),
            contact=contact.astype(np.int32),
            antenna_left=antenna_left,
            antenna_right=antenna_right,
            olfaction=olfaction.astype(np.int32),
            gustation=gustation.astype(np.int32),
            left_descending=left_dn,
            right_descending=right_dn,
            notes=notes,
        )

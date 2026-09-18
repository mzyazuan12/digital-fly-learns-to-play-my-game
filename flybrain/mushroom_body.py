"""Mushroom-body associative plasticity.

Do not apply a generic reward rule to every MaleCNS synapse.

Drosophila learning is concentrated in Kenyon-cell → mushroom-body-output
synapses, modulated by dopaminergic neurons (Aso / Heisenberg / Modi).
Anatomical counts stay frozen. Only the plastic component of KC→MBON
edges moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.loader import Connectome
from flybrain.network import LIFNetwork
from organism.config import ParameterProvenance


def _lookup_any(connectome: Connectome, *, prefixes: tuple[str, ...], superclasses: tuple[str, ...] = ()) -> np.ndarray:
    chunks = []
    for prefix in prefixes:
        exact = connectome.lookup(type_exact=prefix)
        if exact.size:
            chunks.append(exact)
        else:
            found = connectome.lookup(type_prefix=prefix)
            if found.size:
                chunks.append(found)
    for name in superclasses:
        found = connectome.lookup(superclass=name)
        if found.size:
            chunks.append(found)
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32)


@dataclass
class MushroomBodyParams:
    lr: float = 5e-3
    eligibility_decay: float = 0.92
    plastic_min: float = -0.95
    plastic_max: float = 4.0
    dopamine_scale: float = 1.0


class MushroomBodyLearning:
    """Three-factor rule on identified KC→MBON synapses only."""

    def __init__(self, net: LIFNetwork, params: MushroomBodyParams | None = None):
        self.net = net
        self.params = params or MushroomBodyParams()
        graph = net.connectome
        self.kenyon = _lookup_any(
            graph,
            prefixes=("KC", "KCg", "KCab", "KCa'b'", "Kenyon"),
            superclasses=("kenyon_cell",),
        )
        self.mbon = _lookup_any(
            graph,
            prefixes=("MBON",),
            superclasses=("mushroom_body_output", "mbon"),
        )
        self.dan = _lookup_any(
            graph,
            prefixes=("DAN", "PAM", "PPL1"),
            superclasses=("dopaminergic",),
        )
        self.edge_mask = self._kc_mbon_mask()
        self.eligibility = np.zeros(graph.n_edges, dtype=np.float32)
        self.n_updates = 0
        self.last_dopamine = 0.0
        self.changed_edges = 0
        self.provenance = {
            "rule": "KC_eligibility x DAN x MBON_post",
            "architecture": "Aso / Modi mushroom-body compartments (simplified)",
            "plastic_edges_only": "KC→MBON",
            "anatomical_frozen": True,
            "parameter_provenance": ParameterProvenance.LITERATURE_DERIVED.value,
            "numeric_lr": ParameterProvenance.ASSUMED.value,
        }
        self.notes = {
            "n_kc": int(self.kenyon.size),
            "n_mbon": int(self.mbon.size),
            "n_dan": int(self.dan.size),
            "n_plastic_edges": int(self.edge_mask.sum()),
            "missing_circuit": bool(self.edge_mask.sum() == 0),
        }

    def _kc_mbon_mask(self) -> np.ndarray:
        n_edges = self.net.connectome.n_edges
        mask = np.zeros(n_edges, dtype=bool)
        if self.kenyon.size == 0 or self.mbon.size == 0:
            return mask
        mbon_set = set(int(i) for i in self.mbon)
        ptr = self.net.ptr
        post = self.net.post
        for kc in self.kenyon:
            start = int(ptr[int(kc)])
            end = int(ptr[int(kc) + 1])
            for e in range(start, end):
                if int(post[e]) in mbon_set:
                    mask[e] = True
        return mask

    def observe_activity(self) -> None:
        if not np.any(self.edge_mask):
            return
        spiked = np.flatnonzero(self.net.last_spikes)
        decay = np.float32(self.params.eligibility_decay)
        if spiked.size == 0:
            self.eligibility *= decay
            return
        kc_set = set(int(i) for i in self.kenyon) if self.kenyon.size else set()
        ptr = self.net.ptr
        post = self.net.post
        post_trace = self.net.post_trace
        for i in spiked:
            if int(i) not in kc_set:
                continue
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            sl = slice(start, end)
            self.eligibility[sl] += post_trace[post[sl]] * self.edge_mask[sl]
        self.eligibility *= decay

    def apply_dopamine(self, dopamine: float) -> int:
        """Modulate KC→MBON plastic components. Anatomical weights stay frozen."""
        self.last_dopamine = float(dopamine)
        if dopamine == 0.0 or not np.any(self.edge_mask):
            return 0
        delta = np.float32(self.params.lr * dopamine * self.params.dopamine_scale) * self.eligibility
        delta = np.where(self.edge_mask, delta, 0)
        moved = delta != 0
        if not np.any(moved):
            return 0
        self.net.plastic_component[moved] = np.clip(
            self.net.plastic_component[moved] + delta[moved],
            self.params.plastic_min,
            self.params.plastic_max,
        )
        self.net._rebuild_weights()
        self.n_updates += 1
        self.changed_edges = int(np.count_nonzero(np.abs(self.net.plastic_component) > 1e-6))
        self.eligibility *= np.float32(0.1)
        return int(np.count_nonzero(moved))

    def apply_reward(self, reward: float) -> int:
        """External/experimental reinforcement. Still restricted to KC→MBON."""
        return self.apply_dopamine(reward)

    def snapshot(self) -> dict:
        return {
            **self.notes,
            "n_updates": self.n_updates,
            "changed_edges": self.changed_edges,
            "last_dopamine": self.last_dopamine,
            "plastic_mean": float(self.net.plastic_component.mean()),
            "provenance": self.provenance,
        }

"""Reward-modulated plasticity on EXISTING MaleCNS edges only.

Anatomical synapse counts stay frozen. A separate efficacy scale is the
only learned value. No new connections are created.

Weight changes alone are not evidence of learning — see training controls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.network import LIFNetwork, MAX_PLASTIC_FACTOR, MIN_PLASTIC_FACTOR


@dataclass
class PlasticityParams:
    lr: float = 1e-4
    eligibility_decay: float = 0.95
    efficacy_min: float = 0.05
    efficacy_max: float = 5.0
    trace_from: str = "pre_post"
    # If set, only these edge indices are plastic (still existing edges).
    edge_mask: np.ndarray | None = None


class RewardModulatedPlasticity:
    def __init__(self, net: LIFNetwork, params: PlasticityParams | None = None):
        self.net = net
        self.params = params or PlasticityParams()
        self.eligibility = np.zeros(net.connectome.n_edges, dtype=np.float32)
        self.n_updates = 0
        self.last_reward = 0.0
        self.changed_edges = 0

    def observe_activity(self) -> None:
        """Accumulate eligibility from the spikes of the last tick/window."""
        spiked = np.flatnonzero(self.net.last_spikes)
        if spiked.size == 0:
            self.eligibility *= np.float32(self.params.eligibility_decay)
            return
        ptr = self.net.ptr
        post = self.net.post
        post_trace = self.net.post_trace
        for i in spiked:
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            self.eligibility[start:end] += post_trace[post[start:end]]
        self.eligibility *= np.float32(self.params.eligibility_decay)

    def apply_reward(self, reward: float) -> int:
        """Modulate existing efficacies. Returns how many synapses moved."""
        self.last_reward = float(reward)
        if reward == 0.0:
            return 0
        p = self.params
        delta = np.float32(p.lr * reward) * self.eligibility
        if p.edge_mask is not None:
            delta = np.where(p.edge_mask, delta, 0)
        moved = delta != 0
        if not np.any(moved):
            return 0
        lo = getattr(p, "plastic_min", MIN_PLASTIC_FACTOR - 1.0)
        hi = getattr(p, "plastic_max", MAX_PLASTIC_FACTOR - 1.0)
        self.net.plastic_component[moved] = np.clip(
            self.net.plastic_component[moved] + delta[moved], lo, hi
        )
        # Keep combined scale in sync for old readers. Anatomy is never written.
        self.net._rebuild_weights()
        self.n_updates += 1
        self.changed_edges = int(np.count_nonzero(np.abs(self.net.efficacy - 1.0) > 1e-6))
        # Consume traces after a meaningful outcome so credit does not linger
        # across unrelated later rewards.
        self.eligibility *= np.float32(0.1)
        return int(np.count_nonzero(moved))

    def snapshot(self) -> dict:
        eff = self.net.efficacy
        return {
            "n_updates": self.n_updates,
            "changed_edges": self.changed_edges,
            "efficacy_mean": float(eff.mean()),
            "efficacy_std": float(eff.std()),
            "efficacy_min": float(eff.min()),
            "efficacy_max": float(eff.max()),
            "last_reward": self.last_reward,
        }

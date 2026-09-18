"""Control policies. Topology or plasticity changes that are NOT 'the fly learned'."""

from __future__ import annotations

import random

from flybrain.loader import Connectome, rewired_connectome, shuffled_connectome
from flybrain.populations import ACTIONS


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, _observation: dict) -> str:
        return self.rng.choice(ACTIONS)


def apply_control(connectome: Connectome, name: str, seed: int) -> Connectome:
    rng = __import__("numpy").random.default_rng(seed)
    if name in {"none", "intact", "frozen"}:
        return connectome
    if name == "shuffled":
        return shuffled_connectome(connectome, rng)
    if name == "rewired":
        return rewired_connectome(connectome, rng)
    raise ValueError(f"Unknown control {name}")

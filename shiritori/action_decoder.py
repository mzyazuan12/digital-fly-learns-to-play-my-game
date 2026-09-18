"""Keyboard readout: descending-population rates → A–Z / BACKSPACE / ENTER / NO-OP.

The decoder measures spike counts. It does not contain a dictionary, prefix
table, or word-solving model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.populations import ACTIONS, InterfaceMap


@dataclass
class DecodeParams:
    min_rate_hz: float = 1.0
    # If the top two groups are within this fraction, emit NOOP instead of guessing.
    tie_frac: float = 0.05


class KeyboardDecoder:
    def __init__(self, interface: InterfaceMap, params: DecodeParams | None = None):
        self.interface = interface
        self.params = params or DecodeParams()
        self.last_rates = np.zeros(len(ACTIONS), dtype=np.float64)
        self.last_action = "NOOP"

    def decode(self, counts: np.ndarray, duration_s: float) -> str:
        if duration_s <= 0:
            raise ValueError("Positive readout window required")
        rates = np.array(
            [
                float(counts[group].sum()) / (duration_s * max(1, group.size))
                for group in self.interface.action_groups
            ],
            dtype=np.float64,
        )
        self.last_rates = rates
        top = int(np.argmax(rates))
        best = rates[top]
        if best < self.params.min_rate_hz:
            self.last_action = "NOOP"
            return "NOOP"
        second = float(np.partition(rates, -2)[-2]) if rates.size > 1 else 0.0
        if second > 0 and (best - second) / best < self.params.tie_frac:
            self.last_action = "NOOP"
            return "NOOP"
        self.last_action = ACTIONS[top]
        return self.last_action

    def action_index(self, name: str) -> int:
        return ACTIONS.index(name)

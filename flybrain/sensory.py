"""Game-state → current injection. No learned encoder, no word solver."""

from __future__ import annotations

from dataclasses import dataclass

from flybrain.network import LIFNetwork
from flybrain.populations import (
    ACTIONS,
    LETTERS,
    N_PREFIX_SLOTS,
    N_WORD_SLOTS,
    InterfaceMap,
)


@dataclass
class EncodeParams:
    letter_current: float = 30.0
    analog_current: float = 12.0
    reward_current: float = 20.0


class SensoryEncoder:
    """Fixed one-hot / analog currents into documented neuron groups."""

    def __init__(self, interface: InterfaceMap, params: EncodeParams | None = None):
        self.interface = interface
        self.params = params or EncodeParams()

    def apply(self, net: LIFNetwork, observation: dict, reward_pulse: float = 0.0) -> None:
        net.clear_drive()
        p = self.params
        self._spell(net, observation.get("prefix", ""), self.interface.letter_prefix, N_PREFIX_SLOTS, p.letter_current)
        self._spell(net, observation.get("previous_word", ""), self.interface.letter_previous, N_WORD_SLOTS, p.letter_current)
        self._spell(net, observation.get("buffer", ""), self.interface.letter_buffer, N_WORD_SLOTS, p.letter_current)
        self._analog(net, self.interface.timer, observation.get("timer_frac", 1.0), p.analog_current)
        self._analog(net, self.interface.tries, observation.get("tries_frac", 1.0), p.analog_current)
        self._analog(net, self.interface.lives, observation.get("lives_frac", 1.0), p.analog_current)
        self._analog(net, self.interface.score, observation.get("score_frac", 0.0), p.analog_current)
        if reward_pulse:
            net.inject(self.interface.reward, p.reward_current * float(reward_pulse))

    def _spell(self, net, text: str, groups, n_slots: int, current: float) -> None:
        cleaned = "".join(ch for ch in str(text).lower() if ch in LETTERS)
        cleaned = cleaned[-n_slots:]
        # Right-align so the last letter sits in the last slot.
        padded = cleaned.rjust(n_slots)
        for slot, ch in enumerate(padded):
            if ch not in LETTERS:
                continue
            li = LETTERS.index(ch)
            net.inject(groups[slot * 26 + li], current)

    @staticmethod
    def _analog(net, indices, frac: float, current: float) -> None:
        f = max(0.0, min(1.0, float(frac)))
        if f > 0:
            net.inject(indices, current * f)

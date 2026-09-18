"""Reward table. Values are experimental, not biology."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardTable:
    correct_letter: float = 0.02
    valid_word: float = 1.0
    continue_chain: float = 2.0
    invalid_word: float = -1.0
    timeout: float = -2.0
    win: float = 10.0
    lose: float = -5.0
    wrong_letter: float = -0.01
    noop: float = -0.001
    backspace: float = -0.002
    letter_match: float = 1.0  # stage-1 identity task

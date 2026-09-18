"""Shared Shiritori timing / chain / scoring constants from last-dletter."""

from __future__ import annotations

MAX_LIVES = 2
MAX_TRIES = 5
MAX_CHAIN = 4
CASUAL_MIN_GROUPS = 4
DIFFICULTIES = ("casual", "pro", "featherine")


def desired_chain_len(completed_rounds: int) -> int:
    n = max(0, int(completed_rounds or 0))
    if n < 3:
        return 1
    if n < 7:
        return 2
    if n < 11:
        return 3
    return 4


def turn_time_limit(rounds_since_reset: int) -> int:
    """Seconds on the live site. Training discretizes this into keystroke budgets."""
    if rounds_since_reset <= 10:
        return 15
    if rounds_since_reset <= 10 + 15:
        return 13
    if rounds_since_reset <= 10 + 15 + 17:
        return 10
    if rounds_since_reset <= 10 + 15 + 17 + 18:
        return 7
    return 5


def keystroke_budget(rounds_since_reset: int, keys_per_second: float = 3.0) -> int:
    """Training stand-in for the wall-clock turn timer."""
    return max(8, int(round(turn_time_limit(rounds_since_reset) * keys_per_second)))

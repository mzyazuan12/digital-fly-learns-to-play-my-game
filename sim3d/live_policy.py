"""Live-site play policy: type a remaining word, or give up only when none exist.

The fly never clicks Skip / forfeit as a strategy. A give-up is allowed only
when the local last-dletter dictionary has zero remaining legal replies for
the prefix currently on shiritori.lol — that is a model/dictionary failure.
"""

from __future__ import annotations

import random

from shiritori.dictionary import GameDict, playable_count

LIVE_MODES = ("casual", "pro", "featherine", "lambdadelta")
MODE_BUTTON_IDS = {
    "casual": "btn-casual",
    "pro": "btn-pro",
    "featherine": "btn-featherine",
    "lambdadelta": "btn-lambdadelta",
}


def remaining_words(dict_: GameDict, prefix: str, used: set[str]) -> list[str]:
    key = str(prefix or "").lower()
    bucket = dict_.by_prefix.get(key) or []
    plen = len(key)
    return [w for w in bucket if w not in used and len(w) > plen]


def should_give_up(dict_: GameDict, prefix: str, used: set[str]) -> bool:
    """True only when the model has no remaining legal word for this prefix."""
    if not str(prefix or "").strip():
        return False
    return playable_count(dict_, str(prefix).lower(), used, cap=1) == 0


def pick_live_word(
    dict_: GameDict,
    prefix: str,
    used: set[str],
    rng: random.Random | None = None,
) -> str | None:
    """Any still-legal letters-only reply. None means the model failed."""
    rng = rng or random.Random()
    options = remaining_words(dict_, prefix, used)
    if not options:
        return None
    return options[rng.randrange(len(options))]


def normalize_mode(raw: str | None) -> str:
    mode = str(raw or "casual").strip().lower()
    if mode in {"lambda", "lambda-delta", "spam"}:
        return "lambdadelta"
    if mode not in LIVE_MODES:
        raise ValueError(f"unknown live mode {raw!r}")
    return mode

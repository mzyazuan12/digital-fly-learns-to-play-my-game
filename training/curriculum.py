"""Vocabulary curriculum. Advance only when evaluation beats controls."""

from __future__ import annotations

TINY_20 = [
    "apple",
    "elephant",
    "tiger",
    "rabbit",
    "tree",
    "egg",
    "goat",
    "ant",
    "tomato",
    "orange",
    "eagle",
    "ear",
    "rat",
    "tap",
    "pear",
    "red",
    "dog",
    "grape",
    "eel",
    "lemon",
]

# Closed-ish extra rungs used after the 20-word toy set.
TINY_100_SEED = TINY_20 + [
    "night", "time", "earth", "hand", "day", "year", "water", "river",
    "road", "door", "room", "moon", "north", "house", "east", "town",
    "name", "end", "dream", "map", "plant", "table", "engine", "echo",
    "ocean", "nest", "star", "rock", "king", "garden", "needle", "lamp",
    "paper", "ring", "gold", "dust", "treehouse", "ember", "rain", "nestle",
    "ice", "edge", "open", "nesting", "gift", "trail", "leaf", "fire",
    "engineered", "delta", "anchor", "rope", "ember", "sand", "dusk", "kite",
    "elm", "mint", "token", "number", "root", "thorn", "nest", "stone",
    "echoed", "dawn", "winter", "ridge", "elm", "meadow", "wolf", "forest",
    "tide", "ember", "raven", "nest", "tower", "ridge", "ember", "reed",
]


def tiny_words(n: int, catalog: list[str] | None = None) -> set[str]:
    src = catalog or TINY_20
    # Preserve order, drop duplicates, cut to n.
    out = []
    seen = set()
    for w in src:
        if w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= n:
            break
    return set(out)


def stage_vocab(stage: str, full_casual: set[str] | None = None) -> set[str] | None:
    """None means 'use the loaded letters-only dictionary'."""
    if stage in {"letters"}:
        return set()
    if stage in {"tiny", "tiny20", "fly1"}:
        return tiny_words(20)
    if stage in {"tiny100", "100"}:
        return tiny_words(100, TINY_100_SEED)
    if stage in {"500"}:
        return None  # caller samples
    if stage in {"2000"}:
        return None
    if stage in {"casual", "pro", "featherine", "fly2", "fly3", "fly4"}:
        return full_casual
    raise ValueError(f"Unknown stage {stage}")

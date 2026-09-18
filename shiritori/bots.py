"""Opponent word pickers ported from last-dletter/lib/gameDict.js.

These bots are the *environment*, not the fly. The fly never calls them to
choose its own word.
"""

from __future__ import annotations

import random
from typing import Literal

from shiritori.dictionary import (
    GameDict,
    group_count,
    has_playable_words,
    min_solve_len,
    playable_count,
    resolve_prefix_len,
)
from shiritori.rules import MAX_CHAIN

Playstyle = Literal["special", "sparse", "longSolve", "chaos", "absurd"]

SEED_LETTERS = list("abcdefghilmnoprstuvw")
PLAYSTYLE_WEIGHTS = {
    "special": {"special": 4.0, "sparse": 3.0, "longSolve": 2.2, "hold": 2.0, "chaos": 0.6},
    "sparse": {"special": 1.4, "sparse": 4.2, "longSolve": 2.6, "hold": 2.2, "chaos": 0.7},
    "longSolve": {"special": 1.6, "sparse": 3.2, "longSolve": 4.2, "hold": 2.2, "chaos": 0.7},
    "chaos": {"special": 1.2, "sparse": 2.8, "longSolve": 2.4, "hold": 1.4, "chaos": 3.0},
    "absurd": {"special": 2.8, "sparse": 3.4, "longSolve": 3.2, "hold": 2.4, "chaos": 1.2},
}


def roll_featherine_playstyle(rng: random.Random | None = None) -> Playstyle:
    rng = rng or random
    return rng.choice(list(PLAYSTYLE_WEIGHTS))  # type: ignore[return-value]


def _weighted_pick(items, weight_of, rng: random.Random):
    total = sum(max(0.0, weight_of(it)) for it in items)
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    for it in items:
        r -= max(0.0, weight_of(it))
        if r <= 0:
            return it
    return items[-1]


def pick_bot_word(
    dict_: GameDict,
    required_prefix: str,
    used: set[str],
    *,
    min_next_groups: int = 0,
    min_next_playable: int = 1,
    next_chain_len: int = 1,
    rng: random.Random | None = None,
) -> str | None:
    rng = rng or random.Random()
    prefix = str(required_prefix or "").lower()
    bucket = dict_.by_prefix.get(prefix)
    if not bucket:
        return None
    min_next_playable = max(1, int(min_next_playable or 1))

    def suitable(word: str) -> bool:
        if word in used or len(word) <= len(prefix):
            return False
        next_used = set(used)
        next_used.add(word)
        resolved = resolve_prefix_len(
            dict_, word, next_chain_len, next_used, min_next_groups, min_next_playable
        )
        return playable_count(dict_, word[-resolved:], next_used, min_next_playable) >= min_next_playable

    attempts = min(200, len(bucket))
    for _ in range(attempts):
        word = bucket[rng.randrange(len(bucket))]
        if suitable(word):
            return word
    for word in bucket:
        if suitable(word):
            return word
    if min_next_groups > 0 or next_chain_len > 1:
        return pick_bot_word(
            dict_,
            prefix,
            used,
            min_next_groups=0,
            min_next_playable=min_next_playable,
            next_chain_len=1,
            rng=rng,
        )
    if min_next_playable > 1:
        return None
    for word in bucket:
        if word not in used and len(word) > len(prefix):
            return word
    return None


def _playstyle_score(s: dict, weights: dict, rng: random.Random) -> float:
    w = weights
    trap = max(1, s["trap"])
    sparse = max(0.0, 14 - min(trap, 14)) ** 1.85
    rarity_gate = 10 / (trap + 2)
    long_solve = min(s["min_solve"], 28)
    groups = max(0, 12 - min(s["groups"], 12))
    special = 1 if s["special"] else 0
    hold = 1 if s["held"] else 0
    noise = rng.random() * 4 * (w["chaos"] or 1)
    trap_quality = (sparse * 3.2 * w["sparse"] + long_solve * 5.5 * w["longSolve"]) * rarity_gate
    return (
        trap_quality
        + special * 28 * w["special"] * rarity_gate
        + groups * 0.6 * rarity_gate
        + hold * 18 * w["hold"]
        + noise
    )


def pick_featherine_word(
    dict_: GameDict,
    required_prefix: str,
    used: set[str],
    *,
    next_chain_len: int = 1,
    playstyle: Playstyle | None = None,
    rng: random.Random | None = None,
) -> str | None:
    rng = rng or random.Random()
    prefix = str(required_prefix or "").lower()
    bucket = dict_.by_prefix.get(prefix)
    if not bucket:
        return None
    style: Playstyle = playstyle if playstyle in PLAYSTYLE_WEIGHTS else roll_featherine_playstyle(rng)
    weights = PLAYSTYLE_WEIGHTS[style]
    caches_playable: dict[str, int] = {}
    caches_groups: dict[str, int] = {}
    caches_min: dict[str, int] = {}

    def info_for(word: str) -> dict | None:
        length = resolve_prefix_len(dict_, word, next_chain_len, used, 0)
        pfx = word[-length:]
        trap = caches_playable.get(pfx)
        if trap is None:
            trap = playable_count(dict_, pfx, used)
            caches_playable[pfx] = trap
        if trap <= 0:
            return None
        if pfx not in caches_groups:
            caches_groups[pfx] = group_count(dict_, pfx)
        if pfx not in caches_min:
            caches_min[pfx] = min_solve_len(dict_, pfx, used)
        held = length >= min(next_chain_len, MAX_CHAIN, len(word))
        return {
            "word": word,
            "prefix": pfx,
            "trap": trap,
            "min_solve": caches_min[pfx],
            "groups": caches_groups[pfx],
            "special": ("'" in pfx) or ("-" in pfx),
            "held": held,
            "len": length,
        }

    scored = []
    max_scan = len(bucket) if len(bucket) <= 4000 else 3500
    step = 1 if len(bucket) <= max_scan else len(bucket) // max_scan
    offset = rng.randrange(step) if step > 1 else 0
    for i in range(offset, len(bucket), step):
        word = bucket[i]
        if word in used or len(word) <= len(prefix):
            continue
        info = info_for(word)
        if info:
            scored.append(info)
    if not scored and step > 1:
        for word in bucket:
            if word in used or len(word) <= len(prefix):
                continue
            info = info_for(word)
            if info:
                scored.append(info)
            if len(scored) >= 400:
                break
    if not scored:
        return pick_bot_word(dict_, prefix, used, next_chain_len=next_chain_len, rng=rng)

    holders = [s for s in scored if s["held"]]
    pool = holders or scored
    min_trap = min(s["trap"] for s in pool)
    ceiling = max(min_trap + 1, min(5, min_trap * 2))
    rare = [s for s in pool if s["trap"] <= ceiling]
    pool = rare or pool
    if style != "chaos":
        specials = [s for s in pool if s["special"]]
        if specials:
            pool = specials
    ranked = sorted(
        ((_playstyle_score(s, weights, rng), s) for s in pool),
        key=lambda x: (-x[0], x[1]["trap"], -x[1]["min_solve"]),
    )
    best_c = ranked[0][0]
    band_ratio = 0.45 if style == "chaos" else 0.7
    band_cap = 28 if style == "chaos" else 16
    threshold = best_c * band_ratio
    band = []
    for score, s in ranked:
        if score >= threshold and len(band) < band_cap:
            band.append((score, s))
        else:
            break
    if not band:
        band = [ranked[0]]
    chosen = _weighted_pick(band, lambda e: max(0.1, e[0]) ** 1.6, rng)
    return chosen[1]["word"]


def pick_seed_word(
    dict_: GameDict,
    min_next_groups: int,
    *,
    featherine: bool = False,
    min_next_playable: int = 1,
    rng: random.Random | None = None,
) -> str:
    rng = rng or random.Random()
    empty: set[str] = set()
    for _ in range(40):
        letter = rng.choice(SEED_LETTERS)
        if featherine and min_next_playable <= 1:
            word = pick_featherine_word(dict_, letter, empty, next_chain_len=1, rng=rng)
        else:
            word = pick_bot_word(
                dict_,
                letter,
                empty,
                min_next_groups=min_next_groups,
                min_next_playable=min_next_playable,
                next_chain_len=1,
                rng=rng,
            )
        if word and len(word) >= 3:
            return word
    if featherine and min_next_playable <= 1:
        return pick_featherine_word(dict_, "a", empty, next_chain_len=1, rng=rng) or "apple"
    for letter in SEED_LETTERS:
        word = pick_bot_word(
            dict_,
            letter,
            empty,
            min_next_playable=min_next_playable,
            next_chain_len=1,
            rng=rng,
        )
        if word and len(word) >= 3:
            return word
    return pick_bot_word(dict_, "a", empty, min_next_playable=min_next_playable, rng=rng) or "apple"


def opponent_word(
    dict_: GameDict,
    prefix: str,
    used: set[str],
    difficulty: str,
    *,
    next_chain_len: int,
    playstyle: Playstyle | None = None,
    rng: random.Random | None = None,
) -> str | None:
    min_groups = 4 if difficulty == "casual" else 0
    if difficulty == "featherine":
        return pick_featherine_word(
            dict_, prefix, used, next_chain_len=next_chain_len, playstyle=playstyle, rng=rng
        )
    return pick_bot_word(
        dict_,
        prefix,
        used,
        min_next_groups=min_groups,
        next_chain_len=next_chain_len,
        rng=rng,
    )


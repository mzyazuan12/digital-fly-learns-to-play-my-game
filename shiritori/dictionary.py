"""Prefix-indexed dictionary and validation, ported from last-dletter/lib/gameDict.js."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from shiritori.prefix_blacklist import is_blocked_prefix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORDS = ROOT / "data" / "shiritori" / "words.txt"

MAX_PREFIX = 4
MAX_WORD_LEN = 128
MAX_USED = 8000
WORD_SHAPE_RE = re.compile(r"^(?=.*[a-z])[a-z'-]+$")
LETTER_ONLY_RE = re.compile(r"^[a-z]+$")


def is_word_shape(raw: str) -> bool:
    return isinstance(raw, str) and bool(WORD_SHAPE_RE.match(raw))


@dataclass
class GameDict:
    words: set[str]
    by_prefix: dict[str, list[str]]
    group_cache: dict[str, int] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.words)


def build_dict(word_set: set[str]) -> GameDict:
    words: set[str] = set()
    by_prefix: dict[str, list[str]] = {}
    for raw in word_set:
        if len(raw) < 2 or len(raw) > MAX_WORD_LEN:
            continue
        words.add(raw)
        max_len = min(MAX_PREFIX, len(raw))
        for length in range(1, max_len + 1):
            key = raw[:length]
            by_prefix.setdefault(key, []).append(raw)
    return GameDict(words=words, by_prefix=by_prefix)


def load_word_set(path: Path | None = None, *, letters_only: bool = False) -> set[str]:
    text = Path(path or DEFAULT_WORDS).read_text(encoding="utf-8")
    words: set[str] = set()
    for line in text.splitlines():
        w = line.strip().lower()
        if not w:
            continue
        if letters_only:
            if LETTER_ONLY_RE.match(w) and len(w) >= 2:
                words.add(w)
        elif re.match(r"^[a-z]+(?:['-][a-z]+)*$", w) and len(w) >= 2:
            words.add(w)
    if len(words) < 20:
        raise ValueError(f"dictionary too small: {len(words)}")
    return words


def load_game_dict(
    path: Path | None = None,
    *,
    letters_only: bool = True,
    subset: set[str] | None = None,
) -> GameDict:
    if subset is not None:
        words = {w.lower() for w in subset}
        if letters_only:
            words = {w for w in words if LETTER_ONLY_RE.match(w)}
        return build_dict(words)
    return build_dict(load_word_set(path, letters_only=letters_only))


def is_valid_word(dict_: GameDict, word: str) -> bool:
    return str(word or "").lower() in dict_.words


def group_count(dict_: GameDict, prefix: str) -> int:
    key = str(prefix or "").lower()
    if key in dict_.group_cache:
        return dict_.group_cache[key]
    bucket = dict_.by_prefix.get(key)
    if not bucket:
        dict_.group_cache[key] = 0
        return 0
    nxt = {w[len(key)] for w in bucket if len(w) > len(key)}
    dict_.group_cache[key] = len(nxt)
    return len(nxt)


def playable_count(dict_: GameDict, prefix: str, used: set[str], cap: int = 64) -> int:
    bucket = dict_.by_prefix.get(str(prefix or "").lower())
    if not bucket:
        return 0
    n = 0
    plen = len(prefix)
    for w in bucket:
        if w not in used and len(w) > plen:
            n += 1
            if n >= cap:
                return cap
    return n


def has_playable_words(dict_: GameDict, prefix: str, used: set[str]) -> bool:
    return playable_count(dict_, prefix, used, cap=1) > 0


def min_solve_len(dict_: GameDict, prefix: str, used: set[str], cap: int = 4096) -> int:
    bucket = dict_.by_prefix.get(str(prefix or "").lower())
    if not bucket:
        return 0
    best = 0
    seen = 0
    for w in bucket:
        if w in used or len(w) <= len(prefix):
            continue
        if best == 0 or len(w) < best:
            best = len(w)
        seen += 1
        if seen >= cap:
            break
    return best


def resolve_prefix_len(
    dict_: GameDict,
    word: str,
    desired_len: int,
    used: set[str],
    min_groups: int = 0,
    min_playable: int = 1,
) -> int:
    """Longest usable suffix length for `word`, from desired_len down to 1."""
    max_len = min(max(1, desired_len), MAX_PREFIX, len(word))
    required = max(1, int(min_playable or 1))

    def try_len(length: int, groups: int) -> bool:
        prefix = word[-length:]
        if is_blocked_prefix(prefix):
            return False
        if playable_count(dict_, prefix, used, required) < required:
            return False
        if groups > 0 and group_count(dict_, prefix) < groups:
            return False
        return True

    for length in range(max_len, 0, -1):
        if try_len(length, min_groups):
            return length
    if min_groups > 0:
        for length in range(max_len, 0, -1):
            if try_len(length, 0):
                return length
    return 1


def is_valid_prefix_of_playable(dict_: GameDict, buffer: str, prefix: str, used: set[str]) -> bool:
    """True if buffer is a prefix of at least one still-playable legal word."""
    if not buffer.startswith(prefix):
        return False
    if len(buffer) <= len(prefix):
        return True
    if len(buffer) <= MAX_PREFIX:
        bucket = dict_.by_prefix.get(buffer, [])
        return any(w not in used and len(w) > len(prefix) for w in bucket)
    head = buffer[:MAX_PREFIX]
    bucket = dict_.by_prefix.get(head, [])
    return any(
        w.startswith(buffer) and w not in used and len(w) > len(prefix) for w in bucket
    )

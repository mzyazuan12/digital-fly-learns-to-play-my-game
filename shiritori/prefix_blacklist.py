"""Exact-match required-prefix blacklist, ported from last-dletter/lib/prefixBlacklist.js."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "shiritori" / "prefix_blacklist.json"

_CONSONANT = set("bcdfghjklmnpqrstvwxyz")
_BLOCKED: set[str] | None = None


def load_blocked(path: Path | None = None) -> set[str]:
    global _BLOCKED
    if _BLOCKED is not None and path is None:
        return _BLOCKED
    data = json.loads(Path(path or DEFAULT_PATH).read_text())
    blocked = set(data["all"]) | set(data["lastLetterOnly"])
    if path is None:
        _BLOCKED = blocked
    return blocked


def is_blocked_prefix(prefix: str, blocked: set[str] | None = None) -> bool:
    """True when `prefix` is exactly blacklisted or a double-consonant digraph."""
    if not prefix:
        return False
    p = str(prefix).lower()
    table = blocked if blocked is not None else load_blocked()
    if p in table:
        return True
    if len(p) == 2 and p[0] == p[1] and p[0] in _CONSONANT:
        return True
    return False

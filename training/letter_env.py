"""Stage-1 letter identity environment: show a letter, require the same key.

No dictionary. The fly must route a stimulated letter population to the
matching descending readout. Sequences are optional later trials.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from shiritori.rewards import RewardTable

LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
ACTIONS = LETTERS + ("BACKSPACE", "ENTER", "NOOP")


@dataclass
class LetterEnv:
    rewards: RewardTable = field(default_factory=RewardTable)
    seed: int = 0
    max_steps: int = 8
    sequences: bool = False
    sequence_len: int = 3

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.reset()

    def reset(self) -> dict:
        if self.sequences:
            n = self.sequence_len
            self.target = "".join(self.rng.choice(LETTERS) for _ in range(n))
        else:
            self.target = self.rng.choice(LETTERS)
        self.buffer = ""
        self.steps = 0
        self.done = False
        return self.observe()

    def observe(self) -> dict:
        shown = self.target if not self.sequences else self.target
        return {
            "prefix": shown[:1],
            "previous_word": shown,
            "buffer": self.buffer,
            "timer_frac": max(0.0, 1.0 - self.steps / max(1, self.max_steps)),
            "tries_frac": 1.0,
            "lives_frac": 1.0,
            "score_frac": 0.0,
            "target": self.target,
        }

    def step(self, action: str):
        if self.done:
            return self.observe(), 0.0, True, {"reason": "done"}
        self.steps += 1
        info = {"action": action, "target": self.target}
        if self.steps > self.max_steps:
            self.done = True
            return self.observe(), self.rewards.timeout, True, {**info, "timeout": True}

        if not self.sequences:
            if action == self.target:
                self.done = True
                return self.observe(), self.rewards.letter_match, True, {**info, "correct": True}
            if action == "NOOP":
                return self.observe(), self.rewards.noop, False, info
            return self.observe(), self.rewards.wrong_letter, False, info

        if action == "BACKSPACE":
            self.buffer = self.buffer[:-1]
            return self.observe(), self.rewards.backspace, False, info
        if action == "NOOP":
            return self.observe(), self.rewards.noop, False, info
        if action == "ENTER":
            ok = self.buffer == self.target
            self.done = True
            reward = self.rewards.letter_match if ok else self.rewards.invalid_word
            return self.observe(), reward, True, {**info, "correct": ok, "buffer": self.buffer}
        if action in LETTERS:
            self.buffer += action
            want = self.target[: len(self.buffer)]
            reward = self.rewards.correct_letter if self.buffer == want else self.rewards.wrong_letter
            return self.observe(), reward, False, {**info, "buffer": self.buffer}
        raise ValueError(action)

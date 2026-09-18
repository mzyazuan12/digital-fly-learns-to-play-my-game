"""Fast Shiritori environment using the same rules as shiritori.lol / last-dletter.

No browser. Letter actions only (A–Z, BACKSPACE, ENTER, NOOP) because that is
the fly keyboard. Hyphen/apostrophe words are therefore out of reach and the
training dictionaries are letters-only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from shiritori.bots import opponent_word, pick_seed_word, roll_featherine_playstyle
from shiritori.dictionary import (
    MAX_WORD_LEN,
    GameDict,
    is_valid_prefix_of_playable,
    is_valid_word,
    is_word_shape,
    load_game_dict,
    playable_count,
    resolve_prefix_len,
)
from shiritori.rewards import RewardTable
from shiritori.rules import (
    CASUAL_MIN_GROUPS,
    MAX_LIVES,
    MAX_TRIES,
    desired_chain_len,
    keystroke_budget,
)

ACTIONS = tuple("abcdefghijklmnopqrstuvwxyz") + ("BACKSPACE", "ENTER", "NOOP")


@dataclass
class StepResult:
    observation: dict
    reward: float
    done: bool
    info: dict


@dataclass
class ShiritoriEnv:
    dict_: GameDict
    difficulty: str = "casual"
    rewards: RewardTable = field(default_factory=RewardTable)
    seed: int = 0
    letter_shaping: bool = True

    def __post_init__(self) -> None:
        if self.difficulty not in {"casual", "pro", "featherine"}:
            raise ValueError(self.difficulty)
        self.rng = random.Random(self.seed)
        self.reset()

    @classmethod
    def from_words(
        cls,
        words: set[str] | None = None,
        *,
        difficulty: str = "casual",
        letters_only: bool = True,
        seed: int = 0,
    ) -> "ShiritoriEnv":
        dict_ = load_game_dict(subset=words, letters_only=letters_only) if words else load_game_dict(
            letters_only=letters_only
        )
        return cls(dict_=dict_, difficulty=difficulty, seed=seed)

    def reset(self) -> dict:
        self.used: set[str] = set()
        self.player_lives = MAX_LIVES
        self.bot_lives = MAX_LIVES
        self.tries_left = MAX_TRIES
        self.score = 0
        self.completed_rounds = 0
        self.timer_round = 1
        self.buffer = ""
        self.phase = "player"
        self.winner = None
        self.turn_steps = 0
        self.playstyle = roll_featherine_playstyle(self.rng)
        min_groups = CASUAL_MIN_GROUPS if self.difficulty == "casual" else 0
        starter = pick_seed_word(
            self.dict_,
            min_groups,
            featherine=self.difficulty == "featherine",
            rng=self.rng,
        )
        self.used.add(starter)
        self.last_word = starter
        self.chain_len = resolve_prefix_len(
            self.dict_, starter, 1, self.used, min_groups, 1
        )
        self.prefix = starter[-self.chain_len :]
        self.budget = keystroke_budget(self.timer_round)
        self.history = [("starter", starter)]
        return self.observe()

    def observe(self) -> dict:
        return {
            "prefix": self.prefix,
            "previous_word": self.last_word,
            "buffer": self.buffer,
            "timer_frac": max(0.0, 1.0 - self.turn_steps / max(1, self.budget)),
            "tries_frac": self.tries_left / MAX_TRIES,
            "lives_frac": self.player_lives / MAX_LIVES,
            "score_frac": min(1.0, self.score / 50.0),
            "tries_left": self.tries_left,
            "lives": self.player_lives,
            "bot_lives": self.bot_lives,
            "score": self.score,
            "chain_len": self.chain_len,
            "difficulty": self.difficulty,
            "phase": self.phase,
        }

    def step(self, action: str) -> StepResult:
        if self.phase == "finished":
            return StepResult(self.observe(), 0.0, True, {"reason": "already_finished"})
        action = str(action)
        self.turn_steps += 1
        info = {"action": action, "event": "key"}

        if self.turn_steps > self.budget:
            return self._fail(self.rewards.timeout, "timeout", life=True)

        if action == "NOOP":
            return StepResult(self.observe(), self.rewards.noop, False, info)

        if action == "BACKSPACE":
            if self.buffer:
                self.buffer = self.buffer[:-1]
            info["buffer"] = self.buffer
            return StepResult(self.observe(), self.rewards.backspace, False, info)

        if action in "abcdefghijklmnopqrstuvwxyz":
            self.buffer += action
            reward = 0.0
            if self.letter_shaping:
                if is_valid_prefix_of_playable(self.dict_, self.buffer, self.prefix, self.used):
                    reward = self.rewards.correct_letter
                else:
                    reward = self.rewards.wrong_letter
            info["buffer"] = self.buffer
            return StepResult(self.observe(), reward, False, info)

        if action != "ENTER":
            raise ValueError(f"Unknown action {action!r}")

        return self._submit()

    def _submit(self) -> StepResult:
        word = self.buffer.lower()
        self.buffer = ""
        if not is_word_shape(word) or len(word) < 2 or len(word) > MAX_WORD_LEN:
            return self._fail(self.rewards.invalid_word, "shape")
        if not word.startswith(self.prefix) or len(word) <= len(self.prefix):
            return self._fail(self.rewards.invalid_word, "prefix")
        if word in self.used:
            return self._fail(self.rewards.invalid_word, "used")
        if not is_valid_word(self.dict_, word):
            return self._fail(self.rewards.invalid_word, "not_a_word")
        return self._accept(word)

    def _accept(self, word: str) -> StepResult:
        min_groups = CASUAL_MIN_GROUPS if self.difficulty == "casual" else 0
        self.used.add(word)
        self.last_word = word
        self.score += 1
        self.timer_round += 1
        self.tries_left = MAX_TRIES
        self.turn_steps = 0
        self.history.append(("player", word))
        desired = desired_chain_len(self.completed_rounds)
        self.chain_len = resolve_prefix_len(self.dict_, word, desired, self.used, min_groups, 1)
        self.prefix = word[-self.chain_len :]
        reward = self.rewards.valid_word + self.rewards.continue_chain
        info = {"event": "valid_word", "word": word, "prefix": self.prefix}

        bot = opponent_word(
            self.dict_,
            self.prefix,
            self.used,
            self.difficulty,
            next_chain_len=desired_chain_len(self.completed_rounds + 1),
            playstyle=self.playstyle,
            rng=self.rng,
        )
        if not bot:
            # Same rule as the live game: a bot miss only counts when the prefix
            # actually has zero remaining solves (or the bot could not move).
            if playable_count(self.dict_, self.prefix, self.used, 1) == 0:
                self.bot_lives -= 1
                info["bot_miss"] = True
                if self.bot_lives <= 0:
                    self.phase = "finished"
                    self.winner = "player"
                    return StepResult(self.observe(), reward + self.rewards.win, True, {**info, "win": True})
            self.completed_rounds += 1
            self.budget = keystroke_budget(self.timer_round)
            return StepResult(self.observe(), reward, False, info)

        self.used.add(bot)
        self.last_word = bot
        self.history.append(("bot", bot))
        self.completed_rounds += 1
        desired = desired_chain_len(self.completed_rounds)
        self.chain_len = resolve_prefix_len(self.dict_, bot, desired, self.used, min_groups, 1)
        self.prefix = bot[-self.chain_len :]
        self.budget = keystroke_budget(self.timer_round)
        info["bot_word"] = bot
        info["prefix"] = self.prefix
        if playable_count(self.dict_, self.prefix, self.used, 1) == 0:
            # Player is trapped with no legal replies — that is a loss on the
            # next forced miss, but we do not auto-lose until they fail a turn.
            info["trapped"] = True
        return StepResult(self.observe(), reward, False, info)

    def _fail(self, reward: float, reason: str, life: bool = False) -> StepResult:
        info = {"event": "fail", "reason": reason}
        if not life:
            self.tries_left -= 1
            if self.tries_left > 0:
                self.buffer = ""
                return StepResult(self.observe(), reward, False, info)
        self.player_lives -= 1
        self.tries_left = MAX_TRIES
        self.completed_rounds = 0
        self.timer_round = 1
        self.chain_len = 1
        self.prefix = self.last_word[-1:] if self.last_word else self.prefix[-1:]
        self.buffer = ""
        self.turn_steps = 0
        self.budget = keystroke_budget(self.timer_round)
        info["life_lost"] = True
        if self.player_lives <= 0:
            self.phase = "finished"
            self.winner = "bot"
            return StepResult(self.observe(), reward + self.rewards.lose, True, {**info, "lose": True})
        return StepResult(self.observe(), reward, False, info)

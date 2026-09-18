"""Shiritori engine package: real-game rules, no LLM word solver."""

from shiritori.action_decoder import KeyboardDecoder
from shiritori.dictionary import GameDict, load_game_dict
from shiritori.environment import ShiritoriEnv
from shiritori.rewards import RewardTable

__all__ = [
    "GameDict",
    "KeyboardDecoder",
    "RewardTable",
    "ShiritoriEnv",
    "load_game_dict",
]

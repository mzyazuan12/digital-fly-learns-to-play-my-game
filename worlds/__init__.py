"""Environments the same fly can inhabit."""

from worlds.base import Keycap, Light, Odor, Solid, Task, World, empty_arena, stimulus_arena
from worlds.room import living_room, spawn_on_keyboard, spawn_on_rug
from worlds.shiritori_station import ShiritoriWorkstation, workstation

__all__ = [
    "Keycap",
    "Light",
    "Odor",
    "Solid",
    "Task",
    "World",
    "empty_arena",
    "stimulus_arena",
    "living_room",
    "spawn_on_keyboard",
    "spawn_on_rug",
    "ShiritoriWorkstation",
    "workstation",
]

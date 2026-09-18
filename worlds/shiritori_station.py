"""Physical Shiritori workstation. Not a brain method.

The fly that inhabits this room is the same VirtualFly used in an empty
arena. The monitor is a visual surface. Keys are located objects. Nothing
here calls a text-generation function on behalf of the fly.

Intended closed loop (not fully wired in the first-goal slice):

    screen pixels → compound eyes → visual neurons → MaleCNS → motor
    → body walks to a physical key → contact → screen updates
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worlds.base import Light, World


@dataclass
class Key:
    label: str
    x_mm: float
    y_mm: float
    z_mm: float = 0.4


def _qwerty_keys() -> list[Key]:
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    keys: list[Key] = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            keys.append(Key(label=ch, x_mm=4.0 + 4.0 * c, y_mm=-8.0 - 4.0 * r))
    keys.append(Key(label="ENTER", x_mm=44.0, y_mm=-16.0))
    keys.append(Key(label="BACKSPACE", x_mm=48.0, y_mm=-8.0))
    return keys


@dataclass
class ShiritoriWorkstation(World):
    keys: list[Key] = field(default_factory=_qwerty_keys)
    monitor_pixels: object | None = None
    prefix_shown: str = ""

    def __post_init__(self) -> None:
        if not self.lights:
            self.lights = [Light(x_mm=24.0, y_mm=12.0, z_mm=18.0, intensity=20.0)]
        if not self.notes:
            self.notes = (
                "Desk, monitor, physical keyboard. The fly is unchanged. "
                "Do not call a word solver from the brain."
            )

    def set_monitor_frame(self, pixels, prefix: str = "") -> None:
        """Environment may update the screen. This is not a fly API."""
        self.monitor_pixels = pixels
        self.prefix_shown = prefix


def workstation() -> ShiritoriWorkstation:
    return ShiritoriWorkstation(
        name="shiritori_workstation",
        width_mm=800.0,
        depth_mm=600.0,
        height_mm=400.0,
    )

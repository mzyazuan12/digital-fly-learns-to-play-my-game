"""A furnished living room. Not a task, not a maze, not Shiritori.

Scale is household millimetres. The fly is a few millimetres long.
Collision uses simple AABBs; visual detail can be richer later.
"""

from __future__ import annotations

from worlds.base import Keycap, Light, Odor, Solid, World


# ~5 m × 4 m × 2.8 m
ROOM_W = 5000.0
ROOM_D = 4000.0
ROOM_H = 2800.0

# Matches sim3d/js/living_room.js DESK_X / DESK_Z (1.8 m × 0.8 m desk).
DESK_X0 = 2850.0
DESK_Y0 = 80.0
DESK_X1 = 4650.0
DESK_Y1 = 880.0
DESK_Z1 = 740.0

# Matches sim3d/js/keyboard.js ANSI 60% on the desk (board at local z=0.14 m).
U_MM = 19.05
COLS = 15
ROWS = 5
DESK_CX_MM = 3750.0
DESK_CY_MM = 480.0
KEYBOARD_LOCAL_Z_MM = 140.0

# Physics walks on z≈0. Contact z stays on that plane; the renderer plants
# tarsi on the desk keycaps (~811 mm visual Y).
KEY_CONTACT_Z_MM = 1.0

# (label, xu, row, wu) — same units as sim3d/js/keyboard.js LAYOUT.
# Letter labels are lowercase so tarsus contact matches live typing.
_ANSI60 = [
    ("ESC", 0.0, 0, 1.0),
    ("1", 1.0, 0, 1.0),
    ("2", 2.0, 0, 1.0),
    ("3", 3.0, 0, 1.0),
    ("4", 4.0, 0, 1.0),
    ("5", 5.0, 0, 1.0),
    ("6", 6.0, 0, 1.0),
    ("7", 7.0, 0, 1.0),
    ("8", 8.0, 0, 1.0),
    ("9", 9.0, 0, 1.0),
    ("0", 10.0, 0, 1.0),
    ("-", 11.0, 0, 1.0),
    ("=", 12.0, 0, 1.0),
    ("BACKSPACE", 13.0, 0, 2.0),
    ("TAB", 0.0, 1, 1.5),
    ("q", 1.5, 1, 1.0),
    ("w", 2.5, 1, 1.0),
    ("e", 3.5, 1, 1.0),
    ("r", 4.5, 1, 1.0),
    ("t", 5.5, 1, 1.0),
    ("y", 6.5, 1, 1.0),
    ("u", 7.5, 1, 1.0),
    ("i", 8.5, 1, 1.0),
    ("o", 9.5, 1, 1.0),
    ("p", 10.5, 1, 1.0),
    ("[", 11.5, 1, 1.0),
    ("]", 12.5, 1, 1.0),
    ("\\", 13.5, 1, 1.5),
    ("CAPS", 0.0, 2, 1.75),
    ("a", 1.75, 2, 1.0),
    ("s", 2.75, 2, 1.0),
    ("d", 3.75, 2, 1.0),
    ("f", 4.75, 2, 1.0),
    ("g", 5.75, 2, 1.0),
    ("h", 6.75, 2, 1.0),
    ("j", 7.75, 2, 1.0),
    ("k", 8.75, 2, 1.0),
    ("l", 9.75, 2, 1.0),
    (";", 10.75, 2, 1.0),
    ("'", 11.75, 2, 1.0),
    ("ENTER", 12.75, 2, 2.25),
    ("LSHIFT", 0.0, 3, 2.25),
    ("z", 2.25, 3, 1.0),
    ("x", 3.25, 3, 1.0),
    ("c", 4.25, 3, 1.0),
    ("v", 5.25, 3, 1.0),
    ("b", 6.25, 3, 1.0),
    ("n", 7.25, 3, 1.0),
    ("m", 8.25, 3, 1.0),
    (",", 9.25, 3, 1.0),
    (".", 10.25, 3, 1.0),
    ("/", 11.25, 3, 1.0),
    ("RSHIFT", 12.25, 3, 2.75),
    ("LCTRL", 0.0, 4, 1.25),
    ("LWIN", 1.25, 4, 1.25),
    ("LALT", 2.5, 4, 1.25),
    ("SPACE", 3.75, 4, 6.25),
    ("RALT", 10.0, 4, 1.25),
    ("FN", 11.25, 4, 1.25),
    ("MENU", 12.5, 4, 1.25),
    ("RCTRL", 13.75, 4, 1.25),
]


def _key_xy(xu: float, row: int, wu: float) -> tuple[float, float]:
    inner_w = COLS * U_MM
    inner_d = ROWS * U_MM
    local_x = -inner_w / 2.0 + (xu + wu / 2.0) * U_MM
    local_z = -inner_d / 2.0 + (row + 0.5) * U_MM
    return DESK_CX_MM + local_x, DESK_CY_MM + KEYBOARD_LOCAL_Z_MM + local_z


def desk_keyboard_keys() -> list[Keycap]:
    """ANSI 60% on the desk, in front of the monitor. Not on the rug."""
    keys: list[Keycap] = []
    for label, xu, row, wu in _ANSI60:
        x_mm, y_mm = _key_xy(xu, row, wu)
        keys.append(
            Keycap(
                label=label,
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=KEY_CONTACT_Z_MM,
                half_mm=max(6.0, wu * U_MM / 2.0 - 0.8),
            )
        )
    return keys


def living_room() -> World:
    solids = [
        Solid("desk", DESK_X0, DESK_Y0, DESK_X1, DESK_Y1, 0, DESK_Z1, walkable=True),
        Solid("couch", 200, 2700, 1600, 3850, 0, 450),
        Solid("coffee_table", 1700, 1500, 2500, 2300, 0, 400),
        Solid("side_table", 200, 200, 700, 700, 0, 500),
        Solid("plant_pot", 4600, 3400, 4850, 3700, 0, 900),
        Solid("trash_can", 4700, 200, 4900, 450, 0, 400),
        Solid("bookshelf", 50, 1200, 220, 2400, 0, 1800),
        Solid("tv_stand", 1800, 50, 3200, 280, 0, 450),
    ]
    lights = [
        Light(900, 900, 1700, intensity=90.0),       # floor lamp
        Light(2500, 2000, 2700, intensity=40.0),     # ceiling
        Light(4800, 3800, 1400, intensity=55.0),     # window
        Light(200, 3800, 1400, intensity=55.0),      # other window
    ]
    odors = [
        Odor(2100, 1900, 420, intensity=0.9, sigma_mm=500.0),  # fruit bowl on coffee table
        Odor(3750, 520, 760, intensity=0.25, sigma_mm=250.0),  # something on the desk
    ]
    monitor = {"x_mm": 3750.0, "y_mm": 200.0, "z_mm": 1280.0, "intensity": 220.0, "w_mm": 920.0, "h_mm": 518.0}
    return World(
        name="living_room",
        width_mm=ROOM_W,
        depth_mm=ROOM_D,
        height_mm=ROOM_H,
        lights=lights,
        odors=odors,
        solids=solids,
        keys=desk_keyboard_keys(),
        monitor=monitor,
        notes=(
            "Furnished room. Monitor and ANSI 60% keyboard share the desk. "
            "The fly types by standing on those keycaps. No keyboard on the rug."
        ),
    )


def spawn_on_keyboard() -> tuple[float, float, float]:
    x_mm, y_mm = _key_xy(5.75, 2, 1.0)  # G
    return x_mm, y_mm, 0.6


def spawn_on_rug() -> tuple[float, float, float]:
    # Living-room spawn. The fly starts on the desk keyboard, not the rug.
    return spawn_on_keyboard()

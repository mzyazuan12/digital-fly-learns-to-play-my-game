"""Engineered BCI populations on top of MaleCNS annotations.

These groupings are experimental interfaces, not claimed natural functions.
Sensory cells receive artificial current. Descending cells are read out as
keys. The connectome in between is the real wiring graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.loader import Connectome

LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
ACTIONS = LETTERS + ("BACKSPACE", "ENTER", "NOOP")
N_PREFIX_SLOTS = 4
N_WORD_SLOTS = 8


def _partition(indices: np.ndarray, n_groups: int, per_group: int) -> list[np.ndarray]:
    if indices.size == 0:
        raise ValueError("Need at least one neuron to build populations")
    # Real MaleCNS sensory pools are large enough that each group is unique.
    # Tiny synthetic graphs wrap; that is a test convenience, not biology.
    per = max(1, int(per_group))
    if indices.size >= n_groups * per:
        use = indices[: n_groups * per]
        return [np.asarray(use[g * per : (g + 1) * per], dtype=np.int32) for g in range(n_groups)]
    groups = []
    pos = 0
    width = 1 if indices.size < n_groups else max(1, indices.size // n_groups)
    for _ in range(n_groups):
        take = np.empty(width, dtype=np.int32)
        for k in range(width):
            take[k] = indices[pos % indices.size]
            pos += 1
        groups.append(take)
    return groups


@dataclass
class InterfaceMap:
    """Fixed neuron-index bundles used by the encoder and decoder."""

    letter_prefix: list[np.ndarray]  # 4 * 26
    letter_previous: list[np.ndarray]  # 8 * 26
    letter_buffer: list[np.ndarray]  # 8 * 26
    timer: np.ndarray
    tries: np.ndarray
    lives: np.ndarray
    score: np.ndarray
    reward: np.ndarray
    action_groups: list[np.ndarray]  # 29
    sensory_indices: np.ndarray
    descending_indices: np.ndarray
    notes: dict


def build_interface(connectome: Connectome, seed: int = 0) -> InterfaceMap:
    """Assign annotation-defined cells to the keyboard BCI.

    Preference order for inputs: named sensory superclasses, then any remaining
    non-descending cells. Outputs: descending_neuron*, then motor, then the
    last neurons in the ID order as a last-resort documented fallback.
    """
    rng = np.random.default_rng(seed)
    sensory_super = (
        "ol_sensory",
        "cb_sensory",
        "vnc_sensory",
        "sensory_ascending",
        "cb_sensory_tbc",
        "vnc_sensory_tbc",
        "sensory_descending",
        "sensory_ascending_tbc",
    )
    sensory = np.concatenate(
        [connectome.lookup(superclass=name) for name in sensory_super]
    )
    if sensory.size == 0:
        sensory = np.arange(min(4096, connectome.n), dtype=np.int32)
        fallback_in = "first_n_neurons_no_sensory_annotation"
    else:
        fallback_in = "annotated_sensory_superclasses"

    descending = connectome.lookup(superclass="descending_neuron")
    extra_dn = connectome.lookup(superclass="descending_neuron_tbc")
    motor = np.concatenate(
        [
            connectome.lookup(superclass="vnc_motor"),
            connectome.lookup(superclass="cb_motor"),
        ]
    )
    outputs = np.unique(np.concatenate([descending, extra_dn, motor]))
    if outputs.size < len(ACTIONS):
        outputs = np.arange(max(0, connectome.n - 2048), connectome.n, dtype=np.int32)
        fallback_out = "last_n_neurons_insufficient_DNs"
    else:
        fallback_out = "descending_plus_motor"

    # Stable shuffle so the mapping is reproducible but not ID-ordered trivially.
    sensory = rng.permutation(sensory)
    outputs = rng.permutation(outputs)

    n_letter_channels = (N_PREFIX_SLOTS + 2 * N_WORD_SLOTS) * 26
    analog = 4  # timer, tries, lives, score
    n_in_groups = n_letter_channels + analog + 1  # + reward pulse
    per_in = max(4, sensory.size // n_in_groups)
    in_groups = _partition(sensory, n_in_groups, per_in)

    cursor = 0

    def take(k: int) -> list[np.ndarray]:
        nonlocal cursor
        chunk = in_groups[cursor : cursor + k]
        cursor += k
        return chunk

    letter_prefix = take(N_PREFIX_SLOTS * 26)
    letter_previous = take(N_WORD_SLOTS * 26)
    letter_buffer = take(N_WORD_SLOTS * 26)
    timer, tries, lives, score, reward = (take(1)[0] for _ in range(5))

    per_out = max(4, outputs.size // len(ACTIONS))
    action_groups = _partition(outputs, len(ACTIONS), per_out)

    notes = {
        "input_source": fallback_in,
        "output_source": fallback_out,
        "n_sensory": int(sensory.size),
        "n_outputs": int(outputs.size),
        "neurons_per_input_group": int(per_in),
        "neurons_per_action_group": int(per_out),
        "engineered_bci": True,
        "not_a_biological_keyboard": True,
    }
    return InterfaceMap(
        letter_prefix=letter_prefix,
        letter_previous=letter_previous,
        letter_buffer=letter_buffer,
        timer=timer,
        tries=tries,
        lives=lives,
        score=score,
        reward=reward,
        action_groups=action_groups,
        sensory_indices=sensory.astype(np.int32),
        descending_indices=outputs.astype(np.int32),
        notes=notes,
    )

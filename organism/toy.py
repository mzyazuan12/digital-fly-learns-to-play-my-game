"""Miniature annotation-faithful graph for tests. Not MaleCNS.

Cells carry MaleCNS-like type names so MotorBridge, mushroom-body learning,
and the motor map resolve the same pathways as on the real connectome.

There is no phototaxis wiring, no contact-avoidance wiring, and no
anonymous index-range action map. Extra edges are random, not a
behavior controller.
"""

from __future__ import annotations

import numpy as np

from flybrain.loader import Connectome
from flybrain.neurons import nt_sign


def _coo_to_csr(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int):
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = post[order]
    weight = weight[order]
    counts = np.bincount(pre, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts, dtype=np.int64)
    return ptr, post, weight


N = 104


def miniature_connectome(seed: int = 0) -> Connectome:
    rng = np.random.default_rng(seed)
    n = N
    extra = 120
    pre = rng.integers(0, n, size=extra).astype(np.uint32)
    post = rng.integers(0, n, size=extra).astype(np.uint32)
    weight = rng.integers(1, 12, size=extra).astype(np.uint32)
    # Local recurrence on identified DNs so tonic current can elicit spikes.
    dn = np.array([32, 33, 34, 35, 36, 37, 38, 39], dtype=np.uint32)
    pre = np.concatenate([pre, dn, dn])
    post = np.concatenate([post, dn, np.roll(dn, 1)])
    weight = np.concatenate([weight, np.full(dn.size * 2, 24, dtype=np.uint32)])
    # KC → MBON (plastic in mushroom-body learning). Not a motor shortcut.
    kc = np.arange(60, 68, dtype=np.uint32)
    mbon = np.array([70, 71], dtype=np.uint32)
    pre = np.concatenate([pre, np.repeat(kc, mbon.size)])
    post = np.concatenate([post, np.tile(mbon, kc.size)])
    weight = np.concatenate([weight, np.full(kc.size * mbon.size, 8, dtype=np.uint32)])
    # Olfactory/chemo → KC. Sensory representation, not a taxis controller.
    chemo = np.arange(40, 48, dtype=np.uint32)
    pre = np.concatenate([pre, np.repeat(chemo, 2)])
    post = np.concatenate([post, np.tile(kc[:2], chemo.size)])
    weight = np.concatenate([weight, np.full(chemo.size * 2, 6, dtype=np.uint32)])
    # DNp09 → VNC premotor. Identified descending path, not a CPG substitute.
    premotor = np.arange(72, 80, dtype=np.uint32)
    pre = np.concatenate([pre, np.array([32, 33, 32, 33], dtype=np.uint32)])
    post = np.concatenate([post, np.array([72, 73, 74, 75], dtype=np.uint32)])
    weight = np.concatenate([weight, np.full(4, 10, dtype=np.uint32)])
    # Graded premotor → motor neurons.
    motor = np.arange(80, 88, dtype=np.uint32)
    pre = np.concatenate([pre, premotor])
    post = np.concatenate([post, motor])
    weight = np.concatenate([weight, np.full(premotor.size, 12, dtype=np.uint32)])
    # Published walking CPG (Pugliese 2025), miniature and labeled — not MaleCNS.
    # DNg100 → E1; E1↔E2; E1/E2→I1; I1⊣E1/E2; E1/E2→MN. DNp09 can recruit DNg100.
    cpg_pre = np.array(
        [
            32, 33,  # DNp09 → DNg100
            88, 88, 89, 89,  # DNg100 → E1, E2
            93, 94,  # E1 ↔ E2
            93, 94,  # E1/E2 → I1
            95, 95,  # I1 ⊣ E1, E2
            93, 93, 94, 94,  # CPG → MN
            90,  # oDN1 → E1
            91,  # DNb08 weakly → E1
            92,  # Bluebell ⊣ oDN1
            88, 89, 88, 89,  # DNg100 recurrence
        ],
        dtype=np.uint32,
    )
    cpg_post = np.array(
        [
            88, 89,
            93, 94, 93, 94,
            94, 93,
            95, 95,
            93, 94,
            80, 81, 82, 83,
            93,
            93,
            90,
            88, 89, 89, 88,
        ],
        dtype=np.uint32,
    )
    cpg_w = np.array(
        [16, 16, 20, 20, 20, 20, 18, 18, 12, 12, 22, 22, 14, 14, 14, 14, 10, 4, 16, 24, 24, 24, 24],
        dtype=np.uint32,
    )
    pre = np.concatenate([pre, cpg_pre])
    post = np.concatenate([post, cpg_post])
    weight = np.concatenate([weight, cpg_w])

    ptr, post, weight = _coo_to_csr(pre, post, weight, n)

    cell_type = np.array([f"toy{i}" for i in range(n)], dtype=object)
    superclasses = np.empty(n, dtype=object)
    sides = np.array([""] * n, dtype=object)
    cell_class = np.array([""] * n, dtype=object)
    superclasses[0:8] = "ol_sensory"
    superclasses[8:16] = "ol_sensory"
    superclasses[16:32] = "vnc_sensory"
    superclasses[32:40] = "descending_neuron"
    superclasses[40:56] = "cb_sensory"
    superclasses[56:60] = "descending_neuron"
    superclasses[60:68] = "kenyon_cell"
    superclasses[68:70] = "dopaminergic"
    superclasses[70:72] = "mbon"
    superclasses[72:80] = "vnc_premotor"
    superclasses[80:88] = "vnc_motor"
    superclasses[88:104] = "cb_intrinsic"
    sides[0:8] = "L"
    sides[8:16] = "R"
    sides[16:32] = np.array(["L", "R"] * 8)
    cell_type[32] = "DNp09"
    cell_type[33] = "DNp09"
    sides[32], sides[33] = "L", "R"
    cell_type[34] = "DNa02"
    cell_type[35] = "DNa02"
    sides[34], sides[35] = "L", "R"
    cell_type[36] = "DNa01"
    cell_type[37] = "DNa01"
    sides[36], sides[37] = "L", "R"
    cell_type[38] = "DNg13"
    cell_type[39] = "DNg13"
    sides[38], sides[39] = "L", "R"
    sides[40:56] = np.array(["L", "R"] * 8)
    cell_type[56] = "DNg02"
    cell_type[57] = "DNg02"
    sides[56], sides[57] = "L", "R"
    cell_type[58] = "aDN1"
    cell_type[59] = "aDN1"
    sides[58], sides[59] = "L", "R"
    for i, idx in enumerate(range(60, 68)):
        cell_type[idx] = "KC"
        sides[idx] = "L" if i % 2 == 0 else "R"
    cell_type[68] = "PAM"
    cell_type[69] = "PPL1"
    superclasses[68:70] = "dopaminergic"
    sides[68], sides[69] = "L", "R"
    cell_type[70] = "MBON01"
    cell_type[71] = "MBON02"
    sides[70], sides[71] = "L", "R"
    for i, idx in enumerate(range(72, 80)):
        cell_type[idx] = "premotorIN"
        cell_class[idx] = "premotor"
        sides[idx] = "L" if i < 4 else "R"
    for i, idx in enumerate(range(80, 88)):
        cell_type[idx] = "MN"
        cell_class[idx] = "motor"
        sides[idx] = "L" if i < 4 else "R"
    cell_class[0:32] = "sensory"
    cell_type[88] = "DNg100"
    cell_type[89] = "DNg100"
    sides[88], sides[89] = "L", "R"
    superclasses[88:90] = "descending_neuron"
    cell_type[90] = "DNg97"
    sides[90] = "L"
    superclasses[90] = "descending_neuron"
    cell_type[91] = "DNb08"
    sides[91] = "L"
    superclasses[91] = "descending_neuron"
    cell_type[92] = "DNg60"
    sides[92] = "L"
    superclasses[92] = "descending_neuron"
    cell_type[93] = "IN17A001"
    cell_type[94] = "INXXX466"
    cell_type[95] = "IN16B036"
    superclasses[93:96] = "vnc_intrinsic"
    cell_class[93:96] = "premotor"
    sides[93], sides[94], sides[95] = "L", "L", "L"
    cell_type[96] = "AN19A018"
    cell_type[97] = "AN19A018"
    superclasses[96:98] = "ascending_neuron"
    sides[96], sides[97] = "L", "R"

    transmitters = np.array(["acetylcholine"] * n, dtype=object)
    transmitters[68:70] = "dopamine"
    transmitters[92] = "gaba"
    transmitters[95] = "gaba"
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    sign = np.empty(len(post), dtype=np.int8)
    for i in range(n):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    return Connectome(
        neuron_ids=np.arange(1, n + 1, dtype=np.uint64),
        pre_ptr=ptr,
        post=post,
        anatomical=weight,
        sign=sign,
        superclass=superclasses,
        cell_type=cell_type,
        cell_class=cell_class,
        side=sides,
        neurotransmitter=transmitters,
        report={
            "dataset_id": "miniature",
            "retained_neuron_candidates": n,
            "neural_dynamics_validated": False,
            "learning_demonstrated": False,
            "toy_phototaxis_wiring": False,
            "builder_seed": int(seed),
            "note": "Type labels mimic MaleCNS so tests hit MotorBridge/MB/VNC; wiring is not a behavior.",
        },
    )


locomotor_connectome = miniature_connectome

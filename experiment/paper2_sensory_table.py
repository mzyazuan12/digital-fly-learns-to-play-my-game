"""Stage 2B step 2: which sensory neurons feed back from which front leg.

Rule (design §4.1-4.2): leg = rootSide + MaleCNS entryNerve. Included
in 2B: front-leg hair plates (VProN SNpp45, DProN SNpp52, ProLN SNpp45),
front-leg femoral chordotonal neurons (ProLN chordotonal organ), and
front-leg campaniform sensilla (ProLN SNpp53) except 1078099767, whose
motor-neuron synapses all go to the contralateral leg. Excluded: neck
(PrN) hair plates, bristles, 'leg'-subclass afferents of unknown modality,
middle/hind-leg afferents. Each included afferent is checked against the
side of the motor neurons it contacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "paper2" / "stage2B_sensory_table.json"
FRONT_LEG_NERVES = {"ProLN", "VProN", "DProN", "ProAN"}
EXCLUDED_AMBIGUOUS = {1078099767}


def channel_of(subclass: str, nerve: str, typ: str) -> str | None:
    if subclass == "hair plate" and nerve == "VProN" and typ == "SNpp45":
        return "hair_plate_coxa"
    if subclass == "hair plate" and nerve == "DProN" and typ == "SNpp52":
        return "hair_plate_trochanter"
    if subclass == "hair plate" and nerve == "ProLN" and typ == "SNpp45":
        return "hair_plate_coxa"
    if subclass == "chordotonal organ" and nerve == "ProLN":
        return "chordotonal_femur_tibia"
    if subclass == "campaniform sensilla" and nerve == "ProLN" and typ == "SNpp53":
        return "campaniform_trochanter"
    return None


def build() -> dict:
    from flybrain.authors_mcns import load_authors_W, load_authors_wtable

    t = load_authors_wtable()
    ann = pd.read_feather(
        ROOT / "body-annotations-male-cns-v1.0-minconf-0.5.feather", columns=["bodyId", "entryNerve", "type"]
    ).rename(columns={"type": "mcns_type"})
    sens = t[t["class"].astype(str).str.contains("sensory")].copy()
    sens["row"] = sens.index
    sens = sens.merge(ann, on="bodyId", how="left")
    W, _, _ = load_authors_W()
    mn = t["motor module"].notna().to_numpy()
    mn_side = t["somaSide"].astype(str).to_numpy()
    rows = []
    for _, s in sens.iterrows():
        i = int(s["row"])
        out_w = W[i]
        to_mn = np.flatnonzero(mn & (out_w != 0))
        n_l = int(np.abs(out_w[to_mn][mn_side[to_mn] == "L"]).sum())
        n_r = int(np.abs(out_w[to_mn][mn_side[to_mn] == "R"]).sum())
        side = str(s["rootSide"]) if str(s["rootSide"]) in ("L", "R") else None
        nerve = str(s["entryNerve"]) if pd.notna(s["entryNerve"]) else None
        ch = channel_of(str(s["subclass"]), nerve or "", str(s["type"]))
        total = n_l + n_r
        ipsi = (n_l if side == "L" else n_r) / total if (side and total) else None
        include = bool(
            ch is not None
            and side is not None
            and nerve in FRONT_LEG_NERVES
            and int(s["bodyId"]) not in EXCLUDED_AMBIGUOUS
            and (total < 10 or (ipsi is not None and ipsi >= 0.8))
        )
        rows.append(
            {
                "row": i,
                "bodyId": int(s["bodyId"]),
                "type": str(s["type"]),
                "subclass": str(s["subclass"]),
                "entryNerve": nerve,
                "rootSide": side,
                "leg": f"T1{side}" if (side and nerve in FRONT_LEG_NERVES) else None,
                "channel": ch,
                "mn_synapses_L": n_l,
                "mn_synapses_R": n_r,
                "ipsilateral_fraction": ipsi,
                "normalized_size": float(s["size"] / np.nanmedian(t["size"])),
                "included_2B": include,
            }
        )
    inc = [r for r in rows if r["included_2B"]]
    counts = {}
    for r in inc:
        k = f"{r['leg']}:{r['channel']}"
        counts[k] = counts.get(k, 0) + 1
    return {
        "rule": "leg = rootSide + MaleCNS entryNerve; ipsilateral MN-synapse fraction >= 0.8 when >= 10 MN synapses",
        "design": "docs/papers/paper2_stage2B_design.md §4.1-4.2",
        "excluded_ambiguous": sorted(EXCLUDED_AMBIGUOUS),
        "n_sensory_total": len(rows),
        "n_included": len(inc),
        "included_counts": dict(sorted(counts.items())),
        "afferents": rows,
    }


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(ROOT))
    table = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, indent=1) + "\n")
    print(OUT)
    print(json.dumps({k: table[k] for k in ("n_sensory_total", "n_included", "included_counts")}, indent=1))

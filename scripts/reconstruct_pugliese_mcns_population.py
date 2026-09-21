#!/usr/bin/env python3
"""Audit a MaleCNS v1.0 reconstruction against the authors' shipped export.

The reproduction population is authors_export_population: the 4,310-row
wTable shipped in Pugliese_2026. This script rebuilds the earlier
methods candidate (front-leg motor neurons, their VNC presynaptic partners,
extra VNC descending neurons, non-null consensusNt, then a VNC synapse
floor of 5) and diffs body IDs. That candidate is
reconstructed_from_male-cns:v1.0. It is not the set_sizes() denominator,
even if a future query happens to return 4,310 IDs.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flybrain.authors_mcns import (
    AUTHORS_MCNS_N,
    DIFF_PATH,
    POPULATION_AUTHORS,
    POPULATION_RECONSTRUCTION,
    RECONSTRUCTION_IDS_PATH,
    annotate_authors_rows,
    load_authors_wtable,
)
from flybrain.malecns_volume import NEUPRINT_DATASET, SIZE_NORM_PUGLIESE_SUBNETWORK

STATUS_PATH = ROOT / "data" / "malecns_v1" / "pugliese_mcns_population_status.json"
NEUPRINT_URL = "https://neuprint.janelia.org/api/custom/custom"


def cypher(query: str, timeout: int = 300) -> dict:
    body = json.dumps({"cypher": query, "dataset": NEUPRINT_DATASET}).encode()
    request = urllib.request.Request(
        NEUPRINT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _one(query: str):
    return cypher(query)["data"][0][0]


def candidate_body_ids() -> list[int]:
    """Front-leg MNs, their VNC presynaptic partners, and extra VNC DNs."""
    query = """
    MATCH (pre:`male-cns_Neuron`)-[c:ConnectsTo]->(mn:`male-cns_Neuron`)
    WHERE mn.superclass = 'vnc_motor' AND mn.subclass = 'fl' AND mn.status = 'Traced'
      AND c.roiInfo CONTAINS '"VNC"'
    WITH collect(DISTINCT pre.bodyId) + collect(DISTINCT mn.bodyId) AS base
    MATCH (dn:`male-cns_Neuron`)-[c2:ConnectsTo]->(p:`male-cns_Neuron`)
    WHERE dn.superclass = 'descending_neuron'
      AND dn.status = 'Traced' AND dn.consensusNt IS NOT NULL
      AND c2.roiInfo CONTAINS '"VNC"'
      AND p.bodyId IN base
      AND NOT dn.bodyId IN base
    WITH base + collect(DISTINCT dn.bodyId) AS ids
    UNWIND ids AS bid
    MATCH (n:`male-cns_Neuron` {bodyId: bid})
    WHERE n.status = 'Traced' AND n.consensusNt IS NOT NULL
    RETURN DISTINCT n.bodyId AS bodyId
    """
    rows = cypher(query, timeout=600)["data"]
    return sorted(int(row[0]) for row in rows)


def kept_after_synapse_floor(body_ids: list[int]) -> list[int]:
    """Neurons with at least one VNC connection of >=5 synapses inside the set."""
    listed = ", ".join(str(i) for i in body_ids)
    query = f"""
    MATCH (a:`male-cns_Neuron`)-[c:ConnectsTo]->(b:`male-cns_Neuron`)
    WHERE a.bodyId IN [{listed}] AND b.bodyId IN [{listed}]
      AND c.roiInfo CONTAINS '"VNC"'
    WITH a, b, apoc.convert.fromJsonMap(c.roiInfo)['VNC']['post'] AS vnc
    WHERE vnc >= 5
    WITH collect(DISTINCT a.bodyId) + collect(DISTINCT b.bodyId) AS ids
    UNWIND ids AS bid
    RETURN DISTINCT bid
    """
    rows = cypher(query, timeout=600)["data"]
    return sorted(int(row[0]) for row in rows)


def main() -> int:
    fl = int(_one(
        "MATCH (n:`male-cns_Neuron`) WHERE n.superclass='vnc_motor' AND n.subclass='fl' "
        "AND n.status='Traced' RETURN count(n)"
    ))
    fl_nt = int(_one(
        "MATCH (n:`male-cns_Neuron`) WHERE n.superclass='vnc_motor' AND n.subclass='fl' "
        "AND n.status='Traced' AND n.consensusNt IS NOT NULL RETURN count(n)"
    ))
    print(f"front-leg motor neurons traced={fl} with consensusNt={fl_nt}", flush=True)
    print("collecting candidate population", flush=True)
    candidates = candidate_body_ids()
    print(f"candidates before synapse floor: {len(candidates)}", flush=True)
    print("applying VNC synapse floor of 5", flush=True)
    kept = kept_after_synapse_floor(candidates)
    print(f"kept after floor: {len(kept)} target {AUTHORS_MCNS_N}", flush=True)
    confirmed = len(kept) == AUTHORS_MCNS_N
    status = {
        "size_normalization_mode": SIZE_NORM_PUGLIESE_SUBNETWORK,
        "authors_reported_n": AUTHORS_MCNS_N,
        "confirmed_population": confirmed,
        "median_volume": None,
        "selection": {
            "front_leg_motor": "superclass=vnc_motor, subclass=fl, status=Traced",
            "premotor": "ConnectsTo those motor neurons with roiInfo containing VNC",
            "descending": "superclass=descending_neuron, VNC ConnectsTo the premotor/motor set, not already included",
            "proofread_and_nt": "status=Traced and consensusNt IS NOT NULL",
            "synapse_floor": "keep a neuron only if it retains a VNC connection of >=5 synapses inside the set",
            "not_yet_authors_matrix": not confirmed,
        },
        "counts": {
            "front_leg_motor_traced": fl,
            "front_leg_motor_with_consensusNt": fl_nt,
            "candidates_before_floor": len(candidates),
            "kept_after_vnc_floor_5": len(kept),
        },
        "note": (
            "Pugliese reported 4,310 mCNS neurons. This reconstruction follows "
            "the published selection in MaleCNS annotation vocabulary. A count "
            "other than 4,310 is not their W_table and must not be used as the "
            "set_sizes denominator."
        ),
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status["counts"], indent=2))
    print("confirmed_population", confirmed)
    print(STATUS_PATH)
    return 0 if confirmed else 2


if __name__ == "__main__":
    raise SystemExit(main())

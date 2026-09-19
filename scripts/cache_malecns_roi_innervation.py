#!/usr/bin/env python3
"""Cache per-bodyId MaleCNS LegNp innervation from local syn-points.

Does not query neuPrint. Does not average ROI totals across a cell type.
Does not treat MANC T1 matrix indices as MaleCNS body IDs.

Pass --from-syn-points to scan the 12.7 GB table with column projection.
Pass --write-mapping to emit cpg_mapping.json after the parquet exists.
The previous mapping is archived as cpg_mapping.INVALID_pre_roi.json
instead of being silently overwritten.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism.roi_innervation import (
    INVALID_MAPPING_PATH,
    JSON_CACHE,
    MAPPING_PATH,
    METADATA_PATH,
    PARQUET_CACHE,
    PUGLIESE_DNG100_STIM,
    archive_invalid_mapping,
    build_cpg_mapping,
    load_annotations,
    parquet_from_json_and_annotations,
    scan_syn_points,
    verify_pugliese_manc_t1_dng100,
    write_cpg_mapping,
    write_neuron_metadata,
    write_roi_parquet,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-syn-points", action="store_true")
    parser.add_argument("--write-mapping", action="store_true", help="write cpg_mapping.json")
    parser.add_argument("--metadata", action="store_true", help="also write neuron_metadata.parquet from graph.npz")
    args = parser.parse_args(argv)

    checked = verify_pugliese_manc_t1_dng100()
    stim = checked["pugliese_reference"]
    print(
        "Pugliese DNg100_Stim:",
        f"dataset={stim['dataset']}",
        f"matrix_index={stim['matrix_index']}",
        f"body_id={stim['body_id']}",
        f"type={stim['type']}",
    )
    vms = checked["manc_vms16"]
    print(
        "MANC table also has",
        f"matrix_index={vms['matrix_index']}",
        f"body_id={vms['body_id']}",
        f"type={vms['type']}",
        "(not DNg100; same integer may exist independently in MaleCNS)",
    )

    if args.from_syn_points:
        anns = load_annotations()
        bodies = {int(row["malecns_body_id"]) for row in anns}
        print(f"scanning syn-points for {len(bodies)} CPG/DNg100 MaleCNS body IDs")
        rois = scan_syn_points(bodies)
        JSON_CACHE.parent.mkdir(parents=True, exist_ok=True)
        JSON_CACHE.write_text(
            json.dumps({"bodies": {str(k): v for k, v in rois.items()}, "n_bodies": len(rois)}, indent=2)
        )
        print(f"wrote {JSON_CACHE}")

    table = parquet_from_json_and_annotations()
    write_roi_parquet(table)
    print(f"wrote {PARQUET_CACHE} rows={table.num_rows}")
    if args.write_mapping:
        archived = archive_invalid_mapping(MAPPING_PATH)
        if archived:
            print(f"archived previous mapping at {INVALID_MAPPING_PATH}")
        mapping = build_cpg_mapping(table.to_pylist())
        write_cpg_mapping(mapping)
        print(f"wrote {MAPPING_PATH}")
        e1 = mapping.get("IN17A001") or {}
        print(f"E1 assigned {e1.get('n_assigned')}/{e1.get('n_expected')}")
        i2 = mapping.get("IN19B007") or {}
        print(
            f"I2 IN19B007 assigned {i2.get('n_assigned')}/{i2.get('n_expected')} "
            f"unassigned={len(i2.get('unassigned') or [])}"
        )
        malecns = (mapping.get("DNg100") or {}).get("malecns") or {}
        left = malecns.get("left") or {}
        right = malecns.get("right") or {}
        print(
            "MaleCNS DNg100 from annotations[type==DNg100]: "
            f"left body_id={left.get('body_id')} side={left.get('side')} "
            f"right body_id={right.get('body_id')} side={right.get('side')}"
        )
        print(
            "Pugliese reference remains "
            f"matrix_index={PUGLIESE_DNG100_STIM['matrix_index']} "
            f"body_id={PUGLIESE_DNG100_STIM['body_id']} "
            f"type={PUGLIESE_DNG100_STIM['type']}"
        )
    else:
        print("skipping cpg_mapping.json (--write-mapping not set)")
    if args.metadata:
        from flybrain.loader import load_connectome

        graph = load_connectome(progress=True)
        write_neuron_metadata(graph)
        print(f"wrote {METADATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

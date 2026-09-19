#!/usr/bin/env python3
"""Cache per-bodyId MaleCNS LegNp innervation from local syn-points.

Identity comes only from:

  body-annotations-male-cns-v1.0-minconf-0.5.feather
  column bodyId  (renamed malecns_body_id)

ROI comes from:

  syn-points-male-cns-v1.0-minconf-0.5.feather
  column body     (renamed malecns_body_id)

The derived parquet is an output, never an identity source.
Pugliese MANC IDs belong only under DNg100.pugliese_reference.

Pass --from-syn-points to scan the 12.7 GB table with column projection.
Pass --write-mapping to emit cpg_mapping.json.
Previous mappings are archived as INVALID_* files, not overwritten.
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
    CORE_CPG_TYPES,
    JSON_CACHE,
    MAPPING_PATH,
    METADATA_PATH,
    PARQUET_CACHE,
    PUGLIESE_DNG100_STIM,
    archive_invalid_mapping,
    archive_invalid_namespace_artifacts,
    build_cpg_mapping,
    load_annotations,
    load_raw_annotation_table,
    malecns_dng100_body_ids,
    parquet_from_json_and_annotations,
    raw_dng100_annotation_rows,
    scan_syn_points,
    verify_pugliese_manc_t1_dng100,
    write_cpg_mapping,
    write_neuron_metadata,
    write_roi_parquet,
)


def _print_dng100_checkpoint() -> None:
    raw = load_raw_annotation_table()
    print("raw annotation columns:", raw.columns.tolist())
    print("raw index.name:", raw.index.name)
    print("raw shape:", tuple(raw.shape))
    dng = raw_dng100_annotation_rows()
    print("raw type == DNg100:")
    print(dng.to_string(index=False))
    wanted = {"DNg100", *CORE_CPG_TYPES.values()}
    cells = raw[raw["type"].astype(str).str.strip().isin(wanted)]
    print("groupby type bodyId count/nunique:")
    print(cells.groupby("type")["bodyId"].agg(["count", "nunique"]))
    ids = malecns_dng100_body_ids()
    print("malecns_body_id from that query:", ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-syn-points", action="store_true")
    parser.add_argument("--write-mapping", action="store_true", help="write cpg_mapping.json")
    parser.add_argument("--metadata", action="store_true", help="also write neuron_metadata.parquet from graph.npz")
    args = parser.parse_args(argv)

    _print_dng100_checkpoint()

    checked = verify_pugliese_manc_t1_dng100()
    stim = checked["pugliese_reference"]
    print(
        "Pugliese DNg100_Stim:",
        f"source_dataset={stim['source_dataset']}",
        f"source_matrix_index={stim['source_matrix_index']}",
        f"source_body_id={stim['source_body_id']}",
        f"type={stim['type']}",
    )
    vms = checked["manc_vms16"]
    print(
        "MANC table also has",
        f"source_matrix_index={vms['source_matrix_index']}",
        f"source_body_id={vms['source_body_id']}",
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
    print(f"wrote {PARQUET_CACHE} rows={table.num_rows} (derived; identity is the annotation feather)")
    if args.write_mapping:
        archived = archive_invalid_mapping(MAPPING_PATH)
        if archived:
            print(f"pre-ROI archive present at {archived}")
        ns = archive_invalid_namespace_artifacts()
        for kind, path in ns.items():
            print(f"archived previous {kind} at {path}")
        mapping = build_cpg_mapping()
        write_cpg_mapping(mapping)
        print(f"wrote {MAPPING_PATH}")
        e1 = mapping.get("IN17A001") or {}
        print(f"E1 assigned {e1.get('n_assigned')}/{e1.get('n_expected')}")
        i2 = mapping.get("IN19A007") or {}
        print(
            f"I2 IN19A007 assigned {i2.get('n_assigned')}/{i2.get('n_expected')} "
            f"unassigned={len(i2.get('unassigned') or [])}"
        )
        malecns = (mapping.get("DNg100") or {}).get("malecns") or {}
        left = malecns.get("left") or {}
        right = malecns.get("right") or {}
        print(
            "MaleCNS DNg100 from annotations[type==DNg100]: "
            f"left malecns_body_id={left.get('malecns_body_id')} side={left.get('side')} "
            f"mancBodyid={(left.get('manc_correspondence') or {}).get('manc_body_id')} "
            f"right malecns_body_id={right.get('malecns_body_id')} side={right.get('side')} "
            f"mancBodyid={(right.get('manc_correspondence') or {}).get('manc_body_id')}"
        )
        print(
            "Pugliese reference remains "
            f"source_dataset={PUGLIESE_DNG100_STIM['source_dataset']} "
            f"source_matrix_index={PUGLIESE_DNG100_STIM['source_matrix_index']} "
            f"source_body_id={PUGLIESE_DNG100_STIM['source_body_id']} "
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

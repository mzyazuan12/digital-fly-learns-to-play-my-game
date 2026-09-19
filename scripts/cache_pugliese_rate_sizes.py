#!/usr/bin/env python3
"""Fetch neuPrint MaleCNS Neuron.size for the restricted CPG graph.

Writes data/malecns_v1/pugliese_rate_sizes.parquet. The denominator is the
median volume of all male-cns:v1.0 neurons with a positive size property,
not the median of the 408-cell circuit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.restricted_cpg import restricted_cpg_graph
from flybrain.malecns_volume import (
    SIZE_CACHE,
    inspect_neuron_properties,
    query_full_dataset_size_stats,
    write_size_cache,
)
from flybrain.neurons import MALECNS_DNG100_L, MALECNS_DNG100_R


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SIZE_CACHE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.out.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {args.out} (pass --force)")

    inspect = inspect_neuron_properties(MALECNS_DNG100_R)
    print("inspected neuPrint Neuron properties for", inspect["bodyId"], flush=True)
    print("has_size", inspect["has_size"], "size", inspect["size"], flush=True)
    print("keys", inspect["keys"], flush=True)
    if not inspect["has_size"]:
        raise SystemExit("neuPrint did not return a size property; refusing SWC/synapse proxies")

    stats = query_full_dataset_size_stats()
    print("full-dataset n", stats["n_neurons_with_size"], "median", stats["median_volume"], flush=True)
    print(
        "VNC-synapse n",
        stats.get("vnc_n_neurons_with_size"),
        "median",
        stats.get("vnc_median_volume"),
        " (diagnostic only)",
        flush=True,
    )

    graph = restricted_cpg_graph()
    dng = sorted(int(x) for x in graph.neuron_ids[graph.cell_type == "DNg100"])
    if dng != [MALECNS_DNG100_L, MALECNS_DNG100_R]:
        raise RuntimeError(f"restricted graph DNg100 {dng} != [10045, 10056]")
    print("restricted graph n", graph.n, flush=True)
    meta = write_size_cache(graph.neuron_ids, median_stats=stats, path=args.out)
    print(json.dumps({k: meta[k] for k in ("n_cached_neurons", "n_missing_volume", "median_volume", "dng100_r_volume_raw", "parquet")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

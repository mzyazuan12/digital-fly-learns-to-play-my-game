"""MaleCNS neuron volume from neuPrint ``Neuron.size``, not local proxies.

Pugliese normalize MANC and mCNS by the released volume property. The local
annotation feather has no size column. Do not substitute synapse count,
skeleton cable length, SWC frustum volume, soma size, or partner count.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flybrain.loader import DEFAULT_DATA

NEUPRINT_URL = "https://neuprint.janelia.org/api/custom/custom"
NEUPRINT_DATASET = "male-cns:v1.0"
SIZE_CACHE = DEFAULT_DATA / "pugliese_rate_sizes.parquet"
SIZE_CACHE_META = DEFAULT_DATA / "pugliese_rate_sizes.meta.json"
VOLUME_PROPERTY = "size"
VOLUME_SOURCE = "neuprint male-cns:v1.0 Neuron.size (voxel volume)"
FORBIDDEN_SIZE_TOKENS = (
    "swc",
    "synapse count",
    "cable",
    "soma size",
    "partner",
    "skeleton",
    "frustum",
)
PARQUET_COLUMNS = (
    "malecns_body_id",
    "volume_raw",
    "median_volume_reference",
    "normalized_size",
    "source",
    "dataset_version",
)


def assert_volume_size_source(source: str) -> None:
    """Refuse local morphological proxies that are not neuPrint volume."""
    text = str(source).lower()
    for token in FORBIDDEN_SIZE_TOKENS:
        if token in text:
            raise ValueError(
                f"Pugliese MANC/mCNS size is neuPrint neuron volume, not {token!r}. "
                f"Got source={source!r}."
            )
    if "neuprint" not in text and VOLUME_PROPERTY not in text:
        raise ValueError(
            "Size source must be the neuPrint Neuron.size volume property, "
            f"got {source!r}."
        )


def neuprint_cypher(query: str, *, dataset: str = NEUPRINT_DATASET, timeout: int = 120) -> dict:
    """Run a Cypher query against the public neuPrint HTTP API."""
    body = json.dumps({"cypher": query, "dataset": dataset}).encode()
    request = urllib.request.Request(
        NEUPRINT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def inspect_neuron_properties(body_id: int) -> dict:
    """Return neuPrint property names for one bodyId. Do not assume column names."""
    result = neuprint_cypher(
        f"MATCH (n :`male-cns_Neuron`) WHERE n.bodyId = {int(body_id)} "
        "RETURN n.bodyId AS bodyId, n.size AS size, keys(n) AS keys"
    )
    if not result.get("data"):
        raise RuntimeError(f"neuPrint returned no Neuron for bodyId {body_id}")
    row = result["data"][0]
    keys = list(row[2] or [])
    return {
        "bodyId": int(row[0]),
        "size": None if row[1] is None else float(row[1]),
        "keys": keys,
        "has_size": "size" in keys,
        "dataset": NEUPRINT_DATASET,
    }


def query_full_dataset_size_stats() -> dict:
    """Median volume of every MaleCNS Neuron with a positive size property."""
    result = neuprint_cypher(
        "MATCH (n :`male-cns_Neuron`) "
        "WHERE n.size IS NOT NULL AND n.size > 0 "
        "RETURN count(n) AS n, min(n.size) AS min_s, max(n.size) AS max_s, "
        "avg(n.size) AS avg_s, percentileCont(n.size, 0.5) AS median_s"
    )
    n, min_s, max_s, avg_s, median_s = result["data"][0]
    vnc = neuprint_cypher(
        "MATCH (n :`male-cns_Neuron`) "
        "WHERE n.size IS NOT NULL AND n.size > 0 AND n.VNC > 0 "
        "RETURN count(n) AS n, percentileCont(n.size, 0.5) AS median_s"
    )
    vnc_n, vnc_median = vnc["data"][0]
    return {
        "dataset": NEUPRINT_DATASET,
        "volume_property": VOLUME_PROPERTY,
        "n_neurons_with_size": int(n),
        "min_volume": float(min_s),
        "max_volume": float(max_s),
        "mean_volume": float(avg_s),
        "median_volume": float(median_s),
        "vnc_n_neurons_with_size": int(vnc_n),
        "vnc_median_volume": float(vnc_median),
        "median_used_for_normalization": "full_MaleCNS_Neuron_size",
        "note": (
            "Pugliese normalize by the median of the modeled dataset, not the "
            "tiny selected circuit. This cache uses the median of all neuPrint "
            "male-cns:v1.0 Neuron.size values. Their published mCNS model was a "
            "4,310-cell VNC-ROI population whose exact membership is not in this "
            "checkout; the VNC-synapse median is recorded only as a diagnostic."
        ),
    }


def query_neuron_sizes(body_ids, *, batch_size: int = 80) -> dict[int, float]:
    """Map malecns_body_id -> raw neuPrint size. Missing bodies are omitted."""
    ids = [int(x) for x in body_ids]
    out: dict[int, float] = {}
    for start in range(0, len(ids), batch_size):
        chunk = ids[start : start + batch_size]
        listed = ", ".join(str(i) for i in chunk)
        result = neuprint_cypher(
            f"UNWIND [{listed}] AS bid "
            "MATCH (n :`male-cns_Neuron`) WHERE n.bodyId = bid "
            "RETURN n.bodyId AS bodyId, n.size AS size"
        )
        for body_id, size in result.get("data") or []:
            if size is None:
                continue
            out[int(body_id)] = float(size)
    return out


def _normalized_size(volume_raw: float, median: float) -> float:
    if not np.isfinite(volume_raw) or volume_raw <= 0:
        return 1.0
    return float(volume_raw / median)


def write_size_cache(body_ids, *, median_stats: dict | None = None, path: Path = SIZE_CACHE) -> dict:
    """Write the restricted-circuit volume table with the full-dataset median."""
    inspect = inspect_neuron_properties(10056)
    if not inspect["has_size"]:
        raise RuntimeError(
            "neuPrint male-cns Neuron for 10056 has no 'size' property. "
            f"keys={inspect['keys']}"
        )
    stats = dict(median_stats or query_full_dataset_size_stats())
    median = float(stats["median_volume"])
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Full-dataset median volume must be positive and finite")
    raw = query_neuron_sizes(body_ids)
    rows = []
    missing = []
    for body in (int(x) for x in body_ids):
        volume = raw.get(body, float("nan"))
        if not np.isfinite(volume) or volume <= 0:
            missing.append(body)
            volume_raw = float("nan")
            normalized = 1.0
        else:
            volume_raw = float(volume)
            normalized = _normalized_size(volume_raw, median)
        rows.append(
            {
                "malecns_body_id": body,
                "volume_raw": volume_raw,
                "median_volume_reference": median,
                "normalized_size": normalized,
                "source": VOLUME_SOURCE,
                "dataset_version": NEUPRINT_DATASET,
            }
        )
    table = pa.table({name: [row[name] for row in rows] for name in PARQUET_COLUMNS})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    meta = {
        **stats,
        "n_cached_neurons": len(rows),
        "n_missing_volume": len(missing),
        "missing_malecns_body_ids": missing,
        "inspected_body_id": inspect["bodyId"],
        "inspected_keys_include_size": inspect["has_size"],
        "dng100_r_volume_raw": raw.get(10056),
        "dng100_l_volume_raw": raw.get(10045),
        "parquet": str(path.resolve()),
        "source": VOLUME_SOURCE,
    }
    SIZE_CACHE_META.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_size_cache(path: Path = SIZE_CACHE):
    """Load the cached volume table. Refuses a non-volume source column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Fetch neuPrint Neuron.size with "
            "python scripts/cache_pugliese_rate_sizes.py"
        )
    table = pq.read_table(path).to_pandas()
    missing = [name for name in PARQUET_COLUMNS if name not in table.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    sources = {str(s) for s in table["source"].unique()}
    for source in sources:
        assert_volume_size_source(source)
    medians = np.asarray(table["median_volume_reference"], dtype=np.float64)
    if medians.size == 0 or not np.isfinite(medians).all():
        raise ValueError("median_volume_reference is missing or non-finite")
    if float(np.nanmax(medians) - np.nanmin(medians)) > 1e-6 * max(abs(float(medians[0])), 1.0):
        raise ValueError("median_volume_reference must be the same full-dataset median on every row")
    return table


def sizes_for_body_ids(body_ids, *, path: Path = SIZE_CACHE) -> dict:
    """Align cached neuPrint volumes to a neuron order. Missing IDs are NaN."""
    table = load_size_cache(path)
    by_id = table.set_index("malecns_body_id")
    ids = np.asarray([int(x) for x in body_ids], dtype=np.int64)
    volumes = np.full(ids.shape, np.nan, dtype=np.float64)
    normalized = np.full(ids.shape, np.nan, dtype=np.float64)
    missing = []
    median = float(table["median_volume_reference"].iloc[0])
    source = str(table["source"].iloc[0])
    dataset_version = str(table["dataset_version"].iloc[0])
    for i, body in enumerate(ids):
        if body not in by_id.index:
            missing.append(int(body))
            continue
        row = by_id.loc[body]
        volumes[i] = float(row["volume_raw"])
        normalized[i] = float(row["normalized_size"])
        if not np.isfinite(volumes[i]) or volumes[i] <= 0:
            missing.append(int(body))
    return {
        "sizes": volumes,
        "normalized_size": normalized,
        "median_volume_reference": median,
        "source": source,
        "dataset_version": dataset_version,
        "n_missing": len(missing),
        "missing_malecns_body_ids": missing,
        "not_neuprint_volume_property": False,
        "cache_path": str(Path(path).resolve()),
    }

"""Restricted MaleCNS CPG subgraph. Never the full 166k-cell graph.

Identity is the official annotation feather. Edges are core-neuron outgoing
contacts onto core cells and directly downstream motor readouts. Motor
feedback and other populations are omitted.
"""

from __future__ import annotations

import numpy as np
import pyarrow.dataset as ds
import pyarrow.feather as feather

from flybrain.loader import ANN_FILE, DEFAULT_DATA, EDGE_FILE, NT_FILE, Connectome, _coo_to_csr
from flybrain.neurons import MALECNS_DNG100_L, MALECNS_DNG100_R, nt_sign
from organism.roi_innervation import CORE_CPG_TYPES, assert_expected_annotation_counts


def restricted_cpg_graph(*, dng100_bodies: tuple[int, ...] = (MALECNS_DNG100_L, MALECNS_DNG100_R)) -> Connectome:
    """Build the restricted anatomical experiment graph.

    ``dng100_bodies`` selects which MaleCNS DNg100 cells are in the core.
    The primary transfer test uses only DNg100_R 10056. Passing both
    bodies reproduces the all-LIF milestone subgraph.
    """
    wanted = {int(b) for b in dng100_bodies}
    if not wanted.issubset({MALECNS_DNG100_L, MALECNS_DNG100_R}):
        raise ValueError(f"dng100_bodies must be MaleCNS DNg100 {{10045, 10056}}, got {sorted(wanted)}")
    raw = feather.read_table(DEFAULT_DATA / ANN_FILE).to_pandas()
    assert_expected_annotation_counts(raw)
    cpg = raw[raw["type"].isin(list(CORE_CPG_TYPES.values()))]
    dng = raw[(raw["type"] == "DNg100") & raw["bodyId"].isin(wanted)]
    if len(dng) != len(wanted):
        raise RuntimeError(f"Requested DNg100 {sorted(wanted)} not all present in annotations")
    core = dng.bodyId.astype(np.uint64).tolist() + cpg.bodyId.astype(np.uint64).tolist()
    ids = np.sort(np.array(core, dtype=np.uint64))
    chunks = []
    dataset = ds.dataset(str(DEFAULT_DATA / EDGE_FILE), format="feather")
    for batch in dataset.scanner(
        columns=["body_pre", "body_post", "weight"],
        filter=ds.field("body_pre").isin(ids.tolist()),
        batch_size=65536,
    ).to_batches():
        if batch.num_rows:
            chunks.append(batch.to_pandas())
    import pandas as pd

    edges = pd.concat(chunks, ignore_index=True)
    motor_ids = set(
        raw.loc[raw.superclass.astype(str).str.contains("motor", case=False), "bodyId"].astype(int)
    )
    downstream = set(edges.body_post.astype(int)) & motor_ids
    ids = np.array(sorted(set(ids.tolist()) | downstream), dtype=np.uint64)
    edges = edges[edges.body_post.isin(ids)]
    ann = raw.set_index("bodyId").loc[ids]
    nt = (
        feather.read_table(DEFAULT_DATA / NT_FILE, columns=["body", "consensus_nt"])
        .to_pandas()
        .set_index("body")
        .consensus_nt
    )
    tx = np.array([str(nt.get(int(i), "missing")) for i in ids], dtype=object)
    pre = np.searchsorted(ids, edges.body_pre.to_numpy(dtype=np.uint64))
    post = np.searchsorted(ids, edges.body_post.to_numpy(dtype=np.uint64))
    ptr, post, weights = _coo_to_csr(pre, post, edges.weight.to_numpy(dtype=np.uint32), len(ids))
    signs = np.repeat(np.array([nt_sign(t) for t in tx], dtype=np.int8), np.diff(ptr))

    def col(name):
        return ann[name].fillna("").astype(str).to_numpy(dtype=object)

    n_core = int(len(dng) + len(cpg))
    n_motor = int(len(ids) - n_core)
    graph = Connectome(
        ids,
        ptr,
        post,
        weights,
        signs,
        col("superclass"),
        col("type"),
        col("class"),
        col("somaSide"),
        tx,
        {
            "dataset_id": "MaleCNS_v1.0",
            "subset": (
                f"{n_core} core neurons (DNg100 {sorted(wanted)} plus E1/E2/E3/I1/I2) "
                f"plus {n_motor} directly downstream motor readouts; motor feedback omitted"
            ),
            "dng100_bodies": sorted(wanted),
            "n_core": n_core,
            "n_motor_readouts": n_motor,
            "motor_feedback_included": False,
        },
    )
    return graph

"""MaleCNS v1.0 sparse graph importer.

Node policy (DoomFly / MaleCNS retained graph):
  keep every annotation with an assigned superclass; drop explicit Glia.
Edge policy:
  keep every released edge whose pre and post bodies are both retained;
  no extra weight threshold; autapses kept.

The released weights table has ~152M rows because it includes edges onto
unannotated/non-neuronal objects. After the node filter the directed graph
is ~25.6M rows / ~124M synaptic contacts.

This module stores topology. It does not infer missing electrophysiology.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from flybrain.ids import exact_ids, index_edges
from flybrain.neurons import nt_sign

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "malecns_v1"

EDGE_FILE = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
ANN_FILE = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NT_FILE = "body-neurotransmitters-male-cns-v1.0.feather"

EXPECTED_RETAINED = 166_700
EXPECTED_EDGES = 25_582_938
EXPECTED_CONTACTS = 124_177_617


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _memory_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports kilobytes.
        return usage / (1024 * 1024) if usage > 10**8 else usage / 1024
    except Exception:
        return float("nan")


@dataclass
class Connectome:
    """Sparse directed MaleCNS graph plus annotations."""

    neuron_ids: np.ndarray  # uint64, sorted
    pre_ptr: np.ndarray  # int64 CSR indptr, outgoing
    post: np.ndarray  # uint32
    anatomical: np.ndarray  # uint32 synapse counts
    sign: np.ndarray  # int8, per edge (presynaptic NT sign)
    superclass: np.ndarray  # object/str
    cell_type: np.ndarray
    cell_class: np.ndarray
    side: np.ndarray
    neurotransmitter: np.ndarray
    report: dict

    @property
    def n(self) -> int:
        return int(len(self.neuron_ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.post))

    def index_of(self, body_id: int) -> int:
        ids = self.neuron_ids
        body = np.uint64(body_id)
        i = int(np.searchsorted(ids, body))
        if i >= len(ids) or ids[i] != body:
            raise KeyError(body_id)
        return i

    def lookup(
        self,
        *,
        superclass: str | None = None,
        cell_class: str | None = None,
        type_prefix: str | None = None,
        type_exact: str | None = None,
        side: str | None = None,
        contains: str | None = None,
    ) -> np.ndarray:
        """Return node indices matching annotation filters (AND)."""
        mask = np.ones(self.n, dtype=bool)
        if superclass is not None:
            mask &= self.superclass == superclass
        if cell_class is not None:
            mask &= self.cell_class == cell_class
        if type_exact is not None:
            mask &= self.cell_type == type_exact
        if type_prefix is not None:
            mask &= np.array(
                [str(t).startswith(type_prefix) for t in self.cell_type], dtype=bool
            )
        if side is not None:
            mask &= self.side == side
        if contains is not None:
            needle = contains.lower()
            mask &= np.array(
                [needle in str(t).lower() for t in self.cell_type], dtype=bool
            )
        return np.flatnonzero(mask).astype(np.int32)

    def stats(self) -> dict:
        incoming = np.zeros(self.n, dtype=np.int64)
        outgoing = np.zeros(self.n, dtype=np.int64)
        np.add.at(outgoing, self._pre_index(), self.anatomical.astype(np.int64))
        np.add.at(incoming, self.post, self.anatomical.astype(np.int64))
        return {
            "neurons": self.n,
            "directed_edges": self.n_edges,
            "synaptic_contacts": int(self.anatomical.sum(dtype=np.uint64)),
            "autapses": int(np.count_nonzero(self._pre_index() == self.post)),
            "isolated_neurons": int(
                np.count_nonzero((incoming == 0) & (outgoing == 0))
            ),
            "inhibitory_edges": int(np.count_nonzero(self.sign < 0)),
            "excitatory_edges": int(np.count_nonzero(self.sign > 0)),
        }

    def _pre_index(self) -> np.ndarray:
        pre = np.empty(self.n_edges, dtype=np.uint32)
        for i in range(self.n):
            pre[self.pre_ptr[i] : self.pre_ptr[i + 1]] = i
        return pre

    def save_normalized(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "graph.npz",
            neuron_ids=self.neuron_ids,
            pre_ptr=self.pre_ptr,
            post=self.post,
            anatomical=self.anatomical,
            sign=self.sign,
            superclass=self.superclass.astype(object),
            cell_type=self.cell_type.astype(object),
            cell_class=self.cell_class.astype(object),
            side=self.side.astype(object),
            neurotransmitter=self.neurotransmitter.astype(object),
        )
        (directory / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    @classmethod
    def from_normalized(cls, directory: Path) -> "Connectome":
        directory = Path(directory)
        data = np.load(directory / "graph.npz", allow_pickle=True)
        report = json.loads((directory / "report.json").read_text())
        return cls(
            neuron_ids=data["neuron_ids"],
            pre_ptr=data["pre_ptr"],
            post=data["post"],
            anatomical=data["anatomical"],
            sign=data["sign"],
            superclass=data["superclass"],
            cell_type=data["cell_type"],
            cell_class=data["cell_class"],
            side=data["side"],
            neurotransmitter=data["neurotransmitter"],
            report=report,
        )


def _coo_to_csr(pre: np.ndarray, post: np.ndarray, weight: np.ndarray, n: int):
    order = np.argsort(pre, kind="stable")
    pre = pre[order]
    post = post[order]
    weight = weight[order]
    counts = np.bincount(pre, minlength=n)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts, dtype=np.int64)
    return ptr, post, weight


def _expand_pre(ptr: np.ndarray) -> np.ndarray:
    n_edges = int(ptr[-1])
    pre = np.empty(n_edges, dtype=np.uint32)
    for i in range(len(ptr) - 1):
        pre[ptr[i] : ptr[i + 1]] = i
    return pre


def import_malecns(
    data_dir: Path | None = None,
    *,
    verify_hash: bool = False,
    progress: bool = True,
) -> Connectome:
    """Load the three official Feather files into a sparse CSR graph."""
    started = time.perf_counter()
    data_dir = Path(data_dir or DEFAULT_DATA)
    ann_path = data_dir / ANN_FILE
    nt_path = data_dir / NT_FILE
    edge_path = data_dir / EDGE_FILE
    for path in (ann_path, nt_path, edge_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing MaleCNS file: {path}")

    if progress:
        print(f"loading annotations from {ann_path.name} …")
    frame = feather.read_table(
        ann_path,
        columns=["bodyId", "superclass", "type", "class", "status", "statusLabel", "somaSide"],
    ).to_pandas()
    source = exact_ids(frame["bodyId"])
    superclass = frame["superclass"]
    retain = superclass.notna() & superclass.astype(str).ne("")
    nonneural = frame["status"].eq("Glia")
    retain = retain & ~nonneural
    if int(retain.sum()) != EXPECTED_RETAINED:
        raise ValueError(
            f"Retained neuron count {int(retain.sum())} != {EXPECTED_RETAINED}. "
            "Check node policy / source files."
        )

    nodes = frame.loc[retain].copy()
    nodes["_id"] = source[retain.to_numpy()]
    nodes = nodes.sort_values("_id", kind="mergesort")
    ids = exact_ids(nodes["_id"].to_numpy())
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate retained body IDs.")

    if progress:
        print(f"loading neurotransmitters from {nt_path.name} …")
    nt_frame = feather.read_table(nt_path, columns=["body", "consensus_nt"]).to_pandas()
    nt_map = dict(zip(exact_ids(nt_frame["body"]).tolist(), nt_frame["consensus_nt"].tolist()))
    transmitters = np.array(
        [str(nt_map.get(int(body), "missing") or "missing") for body in ids],
        dtype=object,
    )
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)

    if progress:
        print(f"streaming {edge_path.name} ({edge_path.stat().st_size / 1e9:.2f} GB) …")
    mmap = pa.memory_map(str(edge_path), "r")
    reader = ipc.open_file(mmap)
    pre_chunks: list[np.ndarray] = []
    post_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    stats = {
        "source_edge_rows": 0,
        "retained_edge_rows": 0,
        "source_synaptic_contacts": 0,
        "retained_synaptic_contacts": 0,
        "retained_self_edges": 0,
        "retained_weight_one_edges": 0,
    }
    n_batches = reader.num_record_batches
    for b in range(n_batches):
        batch = reader.get_batch(b)
        pre_raw = batch.column(batch.schema.get_field_index("body_pre")).to_numpy(zero_copy_only=False)
        post_raw = batch.column(batch.schema.get_field_index("body_post")).to_numpy(zero_copy_only=False)
        w_raw = batch.column(batch.schema.get_field_index("weight")).to_numpy(zero_copy_only=False)
        i, j, count, _keep = index_edges(ids, pre_raw, post_raw, w_raw)
        stats["source_edge_rows"] += len(pre_raw)
        stats["retained_edge_rows"] += int(len(i))
        stats["source_synaptic_contacts"] += int(np.asarray(w_raw).sum(dtype=np.uint64))
        stats["retained_synaptic_contacts"] += int(count.sum(dtype=np.uint64))
        stats["retained_weight_one_edges"] += int(np.count_nonzero(count == 1))
        stats["retained_self_edges"] += int(np.count_nonzero(i == j))
        if len(i):
            pre_chunks.append(i)
            post_chunks.append(j)
            weight_chunks.append(count)
        if progress and ((b + 1) % 20 == 0 or b + 1 == n_batches):
            print(
                f"  batch {b + 1}/{n_batches}  retained edges={stats['retained_edge_rows']:,}  "
                f"rss≈{_memory_mb():.0f} MB"
            )

    if not pre_chunks:
        raise RuntimeError("No retained edges.")
    pre = np.concatenate(pre_chunks)
    post = np.concatenate(post_chunks)
    anatomical = np.concatenate(weight_chunks)
    del pre_chunks, post_chunks, weight_chunks

    if progress:
        print("building outgoing CSR …")
    ptr, post, anatomical = _coo_to_csr(pre, post, anatomical, len(ids))
    del pre
    edge_sign = np.empty(len(post), dtype=np.int8)
    for i in range(len(ids)):
        edge_sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]

    quality = nodes["statusLabel"].astype(object).fillna("unknown")
    report = {
        "dataset_id": "malecns_v1",
        "release": "MaleCNS v1.0",
        "coverage": "brain_and_ventral_nerve_cord",
        "node_policy": (
            "Every entry with an assigned superclass; exclude explicit Glia status; "
            "no restriction to Traced status or typed cells."
        ),
        "edge_policy": (
            "All released edges between retained entries; no additional weight "
            "threshold; autapses kept."
        ),
        "source_annotation_rows": int(len(frame)),
        "retained_neuron_candidates": int(len(ids)),
        "quality_counts": quality.value_counts(dropna=False).to_dict(),
        "superclass_counts": nodes["superclass"].fillna("unknown").value_counts().to_dict(),
        "neurotransmitter_counts": {
            str(k): int(v)
            for k, v in zip(*np.unique(transmitters, return_counts=True))
        },
        "graph": {
            **stats,
            "excluded_edge_rows": stats["source_edge_rows"] - stats["retained_edge_rows"],
            "excluded_synaptic_contacts": (
                stats["source_synaptic_contacts"] - stats["retained_synaptic_contacts"]
            ),
        },
        "expected": {
            "retained_neurons": EXPECTED_RETAINED,
            "directed_edges": EXPECTED_EDGES,
            "synaptic_contacts": EXPECTED_CONTACTS,
        },
        "matches_expected_neurons": int(len(ids)) == EXPECTED_RETAINED,
        "matches_expected_edges": stats["retained_edge_rows"] == EXPECTED_EDGES,
        "matches_expected_contacts": stats["retained_synaptic_contacts"] == EXPECTED_CONTACTS,
        "neural_dynamics_validated": False,
        "learning_demonstrated": False,
        "source_bytes": {
            ANN_FILE: ann_path.stat().st_size,
            NT_FILE: nt_path.stat().st_size,
            EDGE_FILE: edge_path.stat().st_size,
        },
        "import_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mb_approx": round(_memory_mb(), 1),
        "remaining_gaps": [
            "Receptor-dependent synapse dynamics and neuromodulation are unspecified",
            "LIF constants are modeling assumptions, not MaleCNS measurements",
            "Keyboard encoder/decoder is an engineered BCI, not a biological mapping",
        ],
    }
    if verify_hash:
        report["source_sha256"] = {
            ANN_FILE: _file_digest(ann_path),
            NT_FILE: _file_digest(nt_path),
            EDGE_FILE: _file_digest(edge_path),
        }

    connectome = Connectome(
        neuron_ids=ids,
        pre_ptr=ptr,
        post=post.astype(np.uint32, copy=False),
        anatomical=anatomical.astype(np.uint32, copy=False),
        sign=edge_sign,
        superclass=nodes["superclass"].fillna("").astype(str).to_numpy(),
        cell_type=nodes["type"].fillna("").astype(str).to_numpy(),
        cell_class=nodes["class"].fillna("").astype(str).to_numpy(),
        side=nodes["somaSide"].fillna("").astype(str).to_numpy(),
        neurotransmitter=transmitters,
        report=report,
    )
    cache = data_dir / "normalized"
    connectome.save_normalized(cache)
    if progress:
        g = connectome.stats()
        print(
            f"retained neurons={g['neurons']:,}  edges={g['directed_edges']:,}  "
            f"contacts={g['synaptic_contacts']:,}  rss≈{_memory_mb():.0f} MB"
        )
        print(f"wrote cache → {cache}")
    return connectome


def load_connectome(
    data_dir: Path | None = None,
    *,
    rebuild: bool = False,
    progress: bool = True,
) -> Connectome:
    """Load the normalized cache, importing from Feather files if needed."""
    data_dir = Path(data_dir or DEFAULT_DATA)
    cache = data_dir / "normalized" / "graph.npz"
    if rebuild or not cache.exists():
        return import_malecns(data_dir, progress=progress)
    if progress:
        print(f"loading normalized graph from {cache} …")
    return Connectome.from_normalized(cache.parent)


def shuffled_connectome(connectome: Connectome, rng: np.random.Generator) -> Connectome:
    """Destroy pairing: permute post indices, keep out-weights and in-degree sequence."""
    post = connectome.post.copy()
    rng.shuffle(post)
    report = dict(connectome.report)
    report["control"] = "shuffled_post_indices"
    return Connectome(
        neuron_ids=connectome.neuron_ids,
        pre_ptr=connectome.pre_ptr,
        post=post,
        anatomical=connectome.anatomical,
        sign=connectome.sign,
        superclass=connectome.superclass,
        cell_type=connectome.cell_type,
        cell_class=connectome.cell_class,
        side=connectome.side,
        neurotransmitter=connectome.neurotransmitter,
        report=report,
    )


def rewired_connectome(connectome: Connectome, rng: np.random.Generator) -> Connectome:
    """Degree-preserving configuration-model rewire of existing edge endpoints."""
    pre = _expand_pre(connectome.pre_ptr)
    post = connectome.post.copy()
    rng.shuffle(post)
    # Rebuild CSR so outgoing bundles stay contiguous after endpoint shuffle.
    ptr, post, anatomical = _coo_to_csr(pre, post, connectome.anatomical.copy(), connectome.n)
    sign = np.empty(len(post), dtype=np.int8)
    neuron_sign = np.array(
        [nt_sign(name) for name in connectome.neurotransmitter], dtype=np.int8
    )
    for i in range(connectome.n):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    report = dict(connectome.report)
    report["control"] = "degree_preserving_rewire"
    return Connectome(
        neuron_ids=connectome.neuron_ids,
        pre_ptr=ptr,
        post=post.astype(np.uint32, copy=False),
        anatomical=anatomical,
        sign=sign,
        superclass=connectome.superclass,
        cell_type=connectome.cell_type,
        cell_class=connectome.cell_class,
        side=connectome.side,
        neurotransmitter=connectome.neurotransmitter,
        report=report,
    )


def synthetic_connectome(n: int = 64, extra_edges: int = 80, seed: int = 0) -> Connectome:
    """Tiny deterministic graph for unit tests (not biology)."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1, dtype=np.uint64)
    pre = np.arange(n - 1, dtype=np.uint32)
    post = np.arange(1, n, dtype=np.uint32)
    # Strong enough that a single chain hop can elicit a postsynaptic spike
    # under the default LIF gain (modeling assumption, not biology).
    weight = np.full(n - 1, 200, dtype=np.uint32)
    extra_pre = rng.integers(0, n, size=extra_edges, dtype=np.uint32)
    extra_post = rng.integers(0, n, size=extra_edges, dtype=np.uint32)
    extra_w = rng.integers(1, 8, size=extra_edges, dtype=np.uint32)
    pre = np.concatenate([pre, extra_pre])
    post = np.concatenate([post, extra_post])
    weight = np.concatenate([weight, extra_w])
    ptr, post, weight = _coo_to_csr(pre, post, weight, n)
    transmitters = np.array(
        ["gaba" if i % 5 == 4 else "acetylcholine" for i in range(n)], dtype=object
    )
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    sign = np.empty(len(post), dtype=np.int8)
    for i in range(n):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    superclasses = np.array(
        ["cb_sensory" if i < n // 3 else ("descending_neuron" if i >= 2 * n // 3 else "cb_intrinsic")
         for i in range(n)],
        dtype=object,
    )
    return Connectome(
        neuron_ids=ids,
        pre_ptr=ptr,
        post=post,
        anatomical=weight,
        sign=sign,
        superclass=superclasses,
        cell_type=np.array([f"toy{i}" for i in range(n)], dtype=object),
        cell_class=np.array(["sensory" if i < n // 3 else "" for i in range(n)], dtype=object),
        side=np.array(["L" if i % 2 == 0 else "R" for i in range(n)], dtype=object),
        neurotransmitter=transmitters,
        report={
            "dataset_id": "synthetic",
            "retained_neuron_candidates": n,
            "neural_dynamics_validated": False,
            "learning_demonstrated": False,
        },
    )

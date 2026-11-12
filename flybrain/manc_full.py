"""Authors' full MANC VNC network (all six legs), as a sparse matrix.

Source: third_party/Pugliese_2026/data/manc full vnc data/
  wTable_20251006.feather  23,532 neurons (the paper's full-MANC count)
  W_20251006.feather       signed synapse counts, rows = bodyId_pre, columns = post
  (the 20260522_allSynapses pair is a later 23,628-neuron export)

The dense W is 23,532^2 float64 (4.4 GB), so it is read column by column
from the memory-mapped feather and stored once as CSR W[pre, post] in a
gitignored cache. Nothing is re-signed: the shipped W is already signed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANC_FULL_DIR = ROOT / "third_party" / "Pugliese_2026" / "data" / "manc full vnc data"
CACHE_DIR = ROOT / "data" / "manc_full_cache"
VERSIONS = {
    "20251006": ("wTable_20251006.feather", "W_20251006.feather"),
}
LEGS = [f"{seg}{side}" for seg in ("T1", "T2", "T3") for side in ("L", "R")]


def load_wtable(version: str = "20251006") -> pd.DataFrame:
    table = pd.read_feather(MANC_FULL_DIR / VERSIONS[version][0])
    if "bodyId" not in table.columns or "size" not in table.columns:
        raise ValueError("full MANC wTable lacks bodyId or size")
    return table


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_W_csr(version: str = "20251006", *, rebuild: bool = False):
    """Signed W[pre, post] as scipy.sparse CSR (float64)."""
    import pyarrow.feather as pf
    from scipy import sparse

    src = MANC_FULL_DIR / VERSIONS[version][1]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"W_{version}_csr.npz"
    if cache.exists() and not rebuild:
        return sparse.load_npz(cache).tocsr()
    t = pf.read_table(src, memory_map=True)
    names = t.column_names
    if names[-1] != "bodyId_pre":
        raise ValueError("expected bodyId_pre as the last column")
    n = t.num_rows
    post_cols = names[:-1]
    if len(post_cols) != n:
        raise ValueError("W is not square")
    rows, cols, vals = [], [], []
    for j, name in enumerate(post_cols):
        col = t.column(j).to_numpy()
        nz = np.flatnonzero(col)
        if nz.size:
            rows.append(nz.astype(np.int32))
            cols.append(np.full(nz.size, j, dtype=np.int32))
            vals.append(col[nz].astype(np.float64))
    W = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    sparse.save_npz(cache, W)
    return W


def check_alignment(table: pd.DataFrame, version: str = "20251006") -> dict:
    """Row order of W (bodyId_pre) and column names must equal wTable.bodyId."""
    import pyarrow.feather as pf

    t = pf.read_table(MANC_FULL_DIR / VERSIONS[version][1], memory_map=True)
    pre = t.column("bodyId_pre").to_numpy().astype(np.int64)
    post = np.array([int(c) for c in t.column_names[:-1]], dtype=np.int64)
    body = table["bodyId"].to_numpy(dtype=np.int64)
    return {
        "pre_matches_table": bool(np.array_equal(pre, body)),
        "post_matches_table": bool(np.array_equal(post, body)),
    }


def reweight_csr(W, exc_mult: float, inh_mult: float):
    """Sparse transcription of reweight_connectivity: transpose, scale by sign."""
    Wt = W.T.tocsr().astype(np.float64)
    Wt.data = np.where(Wt.data > 0, exc_mult * Wt.data, inh_mult * Wt.data)
    return Wt


def leg_of(table: pd.DataFrame) -> np.ndarray:
    """'T1L' … 'T3R' from somaNeuromere + somaSide, '' when either is missing."""
    seg = table["somaNeuromere"].astype(str).fillna("")
    side = table["somaSide"].astype(str).str[:1].replace({"N": "", "n": ""})
    out = np.where(seg.isin(["T1", "T2", "T3"]) & side.isin(["L", "R"]), seg + side, "")
    return out

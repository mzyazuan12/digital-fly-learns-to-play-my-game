"""Authors' shipped mCNS / IMAC simulation export.

The reproduction network is the pair in Pugliese_2026 commit 10e7661:

  data/imac t1 connectome data/wTable_20260210_vncRoisOnly.csv
  data/imac t1 connectome data/W_20260210_vncRoisOnly.csv

That table is the set_sizes() population. A MaleCNS v1.0 methods
reconstruction is a separate audit and is not this denominator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.neurons import MALECNS_DNG100_L, MALECNS_DNG100_R, PUGLIESE_THRESHOLD_MEAN

ROOT = Path(__file__).resolve().parents[1]
AUTHORS_REPO = ROOT / "third_party" / "Pugliese_2026"
AUTHORS_COMMIT = "10e7661bf414ba7b4c2edf795cd36d0f878c17c0"
IMAC_DIR = AUTHORS_REPO / "data" / "imac t1 connectome data"
WTABLE_PATH = IMAC_DIR / "wTable_20260210_vncRoisOnly.csv"
W_PATH = IMAC_DIR / "W_20260210_vncRoisOnly.csv"
AUTHORS_MCNS_N = 4310
POPULATION_AUTHORS = "authors_export_population"
POPULATION_RECONSTRUCTION = "reconstructed_from_male-cns:v1.0"
STIMULUS_MANC_AMPLITUDE_CONTROL = "MANC amplitude control"
STIMULUS_AUTHORS_MCNS = "authors_mcns"
# Body 20129's own size. It is a row in the table, not nanmedian(size).
NOT_THE_TABLE_MEDIAN = 1_010_837_052
EXPORT_STATUS_PATH = ROOT / "data" / "malecns_v1" / "authors_mcns_export_status.json"
DIFF_PATH = ROOT / "data" / "malecns_v1" / "mcns_body_id_diff.json"
RECONSTRUCTION_IDS_PATH = (
    ROOT / "data" / "malecns_v1" / "reconstructed_from_male-cns_v1.0.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_authors_wtable(path: Path = WTABLE_PATH) -> pd.DataFrame:
    """wTable as load_wTable reads it: CSV with the leading index column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Authors' mCNS wTable is not in this checkout: {path}")
    table = pd.read_csv(path, index_col=0)
    if "bodyId" not in table.columns or "size" not in table.columns:
        raise ValueError(f"{path} is missing bodyId or size")
    return table


def load_authors_W(path: Path = W_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed W[pre, post] as load_W reads it. Do not re-sign from consensusNt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Authors' mCNS W is not in this checkout: {path}")
    frame = pd.read_csv(path)
    if "bodyId_pre" not in frame.columns:
        raise ValueError(f"{path} has no bodyId_pre column")
    pre = frame["bodyId_pre"].to_numpy(dtype=np.int64)
    post = np.asarray([int(column) for column in frame.columns if column != "bodyId_pre"], dtype=np.int64)
    matrix = frame.drop(columns="bodyId_pre").to_numpy(dtype=np.float64)
    return matrix, pre, post


def _consensus_counts(table: pd.DataFrame) -> dict[str, int]:
    labels = table["consensusNt"].fillna("<blank>").astype(str).str.strip().replace({"": "<blank>"})
    counts = labels.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def export_summary(
    *,
    wtable_path: Path = WTABLE_PATH,
    w_path: Path = W_PATH,
) -> dict:
    """Measured facts from the shipped files. Membership is every wTable row."""
    table = load_authors_wtable(wtable_path)
    matrix, pre, post = load_authors_W(w_path)
    body_ids = table["bodyId"].astype(int).to_numpy()
    if len(table) != AUTHORS_MCNS_N:
        raise ValueError(f"Authors' wTable has {len(table)} rows, expected {AUTHORS_MCNS_N}")
    if matrix.shape != (AUTHORS_MCNS_N, AUTHORS_MCNS_N):
        raise ValueError(f"Authors' W shape {matrix.shape} is not {(AUTHORS_MCNS_N, AUTHORS_MCNS_N)}")
    if not np.array_equal(pre, body_ids) or not np.array_equal(post, body_ids):
        raise ValueError("W row/column body IDs do not match wTable bodyId order")
    if int(pd.Series(body_ids).duplicated().sum()) != 0:
        raise ValueError("Authors' wTable bodyId values are not unique")
    sizes = table["size"].to_numpy(dtype=np.float64)
    positive = np.isfinite(sizes) & (sizes > 0)
    missing = ~np.isfinite(sizes)
    zeros = np.isfinite(sizes) & (sizes == 0)
    median = float(np.nanmedian(sizes))
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Authors' size median is not a positive finite number")
    if median == float(NOT_THE_TABLE_MEDIAN):
        raise ValueError(
            f"{NOT_THE_TABLE_MEDIAN} is a single neuron's size, not nanmedian of this table"
        )
    consensus = _consensus_counts(table)
    nt = table["consensusNt"].fillna("").astype(str).str.strip().to_numpy()
    outgoing = {}
    for name in sorted(set(nt.tolist())):
        mask = nt == name
        block = matrix[mask]
        label = name if name else "<blank>"
        outgoing[label] = {
            "n": int(mask.sum()),
            "outgoing_positive": int(np.count_nonzero(block > 0)),
            "outgoing_negative": int(np.count_nonzero(block < 0)),
        }
    dng = {}
    for body in (MALECNS_DNG100_R, MALECNS_DNG100_L):
        hits = np.flatnonzero(body_ids == int(body))
        if hits.size != 1:
            raise ValueError(f"bodyId {body} appears {hits.size} times in the authors' wTable")
        index = int(hits[0])
        row = table.iloc[index]
        size = float(row["size"])
        normalized = size / median
        dng[str(body)] = {
            "bodyId": int(body),
            "matrix_index": index,
            "type": str(row["type"]),
            "instance": None if pd.isna(row["instance"]) else str(row["instance"]),
            "somaSide": None if pd.isna(row["somaSide"]) else str(row["somaSide"]),
            "consensusNt": None if pd.isna(row["consensusNt"]) else str(row["consensusNt"]),
            "size": size,
            "normalized_size": normalized,
            "mean_threshold_after_set_sizes": float(PUGLIESE_THRESHOLD_MEAN) * normalized,
        }
    return {
        "population": POPULATION_AUTHORS,
        "not_population": POPULATION_RECONSTRUCTION,
        "authors_commit": AUTHORS_COMMIT,
        "wtable": str(Path(wtable_path)),
        "W": str(Path(w_path)),
        "wtable_sha256": _sha256(wtable_path),
        "W_sha256": _sha256(w_path),
        "n": int(len(table)),
        "size": {
            "rows": int(len(table)),
            "positive_nonmissing": int(positive.sum()),
            "missing_or_nonfinite": int(missing.sum()),
            "zero": int(zeros.sum()),
            "nanmedian": median,
            "method": "numpy.nanmedian of the shipped size column, matching set_sizes()",
            "quoted_value_that_is_not_the_median": NOT_THE_TABLE_MEDIAN,
            "quoted_value_bodyId": 20129,
            "note": (
                "Every shipped row has a positive size, so set_sizes() does not "
                "substitute the median for a missing cell in this export. "
                "1,010,837,052 is bodyId 20129's size, not the table median."
            ),
        },
        "consensusNt_counts": consensus,
        "membership_rule": (
            "Every wTable row is in the simulation. consensusNt is not a "
            "membership filter. The shipped W is already signed."
        ),
        "outgoing_by_consensusNt": outgoing,
        "W_nnz": int(np.count_nonzero(matrix)),
        "W_n_negative": int(np.count_nonzero(matrix < 0)),
        "W_n_positive": int(np.count_nonzero(matrix > 0)),
        "dng100": dng,
        "stimulated_bodyId": int(MALECNS_DNG100_R),
        "stimulated_instance": dng[str(MALECNS_DNG100_R)]["instance"],
    }


def annotate_authors_rows(body_ids, table: pd.DataFrame | None = None) -> list[dict]:
    """Rows from the shipped wTable. IDs absent from that table are marked absent."""
    if table is None:
        table = load_authors_wtable()
    by_id = table.set_index(table["bodyId"].astype(int))
    rows = []
    for body in body_ids:
        body = int(body)
        if body not in by_id.index:
            rows.append({"bodyId": body, "in_authors_wtable": False})
            continue
        row = by_id.loc[body]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        consensus = row["consensusNt"]
        rows.append(
            {
                "bodyId": body,
                "in_authors_wtable": True,
                "type": None if pd.isna(row["type"]) else str(row["type"]),
                "instance": None if pd.isna(row["instance"]) else str(row["instance"]),
                "superclass": None if pd.isna(row["superclass"]) else str(row["superclass"]),
                "class": None if pd.isna(row["class"]) else str(row["class"]),
                "consensusNt": None if pd.isna(consensus) or str(consensus).strip() == "" else str(consensus),
                "size": None if pd.isna(row["size"]) else float(row["size"]),
                "somaSide": None if pd.isna(row["somaSide"]) else str(row["somaSide"]),
            }
        )
    return rows


def write_export_status(path: Path = EXPORT_STATUS_PATH, summary: dict | None = None) -> dict:
    summary = dict(summary or export_summary())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary

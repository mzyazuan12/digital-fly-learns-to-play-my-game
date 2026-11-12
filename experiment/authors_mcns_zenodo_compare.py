"""Compare this repo's mCNS DNg100 run with the authors' own published run.

Source: Zenodo record 22260924, DNg100_Stim_IMAC_vncOnly.zip
(md5 75779631356162efa7e701847dbbb9a0), run_id=33241780: 1,024 replicates,
seed 626, stimNeurons [[9]] (bodyId 10056, DNg100_R), stimI [[400]].

Checks, in order:
  1. run config      stimulus index/amplitude, seed, solver, T, dt
  2. W               authors' stored W against reweight_connectivity(CSV)
  3. parameters      authors' tau/a/threshold/fr_cap against our sampler at seed 626
  4. input currents  only index 9, value 400, every replicate
  5. population      Figure 2 read-outs on all 1,024 authors' replicates (streamed)
  6. traces          integrate selected replicates with the authors' exact
                     parameters and correlate with their saved R

The 7.3 GB coords array is never loaded whole: four handles read the
replicate, neuron and time rows and the data array in lockstep chunks.
Run from the repository root. The archive is data, not code: it is opened
with zipfile/h5py/numpy only (allow_pickle=False).
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ZIP = ROOT / "data" / "zenodo_pugliese_22260924" / "DNg100_Stim_IMAC_vncOnly.zip"
ZIP_MD5 = "75779631356162efa7e701847dbbb9a0"
RUN = (
    ROOT
    / "data"
    / "zenodo_pugliese_22260924"
    / "extracted"
    / "simulations"
    / "DNg100_Stim_IMAC_vncOnly"
    / "hyak"
    / "run_id=33241780"
)
RS = RUN / "ckpt" / "DNg100_Stim_IMAC_vncOnly_Rs.npz"
PARAMS_H5 = RUN / "ckpt" / "neuron_params.h5"
RUN_CONFIG = RUN / "logs" / "run_config.yaml"
AUTHORS_SEED = 626
AUTHORS_STIM_INDEX = 9


def _header(fh):
    version = np.lib.format.read_magic(fh)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    if fortran:
        raise ValueError("Fortran-ordered array not supported")
    return shape, np.dtype(dtype)


def _skip(fh, nbytes: int, chunk: int = 1 << 26) -> None:
    while nbytes > 0:
        got = len(fh.read(min(chunk, nbytes)))
        if got == 0:
            raise EOFError("unexpected end of stream")
        nbytes -= got


def _read(fh, count: int, dtype) -> np.ndarray:
    nbytes = int(count) * np.dtype(dtype).itemsize
    buf = bytearray()
    while len(buf) < nbytes:
        piece = fh.read(nbytes - len(buf))
        if not piece:
            raise EOFError("unexpected end of stream")
        buf += piece
    return np.frombuffer(bytes(buf), dtype=dtype)


def stream_replicates(path: Path = RS, chunk: int = 4_000_000):
    """Yield (replicate, dense float32 (n_neurons, n_times)) in file order."""
    z = zipfile.ZipFile(path)
    with z.open("shape.npy") as fh:
        shape = tuple(int(x) for x in np.lib.format.read_array(fh, allow_pickle=False))
    _, n_rep, n_neu, n_t = shape
    with z.open("data.npy") as fh:
        (nnz,), ddtype = _header(fh)
    handles = []
    for row in (1, 2, 3):
        fh = z.open("coords.npy")
        (ndim, nnz_c), cdtype = _header(fh)
        if ndim != 4 or nnz_c != nnz:
            raise ValueError("unexpected coords layout")
        _skip(fh, row * nnz * cdtype.itemsize)
        handles.append(fh)
    hd = z.open("data.npy")
    _header(hd)
    current = None
    dense = None
    done = 0
    last_rep = -1
    while done < nnz:
        m = min(chunk, nnz - done)
        rep = _read(handles[0], m, cdtype)
        neu = _read(handles[1], m, cdtype)
        tt = _read(handles[2], m, cdtype)
        val = _read(hd, m, ddtype)
        if rep[0] < last_rep or np.any(np.diff(rep) < 0):
            raise ValueError("coords are not sorted by replicate")
        last_rep = int(rep[-1])
        for r in np.unique(rep):
            sel = rep == r
            if current is not None and r != current:
                yield current, dense
                dense = None
            if dense is None:
                dense = np.zeros((n_neu, n_t), dtype=np.float32)
                current = int(r)
            dense[neu[sel], tt[sel]] = val[sel]
        done += m
    if dense is not None:
        yield current, dense
    for fh in handles + [hd]:
        fh.close()


def compare(out: Path, trace_replicates=(0, 1, 2, 3, 4, 5, 6, 7, 954), workers: int = 9) -> dict:
    import h5py
    import yaml

    from experiment.pugliese_rate_mcns_dng100 import _authors_score, paper_metrics, role_metrics
    from flybrain.authors_mcns import load_authors_W, load_authors_wtable
    from flybrain.pugliese_rate import (
        PuglieseRateParams,
        rates_are_valid,
        reweight_connectivity,
        sample_cell_parameters_block,
        set_sizes,
    )
    from organism.roi_innervation import CORE_CPG_TYPES

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    report: dict = {
        "source": "Zenodo 22260924 DNg100_Stim_IMAC_vncOnly.zip run_id=33241780",
        "zip_md5_expected": ZIP_MD5,
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # 1. config
    cfg = yaml.safe_load(RUN_CONFIG.read_text())
    report["config"] = {
        "stimNeurons": cfg["experiment"]["stimNeurons"],
        "stimI": cfg["experiment"]["stimI"],
        "seed": cfg["experiment"]["seed"],
        "n_replicates": cfg["experiment"]["n_replicates"],
        "sim": {k: cfg["sim"][k] for k in ("T", "dt", "pulseStart", "pulseEnd", "rtol", "atol")},
        "neuron_params": cfg["neuron_params"],
        "wPath_tail": cfg["experiment"]["wPath"].split("data/")[-1],
    }
    assert cfg["experiment"]["stimNeurons"] == [[AUTHORS_STIM_INDEX]]
    assert cfg["experiment"]["stimI"] == [[400]]
    assert cfg["experiment"]["seed"] == AUTHORS_SEED

    table = load_authors_wtable()
    body = table["bodyId"].to_numpy(dtype=np.int64)
    report["config"]["stimulated_bodyId"] = int(body[AUTHORS_STIM_INDEX])
    params = PuglieseRateParams(stim_amplitude=400.0)

    # 2. W
    Wcsv, _, _ = load_authors_W()
    weighted = reweight_connectivity(Wcsv, params.exc_multiplier, params.inh_multiplier)
    with h5py.File(PARAMS_H5, "r") as f:
        Wa = np.asarray(f["W"], dtype=np.float64)
        mn_idxs_h5 = np.asarray(f["mn_idxs"]).astype(int)
        inp = f["input_currents"]
        nz_cols = set()
        vals = set()
        for lo in range(0, inp.shape[1], 128):
            block = np.asarray(inp[0, lo : lo + 128])
            r, c = np.nonzero(block)
            nz_cols.update(int(x) for x in np.unique(c))
            vals.update(float(x) for x in np.unique(block[r, c]))
            if np.count_nonzero(block) != block.shape[0]:
                raise AssertionError("each replicate must have exactly one stimulated cell")
        report["input_currents"] = {"nonzero_indices": sorted(nz_cols), "values": sorted(vals)}
        cand = {"weighted_post_pre": weighted, "weighted_pre_post": weighted.T, "raw_pre_post": Wcsv, "raw_post_pre": Wcsv.T}
        report["W"] = {k: float(np.abs(Wa - v).max()) for k, v in cand.items()}
        report["W"]["best_match"] = min((k for k in cand), key=lambda k: report["W"][k])
        # 3. parameters
        drawn = sample_cell_parameters_block(len(table), 1024, AUTHORS_SEED, params)
        gain, threshold = set_sizes(table["size"].to_numpy(np.float64), drawn["gain"], drawn["threshold"])
        ours = {"tau": drawn["tau"], "a": gain, "threshold": threshold, "fr_cap": drawn["fr_cap"]}
        ours_raw = {"a": drawn["gain"], "threshold": drawn["threshold"]}
        pcmp = {}
        authors_params = {}
        for k in ("tau", "a", "threshold", "fr_cap"):
            A = np.asarray(f[k], dtype=np.float64)
            authors_params[k] = A
            rel = np.abs(A - ours[k]) / np.maximum(np.abs(A), 1e-12)
            pcmp[k] = {"max_rel_diff_vs_ours_after_set_sizes": float(rel.max())}
            if k in ours_raw:
                rel_raw = np.abs(A - ours_raw[k]) / np.maximum(np.abs(A), 1e-12)
                pcmp[k]["max_rel_diff_vs_ours_raw"] = float(rel_raw.max())
        report["parameters"] = pcmp
        report["mn_idxs_h5_n"] = int(mn_idxs_h5.size)
    mn_idx = np.flatnonzero(table["motor module"].notna().to_numpy())
    report["mn_readout"] = {
        "notebook_rule": "wTable['motor module'].notna()",
        "n": int(mn_idx.size),
        "h5_mn_idxs_n": int(mn_idxs_h5.size),
        "h5_only_bodyIds": body[np.setdiff1d(mn_idxs_h5, mn_idx)].tolist(),
    }
    save()

    # 5. population metrics on all authors' replicates (streamed)
    score_fn = _authors_score()
    types = table["type"].astype(str)
    role_idx = {r: np.flatnonzero(types.eq(t).to_numpy()) for r, t in CORE_CPG_TYPES.items()}
    rows = []
    kept = {}
    t0 = time.time()
    for rep, R in stream_replicates():
        R64 = R.astype(np.float64)
        rows.append(
            {
                "replicate": rep,
                "paper": paper_metrics(R64, mn_idx, params.dt, score_fn),
                "roles": {r: role_metrics(R64, ix, params.dt, score_fn) for r, ix in role_idx.items()},
                "dng100_10056_max_hz": float(R64[AUTHORS_STIM_INDEX].max()),
            }
        )
        if rep in trace_replicates:
            kept[rep] = R
        if rep % 128 == 0:
            print(f"authors replicate {rep} ({time.time() - t0:.0f}s)", flush=True)
    from experiment.pugliese_rate_mcns_dng100 import summarize

    for r in rows:
        r["seconds"] = 0.0
        r["rates_valid"] = True
        r["dng100_driven_max_hz"] = r["dng100_10056_max_hz"]
    report["authors_population"] = summarize(rows)
    report["authors_population"]["n_replicates_streamed"] = len(rows)
    (out / "authors_replicates.json").write_text(json.dumps(rows, allow_nan=False) + "\n")
    save()

    # 6. trace-level: our integrator with the authors' exact parameters
    from concurrent.futures import ProcessPoolExecutor

    from experiment.pugliese_rate_mcns_dng100 import _worker_init, _worker_run

    inputs = np.zeros(len(table))
    inputs[AUTHORS_STIM_INDEX] = 400.0
    jobs = [
        {
            "replicate": int(r),
            "tau": authors_params["tau"][r],
            "gain": authors_params["a"][r],
            "threshold": authors_params["threshold"][r],
            "fr_cap": authors_params["fr_cap"][r],
            "inputs": inputs,
            "remove": np.zeros(0, int),
            "shuffle_seed": None,
        }
        for r in sorted(kept)
    ]
    params_kwargs = {k: getattr(params, k) for k in params.__dataclass_fields__}
    traces = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(weighted, params_kwargs)) as pool:
        for res in pool.map(_worker_run, jobs):
            r = res["replicate"]
            ours_R = res["rates"].astype(np.float64)
            A = kept[r].astype(np.float64)
            active = (A.max(axis=1) > 0.01) | (ours_R.max(axis=1) > 0.01)
            mo = paper_metrics(ours_R, mn_idx, params.dt, score_fn)
            ma = paper_metrics(A, mn_idx, params.dt, score_fn)
            traces.append(
                {
                    "replicate": r,
                    "n_active_union": int(active.sum()),
                    "same_active_set": bool(
                        np.array_equal(A.max(axis=1) > 0.01, ours_R.max(axis=1) > 0.01)
                    ),
                    "corrcoef_active": float(np.corrcoef(A[active].ravel(), ours_R[active].ravel())[0, 1]),
                    "max_abs_diff_hz": float(np.abs(A - ours_R).max()),
                    "ours": mo,
                    "authors": ma,
                    "ours_rates_valid": rates_are_valid(ours_R),
                }
            )
            print(f"trace rep {r}: corr={traces[-1]['corrcoef_active']:.6f}", flush=True)
            np.savez_compressed(out / f"trace_{r}.npz", authors=kept[r], ours=res["rates"], mn_index=mn_idx)
    report["traces"] = traces
    report["mean_trace_corrcoef"] = float(np.mean([t["corrcoef_active"] for t in traces]))
    report["min_trace_corrcoef"] = float(np.min([t["corrcoef_active"] for t in traces]))
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / ("authors_mcns_zenodo_compare_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    r = compare(a.out, workers=a.workers)
    print(a.out)
    print(json.dumps({k: r[k] for k in ("W", "parameters", "input_currents", "mean_trace_corrcoef", "min_trace_corrcoef")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

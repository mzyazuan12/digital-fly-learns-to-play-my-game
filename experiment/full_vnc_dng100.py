"""Stage 2A: DNg100 on the authors' full six-leg MANC VNC (open loop).

Network: wTable/W_20251006 (23,532 neurons), sizes normalised by the
nanmedian of that table (set_sizes), same half-tanh rate ODE as Paper 1.
Read-outs: experiment.leg_coordination (pre-registered, Paper 2 protocol).

Saved per replicate: rates of every motor neuron, core CPG cell and DNg100
(float32), not the whole 23,532-row matrix.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CPG_TYPES = {"E1": "IN17A001", "E2": "INXXX466", "E3": "IN19B012", "I1": "IN16B036", "I2": "IN19A007"}

_W = None
_PARAMS = None


def _init(W, params_kwargs):
    global _W, _PARAMS
    from flybrain.pugliese_rate import PuglieseRateParams

    _W = W
    _PARAMS = PuglieseRateParams(**params_kwargs)


def _run(job):
    from flybrain.pugliese_rate import integrate_rate

    t0 = time.time()
    R = integrate_rate(
        _W,
        tau=job["tau"],
        gain=job["gain"],
        threshold=job["threshold"],
        fr_cap=job["fr_cap"],
        inputs=job["inputs"],
        params=_PARAMS,
    )
    keep = job["keep"]
    return {
        "replicate": job["replicate"],
        "seconds": time.time() - t0,
        "kept": R[keep].astype(np.float32),
        "rates_valid": bool(np.all(np.isfinite(R)) and R.min() >= -1e-9 and R.max() <= 1e4),
        "n_active_total": int((R.max(axis=1) > 0.01).sum()),
    }


def run(
    out: Path,
    *,
    version: str = "20251006",
    stim_indices: tuple[int, ...] = (59, 282),
    stim_amplitude: float = 250.0,
    seed: int = 1,
    param_block: int = 128,
    replicate_indices: list[int] | None = None,
    workers: int = 9,
    T: float = 2.0,
) -> dict:
    from experiment.leg_coordination import index_sets, replicate_readouts, summarize
    from experiment.pugliese_rate_mcns_dng100 import _authors_score
    from flybrain.manc_full import check_alignment, leg_of, load_W_csr, load_wtable, reweight_csr
    from flybrain.pugliese_rate import PuglieseRateParams, sample_cell_parameters_block, set_sizes

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    table = load_wtable(version)
    align = check_alignment(table, version)
    if not all(align.values()):
        raise RuntimeError(f"W/wTable misaligned: {align}")
    types = table["type"].astype(str).to_numpy()
    for i in stim_indices:
        if types[i] != "DNg100":
            raise ValueError(f"index {i} is {types[i]}, not DNg100")
    legs = leg_of(table)
    sets = index_sets(table, legs)
    mn = np.flatnonzero(table["motor module"].replace("", None).notna().to_numpy())
    cpg = np.flatnonzero(np.isin(types, list(CPG_TYPES.values())))
    dng = np.flatnonzero(types == "DNg100")
    keep = np.unique(np.concatenate([mn, cpg, dng]))
    remap = {int(g): k for k, g in enumerate(keep)}
    sets_kept = {leg: {k: np.array([remap[int(i)] for i in v]) for k, v in d.items()} for leg, d in sets.items()}

    params = PuglieseRateParams(stim_amplitude=float(stim_amplitude), T=float(T))
    sizes = table["size"].to_numpy(dtype=np.float64)
    if replicate_indices is None:
        replicate_indices = list(range(param_block))
    out.mkdir(parents=True)
    report = {
        "experiment": "stage_2A_full_manc_open_loop",
        "protocol": "docs/papers/paper2_protocol.md",
        "network": f"authors full MANC VNC {version}",
        "n_neurons": int(len(table)),
        "alignment": align,
        "size_median": float(np.nanmedian(sizes)),
        "n_size_missing_or_zero": int(np.sum(~np.isfinite(sizes) | (sizes == 0))),
        "stim_indices": list(map(int, stim_indices)),
        "stim_bodyIds": [int(table["bodyId"].iloc[i]) for i in stim_indices],
        "stim_amplitude": float(stim_amplitude),
        "seed": int(seed),
        "param_block": int(param_block),
        "replicate_indices": [int(i) for i in replicate_indices],
        "T_s": float(T),
        "integrator": "scipy RK45, authors rtol/atol, CSR W",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        report["git_commit"] = None

    def save():
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    drawn = sample_cell_parameters_block(len(table), int(param_block), int(seed), params)
    gain, threshold = set_sizes(sizes, drawn["gain"], drawn["threshold"])
    report["parameter_sampler"] = drawn["sampler"]
    W = reweight_csr(load_W_csr(version), params.exc_multiplier, params.inh_multiplier)
    inputs = np.zeros(len(table))
    inputs[list(stim_indices)] = float(stim_amplitude)
    jobs = [
        {
            "replicate": int(i),
            "tau": drawn["tau"][i],
            "gain": gain[i],
            "threshold": threshold[i],
            "fr_cap": drawn["fr_cap"][i],
            "inputs": inputs,
            "keep": keep,
        }
        for i in replicate_indices
    ]
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"
    score_fn = _authors_score()
    rows, e1s, kept = [], [], []
    pk = {k: getattr(params, k) for k in params.__dataclass_fields__}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(W, pk)) as pool:
        for res in pool.map(_run, jobs):
            r = replicate_readouts(res["kept"].astype(np.float64), sets_kept, score_fn)
            e1s.append(r.pop("_e1"))
            r.update(replicate=res["replicate"], seconds=res["seconds"], rates_valid=res["rates_valid"], n_active_total=res["n_active_total"])
            rows.append(r)
            kept.append(res["kept"])
            legs_txt = " ".join(f"{l}:{r['legs'][l]['score']:.2f}" for l in r["legs"])
            print(f"rep {res['replicate']} {res['seconds']:.0f}s active={res['n_active_total']} {legs_txt}", flush=True)
    np.savez_compressed(out / "kept_rates.npz", rates=np.stack(kept), keep_index=keep, replicate=np.array(replicate_indices))
    report["replicates"] = rows
    report["summary"] = summarize(rows, e1s)
    report["status"] = "done"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    return report


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--stim-indices", default="59,282")
    p.add_argument("--stim-amplitude", type=float, default=250.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--param-block", type=int, default=128)
    p.add_argument("--replicates", default=None, help="start:stop")
    p.add_argument("--workers", type=int, default=9)
    a = p.parse_args(argv)
    reps = None
    if a.replicates:
        lo, hi = map(int, a.replicates.split(":"))
        reps = list(range(lo, hi))
    out = a.out or ROOT / "outputs" / ("full_manc_dng100_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    r = run(
        out,
        stim_indices=tuple(int(x) for x in a.stim_indices.split(",")),
        stim_amplitude=a.stim_amplitude,
        seed=a.seed,
        param_block=a.param_block,
        replicate_indices=reps,
        workers=a.workers,
    )
    print(out)
    s = r["summary"]
    print(json.dumps({"legs": s["legs"], "coupled_pairs": [k for k, v in s["pairs"].items() if v["coupled_pass"]]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

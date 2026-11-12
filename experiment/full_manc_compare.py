"""Stage 2A check against the authors' archived full-MANC runs.

Zenodo 22260924 DNg100_Stim_fullManc.zip (md5 0f3d47ef1f6d33421e6f53ee25e74120):
four Hydra runs x 32 replicates, stimNeurons [[59, 282]] (both DNg100),
stimI 380, seeds 641 / 98 / 445 / 424, W/wTable_20251006.

  1. parameters: our sampler + set_sizes vs each run's neuron_params.h5
  2. traces: integrate a few replicates with the authors' parameters, correlate
  3. read-outs: the pre-registered coordination summary on all 128 replicates
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / "data" / "zenodo_pugliese_22260924" / "fullManc" / "extracted" / "simulations" / "DNg100_Stim_fullManc" / "hyak"
RUNS = {"30662613": 641, "33195857": 98, "33195876": 445, "33195882": 424}
STIM = (59, 282)
STIM_I = 380.0


def main(out: Path, trace_reps=(0, 1, 2, 3), workers: int = 4) -> dict:
    import h5py
    import yaml

    from experiment.authors_mcns_zenodo_compare import stream_replicates
    from experiment.full_vnc_dng100 import CPG_TYPES, _init, _run
    from experiment.leg_coordination import index_sets, replicate_readouts, summarize
    from experiment.pugliese_rate_mcns_dng100 import _authors_score
    from flybrain.manc_full import leg_of, load_W_csr, load_wtable, reweight_csr
    from flybrain.pugliese_rate import PuglieseRateParams, sample_cell_parameters_block, set_sizes

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    table = load_wtable()
    sizes = table["size"].to_numpy(np.float64)
    types = table["type"].astype(str).to_numpy()
    legs = leg_of(table)
    sets = index_sets(table, legs)
    params = PuglieseRateParams(stim_amplitude=STIM_I)
    score_fn = _authors_score()
    report: dict = {"source": "Zenodo 22260924 DNg100_Stim_fullManc", "runs": {}}
    rows, e1s = [], []
    first_params = None
    for run_id, seed in RUNS.items():
        rdir = BASE / f"run_id={run_id}"
        cfg = yaml.safe_load((rdir / "logs" / "run_config.yaml").read_text())
        assert cfg["experiment"]["stimNeurons"] == [list(STIM)]
        assert float(cfg["experiment"]["stimI"][0]) == STIM_I and cfg["experiment"]["seed"] == seed
        drawn = sample_cell_parameters_block(len(table), 32, seed, params)
        gain, thr = set_sizes(sizes, drawn["gain"], drawn["threshold"])
        ours = {"tau": drawn["tau"], "a": gain, "threshold": thr, "fr_cap": drawn["fr_cap"]}
        pc = {}
        with h5py.File(rdir / "ckpt" / "neuron_params.h5", "r") as f:
            inp = np.asarray(f["input_currents"])
            nz = sorted(set(np.nonzero(inp.reshape(-1, inp.shape[-1]))[1].tolist()))
            vals = sorted(set(np.unique(inp[inp != 0]).tolist()))
            auth = {}
            for k in ours:
                A = np.asarray(f[k], dtype=np.float64)
                auth[k] = A
                pc[k] = float((np.abs(A - ours[k]) / np.maximum(np.abs(A), 1e-12)).max())
        if first_params is None:
            first_params = auth
        report["runs"][run_id] = {"seed": seed, "param_max_rel_diff": pc, "input_nonzero_indices": nz, "input_values": vals}
        print(run_id, pc, nz, vals, flush=True)
        for rep, R in stream_replicates(rdir / "ckpt" / "DNg100_Stim_fullManc_Rs.npz"):
            r = replicate_readouts(R.astype(np.float64), sets, score_fn)
            e1s.append(r.pop("_e1"))
            r.update(run_id=run_id, replicate=rep, n_active_total=int((R.max(axis=1) > 0.01).sum()))
            rows.append(r)
            if run_id == "30662613" and rep in trace_reps:
                np.save(out / f"authors_rep{rep}.npy", R)
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["authors_summary"] = summarize(rows, e1s)
    report["authors_replicates"] = rows

    # trace-level: our integrator with run 30662613's parameters
    W = reweight_csr(load_W_csr(), params.exc_multiplier, params.inh_multiplier)
    inputs = np.zeros(len(table))
    inputs[list(STIM)] = STIM_I
    keep = np.arange(len(table))
    jobs = [
        {
            "replicate": int(r),
            "tau": first_params["tau"][r],
            "gain": first_params["a"][r],
            "threshold": first_params["threshold"][r],
            "fr_cap": first_params["fr_cap"][r],
            "inputs": inputs,
            "keep": keep,
        }
        for r in trace_reps
    ]
    pk = {k: getattr(params, k) for k in params.__dataclass_fields__}
    traces = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(W, pk)) as pool:
        for res in pool.map(_run, jobs):
            A = np.load(out / f"authors_rep{res['replicate']}.npy").astype(np.float64)
            O = res["kept"].astype(np.float64)
            act = (A.max(axis=1) > 0.01) | (O.max(axis=1) > 0.01)
            traces.append(
                {
                    "replicate": res["replicate"],
                    "n_active_union": int(act.sum()),
                    "same_active_set": bool(np.array_equal(A.max(axis=1) > 0.01, O.max(axis=1) > 0.01)),
                    "corrcoef_active": float(np.corrcoef(A[act].ravel(), O[act].ravel())[0, 1]),
                    "max_abs_diff_hz": float(np.abs(A - O).max()),
                }
            )
            print("trace", traces[-1], flush=True)
            (out / f"authors_rep{res['replicate']}.npy").unlink()
    report["traces"] = traces
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    o = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / ("full_manc_compare_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    r = main(o)
    s = r["authors_summary"]
    print(json.dumps({"legs": s["legs"], "pairs": {k: {kk: v[kk] for kk in ("R", "null_R_p99", "median_plv", "coupled_pass")} for k, v in s["pairs"].items()}, "traces": r["traces"]}, indent=1))

"""Gate G1 (stage 2B design §2.3): RK4 stepper vs validated integrate_rate.

Replicates 0-7, seed 1, 1,024-block parameters, authors' 4,310-cell mCNS
network, I = 400, for DNg100_R alone and for both DNg100. Pass requires,
per replicate: min Pearson r >= 0.999 over neurons active in either run
(t >= 250 ms); identical active set and active MN set; MN frequency within
0.1 Hz; left-leg stance-vs-swing lag within 0.01 cycle (when measurable).
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = {"right": (9,), "both": (6, 9)}
_CTX = {}


def _setup():
    from scipy.sparse import csr_matrix

    from flybrain.authors_mcns import load_authors_W, load_authors_wtable
    from flybrain.pugliese_rate import PuglieseRateParams, reweight_connectivity, sample_cell_parameters_block, set_sizes

    t = load_authors_wtable()
    p = PuglieseRateParams(stim_amplitude=400.0)
    d = sample_cell_parameters_block(len(t), 1024, 1, p)
    g, th = set_sizes(t["size"].to_numpy(float), d["gain"], d["threshold"])
    W, _, _ = load_authors_W()
    w = csr_matrix(reweight_connectivity(W, p.exc_multiplier, p.inh_multiplier))
    return t, p, d, g, th, w


def _job(args):
    arm, rep = args
    if not _CTX:
        _CTX["v"] = _setup()
    t, p, d, g, th, w = _CTX["v"]
    from flybrain.pugliese_rate import integrate_rate
    from flybrain.rate_stepper import RateStepper

    inp = np.zeros(len(t))
    inp[list(ARMS[arm])] = 400.0
    kw = dict(tau=d["tau"][rep], gain=g[rep], threshold=th[rep], fr_cap=d["fr_cap"][rep])
    # Amendment A2: reference at rtol 1e-9 / atol 1e-11 (the authors' 2e-6 / 5e-9
    # reference carries ~0.003 correlation error on near-flat neurons).
    from flybrain.pugliese_rate import PuglieseRateParams

    p_ref = PuglieseRateParams(stim_amplitude=400.0, rtol=1e-9, atol=1e-11)
    ref = integrate_rate(w, inputs=inp, params=p_ref, **kw)
    st = RateStepper(W=w, dn_input=inp, h=1e-4, **kw).run(T=2.0)
    return arm, rep, ref.astype(np.float32), st.astype(np.float32)


def main(out: Path) -> dict:
    from experiment.leg_coordination import STANCE, SWING, lag_cycles, module_trace, period_autocorr
    from experiment.pugliese_rate_mcns_dng100 import _authors_score, paper_metrics
    from flybrain.authors_mcns import load_authors_wtable

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    t = load_authors_wtable()
    mn = np.flatnonzero(t["motor module"].notna().to_numpy())
    side = t["somaSide"].astype(str).to_numpy()
    mod = t["motor module"]
    swing = np.flatnonzero(mod.eq(SWING).to_numpy() & (side == "L"))
    stance = np.flatnonzero(mod.eq(STANCE).to_numpy() & (side == "L"))
    score = _authors_score()
    rows = []
    jobs = [(a, r) for a in ARMS for r in range(8)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for arm, rep, ref, st in pool.map(_job, jobs):
            ref = ref.astype(float)
            st = st.astype(float)
            a_ref = ref.max(axis=1) > 0.01
            a_st = st.max(axis=1) > 0.01
            act = np.flatnonzero(a_ref | a_st)
            rs = []
            for i in act:
                x, y = ref[i, 250:], st[i, 250:]
                if x.std() < 1e-9 and y.std() < 1e-9:
                    rs.append(1.0 if np.allclose(x, y, atol=1e-3) else 0.0)
                else:
                    rs.append(float(np.corrcoef(x, y)[0, 1]))
            pm_ref = paper_metrics(ref, mn, 0.001, score)
            pm_st = paper_metrics(st, mn, 0.001, score)
            lags = []
            for R in (ref, st):
                past = R[:, 250:]
                sw, sn = module_trace(past, swing), module_trace(past, stance)
                lags.append(lag_cycles(sw, sn, period_autocorr(sw)) if sw is not None and sn is not None else None)
            dlag = None if None in lags else abs(((lags[0] - lags[1]) + 0.5) % 1.0 - 0.5)
            fr, fs = pm_ref["mn_frequency_hz"], pm_st["mn_frequency_hz"]
            row = {
                "arm": arm,
                "replicate": rep,
                "n_active_union": int(act.size),
                "min_r": float(np.min(rs)),
                "mean_r": float(np.mean(rs)),
                "same_active_set": bool(np.array_equal(a_ref, a_st)),
                "same_active_mn_set": bool(np.array_equal(a_ref[mn], a_st[mn])),
                "freq_ref_hz": fr,
                "freq_step_hz": fs,
                "freq_diff_hz": None if (fr is None or fs is None) else abs(fr - fs),
                "lag_ref": lags[0],
                "lag_step": lags[1],
                "lag_diff_cycles": dlag,
                "max_abs_diff_hz": float(np.abs(ref - st).max()),
            }
            row["pass"] = bool(
                row["min_r"] >= 0.999
                and row["same_active_set"]
                and row["same_active_mn_set"]
                and (row["freq_diff_hz"] is None or row["freq_diff_hz"] <= 0.1)
                and (dlag is None or dlag <= 0.01)
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    report = {
        "gate": "G1",
        "design": "docs/papers/paper2_stage2B_design.md §2.3",
        "stepper": "flybrain/rate_stepper.py RK4 h=0.1 ms",
        "reference": "integrate_rate RK45 rtol 1e-9 atol 1e-11 (amendment A2)",
        "rows": rows,
        "pass": bool(all(r["pass"] for r in rows)),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    o = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / ("paper2_gate_g1_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    r = main(o)
    print("G1 pass:", r["pass"])

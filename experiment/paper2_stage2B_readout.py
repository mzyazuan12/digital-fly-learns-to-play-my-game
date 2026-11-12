"""Stage 2B step 10: pre-registered read-outs on the runner's saved traces.

Uses experiment.leg_coordination unchanged (lag_cycles, period_autocorr, plv,
circ_stats, null_pairs) and the authors' oscillation score.

Within-leg (protocol (a), design F9): lag of the coxa-stance module mean
z-trace behind coxa-swing, in cycles, after 250 ms. A replicate is
measurable only if both modules are active and each has an authors' score
>= 0.5 with frequencies within 1 Hz. Unmeasurable replicates count as not
passing: the population resultant R_all is |sum of unit vectors over
measurable replicates| / N_total. Pass: circular mean in [0.375, 0.625],
R_all >= 0.8, N_total >= 64.

Between legs (protocol (b)): E1_L vs E1_R lag and within-run PLV;
cross-replicate null (200 draws). Pass: R >= 0.8, R > null p99, median PLV >= 0.8.

A positive result requires a pass in closed loop and a fail in every
control (open loop, deafferented, scrambled, mismatched) for that cell.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.leg_coordination import circ_stats, lag_cycles, null_pairs, period_autocorr, plv, zscore  # noqa: E402

TRANSIENT = 250
ACTIVE = 0.01


def _module(R, labels, module, side):
    idx = [i for i, l in enumerate(labels) if l.startswith(f"MN|{module}|{side}|")]
    act = [i for i in idx if R[i, TRANSIENT:].max() > ACTIVE]
    return act


def within_leg(R, labels, side, score_fn):
    import jax.numpy as jnp

    past = R[:, TRANSIENT:]
    out = {"measurable": False, "lag": None}
    mods = {}
    for m in ("coxa swing", "coxa stance"):
        act = _module(R, labels, m, side)
        if not act:
            out["reason"] = f"{m} inactive"
            return out
        sc, fr = score_fn(jnp.asarray(past[act]), jnp.ones(len(act), dtype=bool))
        sc, fr = float(sc), float(fr)
        mods[m] = {
            "trace": np.mean([zscore(past[i]) for i in act], axis=0),
            "score": sc if np.isfinite(sc) else 0.0,
            "freq_hz": fr / 0.001 if np.isfinite(fr) and fr > 0 else None,
        }
    sw, st = mods["coxa swing"], mods["coxa stance"]
    out.update(score_swing=sw["score"], score_stance=st["score"], freq_swing=sw["freq_hz"], freq_stance=st["freq_hz"])
    if sw["score"] < 0.5 or st["score"] < 0.5:
        out["reason"] = "module score < 0.5"
        return out
    if sw["freq_hz"] is None or st["freq_hz"] is None or abs(sw["freq_hz"] - st["freq_hz"]) > 1.0:
        out["reason"] = "module frequencies differ > 1 Hz"
        return out
    p = period_autocorr(sw["trace"])
    lag = lag_cycles(sw["trace"], st["trace"], p)
    if lag is None:
        out["reason"] = "no period"
        return out
    out.update(measurable=True, lag=lag, period_samples=p)
    return out


def e1_traces(R, labels):
    out = {}
    for side, leg in (("L", "T1L"), ("R", "T1R")):
        i = labels.index(f"CPG|E1|{side}")
        if R[i, TRANSIENT:].max() > ACTIVE:
            out[leg] = R[i, TRANSIENT:]
    return out


def summarize_within(rows, n_total):
    lags = [r["lag"] for r in rows if r["measurable"]]
    st = circ_stats(lags)
    vec = np.sum(np.exp(2j * np.pi * np.asarray(lags))) if lags else 0
    st["n_total"] = n_total
    st["n_measurable"] = len(lags)
    st["R_all"] = float(np.abs(vec) / n_total) if n_total else None
    st["pass"] = bool(
        n_total >= 64
        and st["mean_cycles"] is not None
        and 0.375 <= st["mean_cycles"] <= 0.625
        and st["R_all"] >= 0.8
    )
    reasons = defaultdict(int)
    for r in rows:
        if not r["measurable"]:
            reasons[r.get("reason", "?")] += 1
    st["unmeasurable_reasons"] = dict(reasons)
    return st


def summarize_between(e1_list, seed=0, n_null=200):
    lags, plvs = [], []
    for e in e1_list:
        if "T1L" in e and "T1R" in e:
            p = period_autocorr(e["T1L"])
            f = 1.0 / (p * 0.001) if np.isfinite(p) else float("nan")
            lags.append(lag_cycles(e["T1L"], e["T1R"], p))
            v = plv(e["T1L"], e["T1R"], f)
            if v is not None:
                plvs.append(v)
    st = circ_stats(lags)
    st["median_plv"] = float(np.median(plvs)) if plvs else None
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        nr = null_pairs(e1_list, rng).get("T1L-T1R")
        if nr:
            nulls.append(circ_stats(nr["lag"])["R"])
    nulls = np.array([x for x in nulls if x is not None])
    st["null_R_p99"] = float(np.percentile(nulls, 99)) if nulls.size else None
    st["pass"] = bool(
        st["R"] is not None
        and st["R"] >= 0.8
        and st["null_R_p99"] is not None
        and st["R"] > st["null_R_p99"]
        and st["median_plv"] is not None
        and st["median_plv"] >= 0.8
    )
    return st


def readout(run_dir: Path) -> dict:
    from experiment.pugliese_rate_mcns_dng100 import _authors_score

    score_fn = _authors_score()
    files = sorted((Path(run_dir) / "runs").glob("rep*_*.npz"))
    groups: dict = defaultdict(lambda: {"L": [], "R": [], "e1": [], "freq": []})
    meta0 = None
    for f in files:
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        meta0 = meta0 or meta
        cell = meta.get("cell")
        key = meta["condition"] + ("" if not cell else ":" + f"{cell['hair_plate_coxa']}-{cell['hair_plate_trochanter']}")
        R = d["rates_hz"].astype(np.float64)
        labels = [str(x) for x in d["labels"]]
        g = groups[key]
        g["L"].append(within_leg(R, labels, "L", score_fn))
        g["R"].append(within_leg(R, labels, "R", score_fn))
        g["e1"].append(e1_traces(R, labels))
    out = {
        "run_dir": str(run_dir),
        "arm": meta0["arm"] if meta0 else None,
        "frozen_sha256": meta0.get("frozen_sha256") if meta0 else None,
        "conditions": {},
    }
    for key, g in sorted(groups.items()):
        n = len(g["L"])
        out["conditions"][key] = {
            "n": n,
            "within_T1L": summarize_within(g["L"], n),
            "within_T1R": summarize_within(g["R"], n),
            "between_T1L_T1R": summarize_between(g["e1"]),
        }
    # positive-result verdict per cell
    verdict = {}
    cells = sorted({k.split(":", 1)[1] for k in out["conditions"] if ":" in k})
    C = out["conditions"]
    for cell in cells:
        v = {}
        for test in ("within_T1L", "within_T1R", "between_T1L_T1R"):
            closed = C.get(f"closed_loop:{cell}", {}).get(test, {}).get("pass", False)
            controls = {
                "open_loop": C.get("open_loop", {}).get(test, {}).get("pass"),
                "deafferented": C.get("deafferented", {}).get(test, {}).get("pass"),
                "scrambled": C.get(f"scrambled:{cell}", {}).get(test, {}).get("pass"),
                "mismatched": C.get(f"mismatched:{cell}", {}).get(test, {}).get("pass"),
            }
            v[test] = {
                "closed_loop_pass": closed,
                "controls_pass": controls,
                "positive": bool(closed and not any(x for x in controls.values() if x is not None)),
            }
        verdict[cell] = v
    out["verdict"] = verdict
    out["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    dest = Path(sys.argv[2])
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    r = readout(run_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    print(dest)
    for k, c in r["conditions"].items():
        a, b, e = c["within_T1L"], c["within_T1R"], c["between_T1L_T1R"]
        fm = lambda x: "—" if x is None else f"{x:.2f}"
        print(
            f"{k:42s} T1L mean={fm(a['mean_cycles'])} Rall={fm(a['R_all'])} meas={a['n_measurable']}/{a['n_total']} {'PASS' if a['pass'] else 'fail'} | "
            f"T1R mean={fm(b['mean_cycles'])} Rall={fm(b['R_all'])} meas={b['n_measurable']} {'PASS' if b['pass'] else 'fail'} | "
            f"L-R R={fm(e['R'])} null99={fm(e['null_R_p99'])} PLV={fm(e['median_plv'])} {'PASS' if e['pass'] else 'fail'}"
        )
    print(json.dumps(r["verdict"], indent=1))

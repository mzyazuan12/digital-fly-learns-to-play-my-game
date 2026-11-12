"""Pre-registered coordination read-outs (docs/papers/paper2_protocol.md).

Input per replicate: a rate matrix R (n_neurons x n_times, dt = 1 ms) plus
the network table. The same functions score our traces and the authors'.

  within-leg phase   lag of coxa-stance (remotor) mean z-trace behind
                     coxa-swing (promotor), in cycles [0, 1)
  between-leg phase  lag of leg j's E1 behind leg i's E1, in cycles
  PLV                |mean(exp(i(phi_i - phi_j)))| of Hilbert phases of
                     band-passed E1 traces (rhythm freq +/- 3 Hz)
  null               leg j taken from a different replicate (authors' method)
"""

from __future__ import annotations

import numpy as np

TRANSIENT = 250
ACTIVE = 0.01
LEGS = ["T1L", "T1R", "T2L", "T2R", "T3L", "T3R"]
SWING = "coxa swing"
STANCE = "coxa stance"


def zscore(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)


def period_autocorr(x: np.ndarray, min_lag: int = 20, max_lag: int = 500) -> float:
    """First autocorrelation peak after the zero-lag lobe, in samples (nan if none)."""
    x = zscore(x)
    if not np.any(x):
        return float("nan")
    n = x.size
    ac = np.correlate(x, x, mode="full")[n - 1 : n - 1 + max_lag + 1] / n
    for k in range(max(min_lag, 2), min(max_lag, ac.size - 1)):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.2:
            return float(k)
    return float("nan")


def lag_cycles(a: np.ndarray, b: np.ndarray, period: float) -> float | None:
    """Lag of b behind a, in cycles [0, 1), from the xcorr peak within one period."""
    if not np.isfinite(period) or period < 2:
        return None
    a = zscore(a)
    b = zscore(b)
    if not np.any(a) or not np.any(b):
        return None
    n = a.size
    lags = np.arange(0, int(round(period)) + 1)
    cc = np.array([np.dot(a[: n - k], b[k:]) / (n - k) for k in lags])
    return float((lags[int(np.argmax(cc))] / period) % 1.0)


def plv(a: np.ndarray, b: np.ndarray, f_hz: float, dt: float = 0.001) -> float | None:
    from scipy.signal import butter, filtfilt, hilbert

    if not np.isfinite(f_hz) or f_hz <= 3.5:
        return None
    a = zscore(a)
    b = zscore(b)
    if not np.any(a) or not np.any(b):
        return None
    nyq = 0.5 / dt
    lo, hi = (f_hz - 3.0) / nyq, (f_hz + 3.0) / nyq
    bb, aa = butter(2, [lo, min(hi, 0.99)], btype="band")
    pa = np.angle(hilbert(filtfilt(bb, aa, a)))
    pb = np.angle(hilbert(filtfilt(bb, aa, b)))
    return float(np.abs(np.mean(np.exp(1j * (pa - pb)))))


def circ_stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean_cycles": None, "R": None}
    c = np.mean(np.exp(2j * np.pi * v))
    return {"n": int(v.size), "mean_cycles": float((np.angle(c) / (2 * np.pi)) % 1.0), "R": float(np.abs(c))}


def index_sets(table, legs: np.ndarray) -> dict:
    """Row indices per leg: motor neurons by module, and E1."""
    module = table["motor module"].replace("", None)
    types = table["type"].astype(str).to_numpy()
    out = {}
    for leg in LEGS:
        on_leg = legs == leg
        out[leg] = {
            "mn": np.flatnonzero(on_leg & module.notna().to_numpy()),
            SWING: np.flatnonzero(on_leg & module.eq(SWING).to_numpy()),
            STANCE: np.flatnonzero(on_leg & module.eq(STANCE).to_numpy()),
            "E1": np.flatnonzero(on_leg & (types == "IN17A001")),
        }
    return out


def module_trace(past: np.ndarray, idx: np.ndarray) -> np.ndarray | None:
    act = [i for i in idx if past[i].max() > ACTIVE]
    if not act:
        return None
    return np.mean([zscore(past[i]) for i in act], axis=0)


def replicate_readouts(R: np.ndarray, sets: dict, score_fn=None, dt: float = 0.001) -> dict:
    past = R[:, TRANSIENT:].astype(np.float64)
    out = {"legs": {}, "pairs": {}}
    e1 = {}
    for leg in LEGS:
        s = sets[leg]
        act_mn = [i for i in s["mn"] if past[i].max() > ACTIVE]
        row = {"n_mn_active": len(act_mn), "score": 0.0, "freq_hz": None}
        if act_mn and score_fn is not None:
            import jax.numpy as jnp

            sub = past[act_mn]
            sc, fr = score_fn(jnp.asarray(sub), jnp.ones(len(act_mn), dtype=bool))
            row["score"] = float(sc) if np.isfinite(float(sc)) else 0.0
            fr = float(fr)
            row["freq_hz"] = fr / dt if np.isfinite(fr) and fr > 0 else None
        sw = module_trace(past, s[SWING])
        st = module_trace(past, s[STANCE])
        period = period_autocorr(sw) if sw is not None else float("nan")
        row["period_samples"] = None if not np.isfinite(period) else period
        row["stance_vs_swing_cycles"] = lag_cycles(sw, st, period) if (sw is not None and st is not None) else None
        if s["E1"].size and past[s["E1"][0]].max() > ACTIVE:
            e1[leg] = past[s["E1"][0]]
        out["legs"][leg] = row
    for i, a in enumerate(LEGS):
        for b in LEGS[i + 1 :]:
            if a in e1 and b in e1:
                p = period_autocorr(e1[a])
                f = 1.0 / (p * dt) if np.isfinite(p) else float("nan")
                out["pairs"][f"{a}-{b}"] = {
                    "lag_cycles": lag_cycles(e1[a], e1[b], p),
                    "plv": plv(e1[a], e1[b], f, dt),
                    "freq_a_hz": f if np.isfinite(f) else None,
                }
    out["_e1"] = e1  # dropped before JSON
    return out


def null_pairs(e1_by_rep: list[dict], rng: np.random.Generator, dt: float = 0.001) -> dict:
    """Leg j from a different replicate than leg i (authors' cross-replicate null)."""
    n = len(e1_by_rep)
    perm = rng.permutation(n)
    for k in range(n):
        if perm[k] == k:
            perm[k] = (k + 1) % n
    res: dict = {}
    for r in range(n):
        A, B = e1_by_rep[r], e1_by_rep[perm[r]]
        for i, a in enumerate(LEGS):
            for b in LEGS[i + 1 :]:
                if a in A and b in B:
                    p = period_autocorr(A[a])
                    f = 1.0 / (p * dt) if np.isfinite(p) else float("nan")
                    res.setdefault(f"{a}-{b}", {"lag": [], "plv": []})
                    res[f"{a}-{b}"]["lag"].append(lag_cycles(A[a], B[b], p))
                    res[f"{a}-{b}"]["plv"].append(plv(A[a], B[b], f, dt))
    return res


def summarize(rows: list[dict], e1_by_rep: list[dict], n_null: int = 200, seed: int = 0) -> dict:
    """Protocol summary: within-leg alternation, pairwise coupling vs null."""
    legs = {}
    for leg in LEGS:
        lags = [r["legs"][leg]["stance_vs_swing_cycles"] for r in rows]
        st = circ_stats(lags)
        st["fraction_replicates_rhythmic"] = float(np.mean([r["legs"][leg]["score"] >= 0.5 for r in rows]))
        st["median_n_mn_active"] = float(np.median([r["legs"][leg]["n_mn_active"] for r in rows]))
        fr = [r["legs"][leg]["freq_hz"] for r in rows if r["legs"][leg]["freq_hz"]]
        st["median_freq_hz"] = float(np.median(fr)) if fr else None
        st["alternation_pass"] = bool(
            st["R"] is not None and st["n"] >= 64 and 0.375 <= st["mean_cycles"] <= 0.625 and st["R"] >= 0.8
        )
        legs[leg] = st
    rng = np.random.default_rng(seed)
    null_R: dict = {}
    null_plv: dict = {}
    for _ in range(n_null):
        nr = null_pairs(e1_by_rep, rng)
        for k, v in nr.items():
            null_R.setdefault(k, []).append(circ_stats(v["lag"])["R"])
            pl = [x for x in v["plv"] if x is not None]
            null_plv.setdefault(k, []).append(float(np.median(pl)) if pl else None)
    pairs = {}
    keys = sorted({k for r in rows for k in r["pairs"]})
    for k in keys:
        lags = [r["pairs"][k]["lag_cycles"] for r in rows if k in r["pairs"]]
        pl = [r["pairs"][k]["plv"] for r in rows if k in r["pairs"] and r["pairs"][k]["plv"] is not None]
        st = circ_stats(lags)
        nR = np.array([x for x in null_R.get(k, []) if x is not None])
        st["null_R_p99"] = float(np.percentile(nR, 99)) if nR.size else None
        st["median_plv"] = float(np.median(pl)) if pl else None
        nP = np.array([x for x in null_plv.get(k, []) if x is not None])
        st["null_median_plv_p99"] = float(np.percentile(nP, 99)) if nP.size else None
        st["coupled_pass"] = bool(
            st["R"] is not None
            and st["R"] >= 0.8
            and st["null_R_p99"] is not None
            and st["R"] > st["null_R_p99"]
            and st["median_plv"] is not None
            and st["median_plv"] >= 0.8
        )
        pairs[k] = st
    return {"legs": legs, "pairs": pairs, "n_replicates": len(rows), "n_null_draws": n_null}

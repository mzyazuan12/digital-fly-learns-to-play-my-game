"""Stage 2B-kappa: fix the afferent synaptic efficacy kappa by an independent criterion (amendment A8).

ENGINEERED parameter: kappa multiplies the outgoing synapses of the 31
included afferents, i.e. their columns in the post x pre weight matrix of
``experiment.paper2_cosim.Network``. It is applied to a copy of a loaded
Network's W; central synapses, the frozen motor bridge (A7) and the
transduction are unchanged.

Set-up (A8):
- Calibration replicates 960-1023 of the seed-1 block only.
- DN drive off. No body.
- Afferents are driven by the transduction (organism.proprio_transduction,
  r* = 100 Hz) of the recorded walking clip, both front legs, in each of the 4
  hair-plate cells. Angles: the clip's 0.1 ms interpolant (FlyGym conversion).
  Velocities: their derivative, computed on the 1 s clip. Both are looped
  (tiled) to 2 s, so the loop seam has an angle step but no velocity spike.
  The one campaniform afferent (right leg) needs a force: as in A3, the
  replayed clip's trochanter-femur actuator force is used, tiled the same way
  (interpretation, flagged).
- Each tick holds the afferent input over one 0.1 ms RK4 step
  (flybrain.rate_stepper), as in the co-simulation.

Criteria (A8):
- A motor neuron is active if its rate exceeds 1 Hz at any time after 250 ms.
- Size terciles per leg: that leg's motor neurons sorted by wTable ``size``
  and split into thirds by rank (numpy.array_split: 23/22/22 left, 21/21/20 right).
- C1: no largest-tercile motor neuron (either leg) is active.
- C2: on each leg, active motor neurons <= one third of that leg's motor neurons.
- A (kappa, cell) passes if C1 and C2 hold together in >= 95% of replicates.
  Requiring them together is the stricter reading of "both must hold"; the
  separate fractions are recorded too.
- kappa* = the largest grid kappa that passes in all 4 cells.
- Effectiveness at kappa*: per cell, the fraction of replicates with >= 1
  active smallest-tercile motor neuron (either leg); effective if >= 50%.
  Overall label: "effective" (all cells), "feedback ineffective" (no cell),
  or "mixed".

Nothing here computes a phase, lag, coherence or coordination read-out
(AST test). The full table for every grid kappa and cell is written to
data/paper2/stage2Bk_kappa_frozen.json. The script refuses to overwrite.

  .venv/bin/python -m experiment.paper2_kappa_calibrate --workers 9
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "paper2" / "stage2Bk_kappa_frozen.json"
AUDIT_DIR = ROOT / "outputs"
GRID = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001)
CALIBRATION_REPLICATES = tuple(range(960, 1024))
SEED = 1
PARAM_BLOCK = 1024
H = 1e-4
T_S = 2.0
TRANSIENT_TICKS = 2500  # 250 ms
RECORD_EVERY = 10
ACTIVE_HZ = 1.0
PASS_FRACTION = 0.95
EFFECTIVE_FRACTION = 0.5
R_STAR_HZ = 100.0
LEGS = ("T1L", "T1R")
BODY_LEG = {"T1L": "lf", "T1R": "rf"}
SIDE_LEG = {"L": "T1L", "R": "T1R"}

_W: dict = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_replicates(reps) -> tuple[int, ...]:
    reps = tuple(int(r) for r in reps)
    bad = [r for r in reps if r not in CALIBRATION_REPLICATES]
    if bad:
        raise ValueError(f"kappa calibration may use only replicates 960-1023; got {bad[:5]}")
    return reps


def check_kappas(kappas) -> tuple[float, ...]:
    ks = tuple(float(k) for k in kappas)
    bad = [k for k in ks if k not in GRID]
    if bad:
        raise ValueError(f"kappa values must be on the A8 grid {GRID}; got {bad}")
    return ks


# ----------------------------------------------------------------------------- network
def afferent_rows() -> np.ndarray:
    st = json.loads((ROOT / "data" / "paper2" / "stage2B_sensory_table.json").read_text())
    rows = np.array(sorted(a["row"] for a in st["afferents"] if a["included_2B"]), dtype=np.int64)
    if rows.size != 31:
        raise RuntimeError(f"expected 31 included afferents, found {rows.size}")
    return rows


def scale_afferent_columns(W, rows: np.ndarray, kappa: float):
    """Copy of the post x pre CSR matrix with the afferents' columns (outgoing synapses) times kappa."""
    from scipy.sparse import csr_matrix

    W = csr_matrix(W, copy=True)
    if float(kappa) != 1.0:
        mask = np.isin(W.indices, rows)
        W.data[mask] = W.data[mask] * float(kappa)
    return W


def kappa_network(net, kappa: float, rows: np.ndarray | None = None, *, dn_off: bool = False):
    """A copy of a loaded paper2_cosim.Network with kappa applied (and optionally no DN drive)."""
    rows = afferent_rows() if rows is None else rows
    kw = {"W": scale_afferent_columns(net.W, rows, kappa)}
    if dn_off:
        kw["dn_input"] = np.zeros_like(net.dn_input)
    return dataclasses.replace(net, **kw)


# ----------------------------------------------------------------------------- motor neurons
def motor_neuron_terciles() -> dict:
    """Per leg: MN rows sorted by wTable size, split into thirds by rank."""
    from flybrain.authors_mcns import load_authors_wtable

    t = load_authors_wtable()
    out = {}
    for side, leg in SIDE_LEG.items():
        idx = np.flatnonzero(t["motor module"].notna().to_numpy() & (t["somaSide"].astype(str).to_numpy() == side))
        size = t["size"].to_numpy(dtype=np.float64)[idx]
        order = idx[np.argsort(size, kind="stable")]
        parts = np.array_split(order, 3)
        out[leg] = {
            "rows": idx.tolist(),
            "smallest": parts[0].tolist(),
            "middle": parts[1].tolist(),
            "largest": parts[2].tolist(),
            "size_bounds": [float(t["size"].iloc[parts[0][-1]]), float(t["size"].iloc[parts[1][-1]])],
            "bodyIds": {k: t["bodyId"].to_numpy(np.int64)[p].tolist() for k, p in zip(("smallest", "middle", "largest"), parts)},
        }
    return out


def run_criteria(mn_max: dict[int, float], terciles: dict) -> dict:
    """C1, C2 and effectiveness inputs for one run. ``mn_max``: row -> max rate after 250 ms."""
    active = {r for r, v in mn_max.items() if v > ACTIVE_HZ}
    row = {}
    c1 = True
    c2 = True
    any_small = False
    for leg in LEGS:
        tl = terciles[leg]
        n = len(tl["rows"])
        n_act = sum(r in active for r in tl["rows"])
        n_large = sum(r in active for r in tl["largest"])
        n_small = sum(r in active for r in tl["smallest"])
        row[leg] = {"n_mn": n, "n_active": n_act, "n_active_largest": n_large, "n_active_smallest": n_small,
                    "n_active_middle": sum(r in active for r in tl["middle"])}
        c1 &= n_large == 0
        c2 &= n_act <= n / 3.0
        any_small |= n_small > 0
    row.update(C1=bool(c1), C2=bool(c2), both=bool(c1 and c2), smallest_active=bool(any_small))
    return row


def summarise(rows: list[dict], kappas, cells) -> dict:
    """Full table per kappa and cell, kappa* and the effectiveness label."""
    table = {}
    for k in kappas:
        table[f"{k:g}"] = {}
        for c in cells:
            rs = [r for r in rows if r["kappa"] == k and r["cell"] == c and r["status"] == "ok"]
            n_err = sum(1 for r in rows if r["kappa"] == k and r["cell"] == c and r["status"] != "ok")
            n = len(rs)
            frac = lambda key: (sum(r[key] for r in rs) / n) if n else None  # noqa: E731
            entry = {
                "n_replicates": n,
                "n_failed_runs": n_err,
                "C1_fraction": frac("C1"),
                "C2_fraction": frac("C2"),
                "both_fraction": frac("both"),
                "smallest_active_fraction": frac("smallest_active"),
                "median_active": {leg: float(np.median([r[leg]["n_active"] for r in rs])) if n else None for leg in LEGS},
                "max_active": {leg: int(max(r[leg]["n_active"] for r in rs)) if n else None for leg in LEGS},
                "median_active_largest": {leg: float(np.median([r[leg]["n_active_largest"] for r in rs])) if n else None
                                          for leg in LEGS},
                "max_mn_rate_hz_median": float(np.median([r["max_mn_rate_hz"] for r in rs])) if n else None,
            }
            entry["pass"] = bool(n > 0 and n_err == 0 and entry["both_fraction"] >= PASS_FRACTION)
            table[f"{k:g}"][c] = entry
    passing = [k for k in kappas if all(table[f"{k:g}"][c]["pass"] for c in cells)]
    kstar = max(passing) if passing else None
    eff = None
    label = "no grid kappa meets C1 and C2"
    if kstar is not None:
        eff = {c: table[f"{kstar:g}"][c]["smallest_active_fraction"] >= EFFECTIVE_FRACTION for c in cells}
        label = "effective" if all(eff.values()) else ("feedback ineffective" if not any(eff.values()) else "mixed")
    return {"table": table, "passing_kappas": passing, "kappa_star": kstar, "effective_by_cell": eff, "label": label}


# ----------------------------------------------------------------------------- stimulus
def stimulus_states(T: float = T_S):
    """Per tick body-state arrays for both front legs: the clip's angles and velocities
    (1 s, differentiated, then tiled to T) and the replayed trochanter-femur force."""
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs
    from organism.proprio_transduction import replay_forces

    clip = load_walking_clip_front_legs()
    h = float(clip["timestep"])
    n = int(round(T / H))
    body = TetheredFlyBody("position")
    forces = replay_forces(body, clip)  # same replay as the A3/A4 campaniform normalisation
    out = {}
    for leg, bl in BODY_LEG.items():
        ang = np.asarray(clip[bl], dtype=np.float64)
        vel = np.gradient(ang, h, axis=0)
        frc = np.zeros_like(ang)
        frc[:, 4] = forces[leg]
        reps = int(np.ceil(n / ang.shape[0]))
        out[bl] = {k: np.tile(v, (reps, 1))[:n] for k, v in (("angle", ang), ("velocity", vel), ("actuator_force", frc))}
    return out


def afferent_current_sequence(states, cell: dict, r_star: float = R_STAR_HZ) -> tuple[np.ndarray, np.ndarray]:
    """(rows, I[n_ticks, 31]) from the transducer's own ``currents`` at every tick."""
    from organism.proprio_transduction import ProprioTransducer

    tr = ProprioTransducer(cell, r_star_hz=r_star)
    rows = np.concatenate([idx for idx, _ in tr.groups.values()])
    n = states["lf"]["angle"].shape[0]
    I = np.empty((n, rows.size))
    buf = np.zeros(tr.n_neurons)
    for k in range(n):
        st = {bl: {key: states[bl][key][k] for key in ("angle", "velocity", "actuator_force")} for bl in ("lf", "rf")}
        I[k] = tr.currents(st, out=buf)[rows]
    return rows, I


# ----------------------------------------------------------------------------- worker
def _init(seqs, mn_rows, T):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    _W["seqs"], _W["mn_rows"], _W["T"] = seqs, mn_rows, T


def _job(args):
    rep, kappa, cell_key = args
    rows, I = _W["seqs"][cell_key]
    mn_rows, T = _W["mn_rows"], _W["T"]
    from experiment.paper2_cosim import Network
    from flybrain.pugliese_rate import RATE_MAX_HZ

    t0 = time.time()
    try:
        if "base" not in _W:
            _W["base"] = Network.load(SEED, PARAM_BLOCK)
            _W["nets"] = {}
        if kappa not in _W["nets"]:
            _W["nets"][kappa] = kappa_network(_W["base"], kappa, afferent_rows(), dn_off=True)
        net = _W["nets"][kappa]
        st = net.stepper(rep, H)
        n = int(round(T / H))
        Ibuf = np.zeros(net.W.shape[0])
        mx = np.zeros(len(mn_rows))
        for k in range(n):
            Ibuf[rows] = I[k]
            st.step(Ibuf)
            if (k + 1) % RECORD_EVERY == 0 and (k + 1) > TRANSIENT_TICKS:
                R = np.clip(st.R, 0.0, RATE_MAX_HZ)
                np.maximum(mx, R[mn_rows], out=mx)
        finite = bool(np.all(np.isfinite(st.R)))
        top = float(mx.max())
        return {"replicate": rep, "kappa": kappa, "cell": cell_key, "status": "ok", "mn_max": mx,
                "finite": finite, "max_mn_rate_hz": top, "seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # kept, not dropped
        return {"replicate": rep, "kappa": kappa, "cell": cell_key, "status": "error",
                "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 2)}


# ----------------------------------------------------------------------------- calibration
def calibrate(replicates=CALIBRATION_REPLICATES, kappas=GRID, cells=None, *, T: float = T_S, workers: int = 9,
              log=print, audit_dir: Path | None = None) -> dict:
    from organism.proprio_transduction import FACTORIAL

    reps = check_replicates(replicates)
    ks = check_kappas(kappas)
    cells = [dict(c) for c in (FACTORIAL if cells is None else cells)]
    name = lambda c: f"{c['hair_plate_coxa']}-{c['hair_plate_trochanter']}"  # noqa: E731
    t0 = time.time()
    terc = motor_neuron_terciles()
    mn_rows = np.array(terc["T1L"]["rows"] + terc["T1R"]["rows"], dtype=np.int64)
    states = stimulus_states(T)
    seqs = {}
    stim_hash = hashlib.sha256()
    for c in cells:
        rows, I = afferent_current_sequence(states, c)
        seqs[name(c)] = (rows, I)
        stim_hash.update(I.tobytes())
    log(f"stimulus ready in {time.time() - t0:.0f} s")
    jobs = [(r, k, name(c)) for k in ks for c in cells for r in reps]
    results = []
    if workers <= 0:
        _init(seqs, mn_rows, T)
        for j in jobs:
            results.append(_job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(seqs, mn_rows, T)) as pool:
            futs = [pool.submit(_job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs)):
                results.append(f.result())
                if (i + 1) % 256 == 0:
                    log(f"{i + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s")
    rows_out = []
    audit = {}
    for res in results:
        r = {k: res[k] for k in ("replicate", "kappa", "cell", "status", "seconds")}
        if res["status"] == "ok":
            r.update(run_criteria(dict(zip(mn_rows.tolist(), res["mn_max"].tolist())), terc))
            r["max_mn_rate_hz"] = res["max_mn_rate_hz"]
            r["finite"] = res["finite"]
            audit[f"{res['kappa']:g}|{res['cell']}|{res['replicate']}"] = res["mn_max"]
        else:
            r["error"] = res["error"]
        rows_out.append(r)
    rows_out.sort(key=lambda r: (-r["kappa"], r["cell"], r["replicate"]))
    summary = summarise(rows_out, ks, [name(c) for c in cells])
    audit_path = None
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=False)
        audit_path = audit_dir / "mn_max_rates_after_250ms.npz"
        np.savez_compressed(audit_path, keys=np.array(sorted(audit)), mn_rows=mn_rows,
                            values=np.stack([audit[k] for k in sorted(audit)]))
    return {
        "replicates": list(reps), "kappas": list(ks), "cells": [name(c) for c in cells], "T_s": T,
        "terciles": terc, "stimulus_sha256": stim_hash.hexdigest(),
        "runs": rows_out, **summary,
        "n_runs": len(rows_out), "n_failed": sum(r["status"] != "ok" for r in rows_out),
        "n_nonfinite": sum(r.get("finite") is False for r in rows_out),
        "audit_file": None if audit_path is None else str(audit_path.relative_to(ROOT)),
        "audit_sha256": None if audit_path is None else sha256(audit_path),
        "seconds": round(time.time() - t0, 1),
    }


def canonical_hash(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_document(cal: dict) -> dict:
    import importlib.metadata as md

    from experiment.paper2_stage2B_run import EXPECTED_FROZEN_SHA256, FROZEN_PATH
    from flybrain.authors_mcns import W_PATH, WTABLE_PATH

    data = ROOT / "data" / "paper2"
    doc = {
        "what": "Stage 2B-kappa: afferent synaptic efficacy kappa (amendment A8)",
        "status": "kappa_star_found" if cal["kappa_star"] is not None else "no_kappa_meets_criteria",
        "kappa_star": cal["kappa_star"],
        "effectiveness_label": cal["label"],
        "effective_by_cell": cal["effective_by_cell"],
        "grid": list(GRID),
        "criteria": {
            "active": f"rate > {ACTIVE_HZ} Hz at any 1 ms sample after 250 ms",
            "terciles": "per leg, MNs sorted by wTable size, numpy.array_split into 3 by rank",
            "C1": "no largest-tercile MN active (either leg)",
            "C2": "each leg: active MNs <= n_MN(leg) / 3",
            "pass": f"C1 and C2 together in >= {PASS_FRACTION:.0%} of calibration replicates, in every cell",
            "kappa_star": "largest grid kappa passing in all 4 cells",
            "effective": f"per cell: >= {EFFECTIVE_FRACTION:.0%} of replicates have >= 1 active smallest-tercile MN",
        },
        "setup": {
            "kappa_applies_to": "outgoing synapses (post x pre columns) of the 31 included afferents",
            "dn_drive": "off", "body": "not simulated", "r_star_hz": R_STAR_HZ,
            "stimulus": "transduction of the walking clip's angles and velocities (1 s, differentiated, tiled to 2 s), both front legs; campaniform force from the A3 replay, tiled",
            "integrator": "flybrain/rate_stepper.py RK4, h = 0.1 ms, afferent input held per step",
            "seed": SEED, "parameter_block": PARAM_BLOCK,
        },
        "calibration": cal,
        "stage2B_frozen_bridge_sha256": sha256(FROZEN_PATH),
        "stage2B_frozen_bridge_expected": EXPECTED_FROZEN_SHA256,
        "data_sha256": {
            "wTable": sha256(WTABLE_PATH), "W": sha256(W_PATH),
            "sensory_table": sha256(data / "stage2B_sensory_table.json"),
            "sensory_norm": sha256(data / "stage2B_sensory_norm.json"),
        },
        "code_sha256": {p: sha256(ROOT / p) for p in (
            "flybrain/rate_stepper.py", "organism/proprio_transduction.py", "organism/flybody_tethered.py",
            "experiment/paper2_cosim.py", "experiment/paper2_kappa_calibrate.py")},
        "versions": {"flygym": md.version("flygym"), "mujoco": md.version("mujoco"), "numpy": md.version("numpy"),
                     "scipy": md.version("scipy")},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        doc["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
        doc["git_dirty_paths"] = subprocess.check_output(["git", "status", "--porcelain"], text=True, cwd=ROOT).split("\n")
    except Exception:
        doc["git_commit"] = None
    doc["content_sha256"] = canonical_hash(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--workers", type=int, default=9)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    from experiment.paper2_stage2B_run import preflight

    preflight()  # the stage 2B frozen bridge and its code/data must still verify
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cal = calibrate(workers=a.workers, audit_dir=AUDIT_DIR / f"paper2_stage2Bk_calibration_{stamp}")
    doc = frozen_document(cal)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} kappa*={doc['kappa_star']} label={doc['effectiveness_label']} sha256={sha256(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

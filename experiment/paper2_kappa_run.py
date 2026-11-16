"""Stage 2B-kappa runner (amendment A8): stage 2B runs with afferent efficacy kappa.

Same semantics as experiment/paper2_stage2B_run.py: the same conditions,
per-run files and refusals, including the A7 frozen-bridge hash and code/data
checks, which reuse its ``preflight``. Two additions: kappa multiplies the
outgoing synapses of the 31 included afferents, applied to a copy of the
loaded Network's W (experiment.paper2_kappa_calibrate.kappa_network), and the
kappa-frozen file must have the SHA-256 given on the command line.

Arms:
  confirm  kappa = kappa* from the kappa-frozen file; replicates 0-63; all five
           conditions x 4 hair-plate cells (as stage 2B primary, A1-A7).
           Refused if the calibration found no kappa*.
  scan     every grid kappa; replicates 0-15; closed loop, scrambled and
           mismatched x 4 cells (descriptive only, A8). Open loop and
           deafferented do not depend on kappa: they are not rerun, and the
           report records that the stage 2B primary outputs (replicates
           0-15) serve for them.

It computes **no** phase, lag, coherence or coordination read-out.

  .venv/bin/python -m experiment.paper2_kappa_run --arm confirm --kappa-sha256 <hash from the A8 follow-up amendment>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment import paper2_stage2B_run as sr
from experiment.paper2_kappa_calibrate import GRID, afferent_rows, kappa_network

ROOT = Path(__file__).resolve().parents[1]
KAPPA_FROZEN_PATH = ROOT / "data" / "paper2" / "stage2Bk_kappa_frozen.json"
KAPPA_GUARDED = ("data/paper2/stage2Bk_kappa_frozen.json", "experiment/paper2_kappa_calibrate.py")
STAGE2B_PRIMARY_OUTPUT = "outputs/paper2_stage2B_primary_20261112T074301Z"
ARMS = {
    "confirm": {"replicates": tuple(range(0, 64)), "conditions": ("open_loop", "deafferented", "closed_loop", "scrambled", "mismatched")},
    "scan": {"replicates": tuple(range(0, 16)), "conditions": ("closed_loop", "scrambled", "mismatched")},
}

_W: dict = {}


def kappa_preflight(kappa_path: Path, expected_sha256: str, dirty_lines: list[str] | None = None) -> dict:
    kappa_path = Path(kappa_path)
    got = sr.sha256(kappa_path)
    if got != expected_sha256:
        raise RuntimeError(f"kappa-frozen SHA-256 {got} != expected {expected_sha256}; refusing to run")
    if dirty_lines is None:
        dirty_lines = [ln for ln in subprocess.check_output(
            ["git", "status", "--porcelain", "--", *KAPPA_GUARDED], text=True, cwd=ROOT).splitlines() if ln.strip()]
    if dirty_lines:
        raise RuntimeError("kappa-frozen file or kappa calibration code not committed; refusing to run:\n" + "\n".join(dirty_lines))
    doc = json.loads(kappa_path.read_text())
    if [float(k) for k in doc["grid"]] != list(GRID):
        raise RuntimeError("kappa-frozen grid differs from the A8 grid")
    if doc["kappa_star"] is not None and float(doc["kappa_star"]) not in GRID:
        raise RuntimeError("kappa* is not on the A8 grid")
    return doc


def kappas_for(arm: str, doc: dict, override=None) -> tuple[float, ...]:
    if arm == "confirm":
        if doc["kappa_star"] is None:
            raise RuntimeError("the kappa calibration found no kappa*; the confirmatory arm is refused")
        ks = (float(doc["kappa_star"]),)
        if override is not None and tuple(float(k) for k in override) != ks:
            raise ValueError("the confirmatory arm runs kappa* only")
        return ks
    ks = tuple(GRID) if override is None else tuple(float(k) for k in override)
    bad = [k for k in ks if k not in GRID]
    if bad:
        raise ValueError(f"kappa values must be on the A8 grid; got {bad}")
    return ks


# ----------------------------------------------------------------------------- worker
def _setup(settings: dict, frozen: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    if _W.get("key") == frozen["content_sha256"]:
        return
    from experiment.paper2_cosim import Network
    from organism.flybody_tethered import TetheredFlyBody
    from organism.mn_joint_bridge import BridgeParams, MNJointBridge

    _W.clear()
    _W["key"] = frozen["content_sha256"]
    _W["base"] = Network.load(frozen["parameter_block"]["seed"], frozen["parameter_block"]["n_replicates"])
    _W["aff"] = afferent_rows()
    _W["nets"] = {}
    _W["body"] = TetheredFlyBody(settings["actuation"])
    _W["bridge"] = MNJointBridge()
    _W["params"] = BridgeParams.from_dict(frozen["bridge_params"])
    _W["rec"] = sr.recorded_rows()
    _W["transducers"] = {}


def _net(kappa: float):
    if kappa not in _W["nets"]:
        _W["nets"][kappa] = kappa_network(_W["base"], kappa, _W["aff"])
    return _W["nets"][kappa]


def _transducer(cell: dict, r_star: float):
    from organism.proprio_transduction import ProprioTransducer

    key = (sr.cell_name(cell), r_star)
    if key not in _W["transducers"]:
        _W["transducers"][key] = ProprioTransducer(cell, r_star_hz=r_star)
    return _W["transducers"][key]


def _save(path: Path, res: dict, meta: dict) -> None:
    from flybrain.pugliese_rate import RATE_MAX_HZ

    rec = _W["rec"]
    full = res["rates_hz"]
    meta["rates_valid"] = bool(np.all(np.isfinite(full)) and full.min() >= 0.0 and full.max() <= RATE_MAX_HZ)
    meta["max_rate_hz"] = float(np.nanmax(full))
    arrays = {"rates_hz": full[rec["rows"]], "rows": rec["rows"], "bodyIds": rec["bodyIds"],
              "labels": np.array(rec["labels"]), "dt_s": np.array(res["dt_s"])}
    for k in ("angle", "velocity", "actuator_force", "command", "afferent_rows", "afferent_current"):
        if k in res:
            arrays[k] = res[k]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def _run_one(out_dir, settings, frozen, kappa_sha, kappa, rep, condition, cell, surrogate=None, keep_state=False):
    from experiment.paper2_cosim import RunConfig, run

    sub = out_dir / "runs" / f"kappa_{kappa:g}"
    name = f"rep{rep:03d}_{condition}" + (f"_{sr.cell_name(cell)}" if cell else "")
    meta = {"arm": settings["arm"], "kappa": kappa, "kappa_frozen_sha256": kappa_sha, "condition": condition,
            "cell": cell, "replicate": rep, "r_star_hz": settings["r_star_hz"] if cell else None,
            "actuation": settings["actuation"], "frozen_sha256": sr.EXPECTED_FROZEN_SHA256,
            "file": f"kappa_{kappa:g}/{name}.npz"}
    t0 = time.time()
    try:
        cfg = RunConfig(condition, rep, seed=frozen["parameter_block"]["seed"], T=frozen["integrator"]["T_s"],
                        hair_plate_directions=cell, r_star_hz=settings["r_star_hz"], record_state_every_tick=keep_state)
        res = run(cfg, net=_net(kappa), params=_W["params"], body=_W["body"], bridge=_W["bridge"],
                  transducer=_transducer(cell, settings["r_star_hz"]) if cell else None, surrogate_source=surrogate)
        meta.update(seconds=round(time.time() - t0, 2), status="ok")
        _save(sub / f"{name}.npz", res, meta)
        return meta, res.get("state_every_tick")
    except Exception as exc:  # failures are kept
        meta.update(status="error", seconds=round(time.time() - t0, 2), error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc())
        (sub / f"{name}.error.json").write_text(json.dumps(meta, indent=1))
        return meta, None


def job(args) -> list[dict]:
    out_dir, settings, frozen, kappa_sha, kappa, rep, cell, conditions = args
    _setup(settings, frozen)
    metas = []
    if cell is None:
        for c in conditions:
            metas.append(_run_one(out_dir, settings, frozen, kappa_sha, kappa, rep, c, None)[0])
        return metas
    state = None
    for c in sr.FEEDBACK_ORDER:  # closed loop is written before its scrambled run
        if c not in conditions:
            continue
        if c == "scrambled":
            if state is None:
                metas.append({"condition": c, "replicate": rep, "cell": cell, "kappa": kappa, "status": "error",
                              "error": "closed loop of this replicate/cell missing or failed"})
                continue
            metas.append(_run_one(out_dir, settings, frozen, kappa_sha, kappa, rep, c, cell, surrogate=state)[0])
        else:
            meta, st = _run_one(out_dir, settings, frozen, kappa_sha, kappa, rep, c, cell, keep_state=(c == "closed_loop"))
            if c == "closed_loop":
                state = st if meta["status"] == "ok" else None
            metas.append(meta)
    return metas


# ----------------------------------------------------------------------------- main
def run_kappa_arm(arm: str, *, kappa_sha256: str, kappa_path: Path = KAPPA_FROZEN_PATH, replicates=None, kappas=None,
                  out: Path | None = None, workers: int = 9, conditions=None, cells=None,
                  frozen_path: Path = sr.FROZEN_PATH, dirty_lines=None, kappa_dirty_lines=None) -> dict:
    from organism.proprio_transduction import FACTORIAL

    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    spec = ARMS[arm]
    reps = sr.check_replicates(spec["replicates"] if replicates is None else replicates)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out) if out is not None else ROOT / "outputs" / f"paper2_stage2Bk_{arm}_{stamp}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    frozen = sr.preflight(frozen_path, dirty_lines=dirty_lines)
    kdoc = kappa_preflight(kappa_path, kappa_sha256, dirty_lines=kappa_dirty_lines)
    ks = kappas_for(arm, kdoc, kappas)
    settings = dict(sr.arm_settings("primary", frozen))
    settings["arm"] = f"kappa_{arm}"
    conditions = spec["conditions"] if conditions is None else tuple(conditions)
    allowed = set(spec["conditions"])
    if not set(conditions) <= allowed:
        raise ValueError(f"arm {arm} runs only {sorted(allowed)}")
    cells = [dict(c) for c in (FACTORIAL if cells is None else cells)]
    base = [c for c in ("open_loop", "deafferented") if c in conditions]
    fb = [c for c in sr.FEEDBACK_ORDER if c in conditions]
    if "scrambled" in fb and "closed_loop" not in fb:
        raise ValueError("scrambled needs closed_loop in the same run")
    jobs = []
    for k in ks:
        for rep in reps:
            if base:
                jobs.append((out, settings, frozen, kappa_sha256, k, rep, None, tuple(base)))
            for cell in cells:
                if fb:
                    jobs.append((out, settings, frozen, kappa_sha256, k, rep, cell, tuple(fb)))
    for k in ks:
        (out / "runs" / f"kappa_{k:g}").mkdir(parents=True)
    report = {
        "what": f"Stage 2B-kappa {arm} runs (amendment A8); traces only, no read-outs",
        "arm": arm, "kappas": list(ks), "kappa_star": kdoc["kappa_star"], "kappa_frozen_sha256": kappa_sha256,
        "replicates": list(reps), "conditions": list(conditions), "cells": cells, "settings": settings,
        "frozen_sha256": sr.EXPECTED_FROZEN_SHA256,
        "kappa_independent_reused": None if base else {
            "conditions": ["open_loop", "deafferented"],
            "note": "kappa does not act without afferent input; stage 2B primary outputs serve for these",
            "source": STAGE2B_PRIMARY_OUTPUT},
        "runner_sha256": sr.sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "runner_git_status": subprocess.check_output(["git", "status", "--porcelain", "--", "experiment/paper2_kappa_run.py"],
                                                     text=True, cwd=ROOT).strip(),
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "n_jobs": len(jobs), "runs": [],
    }
    rp = out / "report.json"
    rp.write_text(json.dumps(report, indent=1))
    t0 = time.time()
    if workers <= 0:
        for j in jobs:
            report["runs"].extend(job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs)):
                try:
                    report["runs"].extend(f.result())
                except Exception as exc:
                    report["runs"].append({"status": "error", "error": f"job crashed: {type(exc).__name__}: {exc}"})
                if i % 9 == 0:
                    report["elapsed_s"] = round(time.time() - t0, 1)
                    rp.write_text(json.dumps(report, indent=1))
    report["runs"].sort(key=lambda m: (-m.get("kappa", 0), m.get("replicate", -1), str(m.get("cell")), m.get("condition", "")))
    report["n_runs"] = len(report["runs"])
    report["n_ok"] = sum(m.get("status") == "ok" for m in report["runs"])
    report["n_failed"] = report["n_runs"] - report["n_ok"]
    report["n_rates_invalid"] = sum(m.get("status") == "ok" and not m.get("rates_valid") for m in report["runs"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "finished"
    rp.write_text(json.dumps(report, indent=1))
    report["out"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--kappa-sha256", required=True, help="SHA-256 of the kappa-frozen file, as committed in the amendment")
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    r = run_kappa_arm(a.arm, kappa_sha256=a.kappa_sha256, out=a.out, workers=a.workers)
    print(json.dumps({k: r[k] for k in ("out", "kappas", "n_runs", "n_ok", "n_failed", "n_rates_invalid", "elapsed_s")}, indent=1))
    return 0 if r["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Stage 2B bridge calibration and freeze (design §5; amendments A1, A3-A6; plan step 7).

Criteria are independent of coordination. Nothing in this module computes a
phase, lag, coherence or cross-correlation (enforced by an AST test).

Calibration uses only replicates 960-1023 of the seed-1, 1,024-replicate
parameter block, with bilateral DNg100 drive, deafferented body and position
actuation (the primary). Test replicates 0-127 are refused.

1. Network pass. Run each calibration replicate's network with no afferent
   input (deafferented = open loop, A1.4). Record the bridge drive u (agonist
   minus antagonist sum, Hz) of all 14 front-leg DOFs at every 0.1 ms tick.
2. c_j, s_j: median (amendment A5; the first attempt used the mean) and 5-95%
   half-range of u_j, per leg, pooled over replicates and t >= 250 ms. Every
   channel, driven or not, is centred on its median drive. A DOF is "driven" if
   s_j >= 0.5 Hz; for undriven DOFs, s = the median s of the driven DOFs
   (design §3.4).
3. Targets: pooled left+right walking-clip mean and 5-95% range per DOF (A4).
   mu_j and Delta_j start at the target mean and half-range.
4. Body passes. Drive the deafferented tethered FlyBody with each replicate's
   recorded u through the bridge's own ``commands`` code. This is exactly the
   co-simulation's deafferented run, because nothing flows back. Measure each
   replicate's joint-angle mean and 5-95% range (t >= 250 ms, 1 ms samples) and
   take the median over replicates. For driven DOFs, correct mu additively and
   Delta multiplicatively toward the targets, at most 8 times (amendment A6;
   the limit was 3 before).
5. Acceptance (design §5.1). Coxa pitch and trochanter-femur pitch of each front
   leg: median mean within +-0.1 rad of the target, and median 5-95% range
   within +-20% of the target. Each criterion is written as pass or fail; a
   failed calibration is written too, not discarded.
6. Equivalence check. Real co-simulation deafferented runs of two calibration
   replicates with the final parameters must reproduce the body-pass joint
   angles bit for bit.
7. Freeze. Write data/paper2/stage2B_bridge_frozen.json: bridge parameters,
   sensory settings (r* = 100 Hz, g_c; hair-plate factorial with no primary,
   A1.5), drive, h, T, replicate sets, surrogate seed rule, versions, data
   hashes, git commit, acceptance and its own content hash. It refuses to
   overwrite.

  .venv/bin/python -m experiment.paper2_calibrate_2B --workers 9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "paper2" / "stage2B_bridge_frozen.json"
SEED = 1
PARAM_BLOCK = 1024
CALIBRATION_REPLICATES = tuple(range(960, 1024))
TEST_REPLICATES = tuple(range(0, 128))
H = 1e-4
T_S = 2.0
TRANSIENT_TICKS = 2500  # 250 ms
RECORD_EVERY = 10  # body statistics on the 1 ms grid
DRIVEN_MIN_S_HZ = 0.5
MAX_CORRECTIONS = 8  # amendment A6 (was 3)
ACCEPT_MEAN_RAD = 0.1
ACCEPT_RANGE_REL = 0.2
ACCEPT_DOFS = {"coxa_pitch": 2, "trfe_pitch": 4}
LEGS = ("T1L", "T1R")
BODY_LEG = {"T1L": "lf", "T1R": "rf"}
HELD = (0, 1)
VERIFY_REPLICATES = (960, 961)
R_STAR_HZ = 100.0
R_STAR_R2_HZ = (50.0, 150.0)  # robustness arm R2; upper value set by amendment A5 (200 Hz equals the rate cap)

_W: dict = {}


def _threads1() -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_replicates(reps) -> tuple[int, ...]:
    reps = tuple(int(r) for r in reps)
    bad = [r for r in reps if r not in CALIBRATION_REPLICATES]
    if bad:
        raise ValueError(f"calibration may use only replicates 960-1023; got {bad[:5]}")
    return reps


# ----------------------------------------------------------------------------- targets
def pooled_targets(clip) -> dict:
    """Pooled left+right walking-clip mean and 5-95% range per DOF (amendment A4)."""
    pooled = np.concatenate([np.asarray(clip["lf"]), np.asarray(clip["rf"])])
    p5, p95 = np.percentile(pooled, 5, axis=0), np.percentile(pooled, 95, axis=0)
    return {"mean": pooled.mean(axis=0), "p5": p5, "p95": p95, "range": p95 - p5}


# ----------------------------------------------------------------------------- network pass
def _network_job(args):
    rep, T = args
    _threads1()
    from experiment.paper2_cosim import Network, RATE_MAX_HZ
    from organism.mn_joint_bridge import MNJointBridge

    if "net" not in _W:
        _W["net"] = Network.load(SEED, PARAM_BLOCK)
        _W["bridge"] = MNJointBridge()
    net, bridge = _W["net"], _W["bridge"]
    st = net.stepper(rep, H)
    n = int(round(T / H))
    u = np.empty((n, 2, 7))
    max_rate = 0.0
    for k in range(n):
        st.step(None)  # deafferented: no afferent input
        R = np.clip(st.R, 0.0, RATE_MAX_HZ)
        d = bridge.drive(R)
        u[k, 0], u[k, 1] = d["T1L"], d["T1R"]
        if (k + 1) % RECORD_EVERY == 0:
            max_rate = max(max_rate, float(R.max()))
    return rep, u, {"max_rate_hz": max_rate, "finite": bool(np.all(np.isfinite(st.R)))}


def drive_stats(us: dict[int, np.ndarray]) -> dict:
    """c, s per leg and DOF, pooled over replicates and t >= 250 ms.

    c is the median drive of every channel (amendment A5). The mean is kept as
    ``c_mean`` for reference only.
    """
    stack = np.concatenate([u[TRANSIENT_TICKS:] for u in us.values()])  # (N, 2, 7)
    c_mean = stack.mean(axis=0)
    c_median = np.median(stack, axis=0)
    s = (np.percentile(stack, 95, axis=0) - np.percentile(stack, 5, axis=0)) / 2
    driven = s >= DRIVEN_MIN_S_HZ
    for li in range(2):
        driven[li, list(HELD)] = False
    fallback = float(np.median(s[driven])) if driven.any() else 1.0
    s_used = np.where(driven, s, fallback)
    return {"c_mean": c_mean, "c_median": c_median, "s_raw": s, "driven": driven, "c": c_median.copy(),
            "s": s_used, "s_fallback_hz": fallback}


# ----------------------------------------------------------------------------- body pass
def make_params(mu, Delta, c, s, lower, upper, driven):
    from organism.mn_joint_bridge import BridgeParams

    p = BridgeParams.unfitted("position")
    for li, leg in enumerate(LEGS):
        scale = np.asarray(Delta[li], dtype=np.float64).copy()
        scale[list(HELD)] = 0.0
        p.offset[leg] = np.asarray(mu[li], dtype=np.float64).copy()
        p.scale[leg] = scale
        p.c[leg] = np.asarray(c[li], dtype=np.float64).copy()
        p.s[leg] = np.asarray(s[li], dtype=np.float64).copy()
        p.lower[leg] = np.asarray(lower[li], dtype=np.float64).copy()
        p.upper[leg] = np.asarray(upper[li], dtype=np.float64).copy()
    p.fitted_by = "experiment/paper2_calibrate_2B.py"
    return p


def replay_bridge(u: np.ndarray):
    """MNJointBridge whose drive is the recorded u[k] (argument k), so ``commands``
    runs the bridge's own arithmetic on the recorded drive."""
    from organism.mn_joint_bridge import MNJointBridge

    class _Replay(MNJointBridge):
        def drive(self, k):
            return {"T1L": u[k, 0], "T1R": u[k, 1]}

    return _Replay()


def body_replay(body, u: np.ndarray, params, *, keep_trace: bool = False) -> dict:
    """The co-simulation's deafferented body sequence, driven by recorded u."""
    bridge = replay_bridge(u)
    # The co-simulation starts the body at the commands for R = 0 (u = 0).
    cmd0 = _commands_at_zero(params)
    body.reset(
        dict(zip(body.dof_names["lf"], cmd0["T1L"])),
        dict(zip(body.dof_names["rf"], cmd0["T1R"])),
        warmup_s=0.1,
        left_angles=dict(zip(body.dof_names["lf"], cmd0["T1L"])),
        right_angles=dict(zip(body.dof_names["rf"], cmd0["T1R"])),
    )
    n = u.shape[0]
    rec = np.empty((n // RECORD_EVERY + 1, 2, 7))
    s0 = body.front_leg_arrays()
    rec[0, 0], rec[0, 1] = s0["lf"]["angle"], s0["rf"]["angle"]
    for k in range(n):
        cmd = bridge.commands(k, params)
        body.set_front_leg_targets_array(cmd["T1L"], cmd["T1R"])
        body.step()
        if (k + 1) % RECORD_EVERY == 0:
            s = body.front_leg_arrays()
            rec[(k + 1) // RECORD_EVERY, 0] = s["lf"]["angle"]
            rec[(k + 1) // RECORD_EVERY, 1] = s["rf"]["angle"]
    post = rec[TRANSIENT_TICKS // RECORD_EVERY:]
    out = {
        "mean": post.mean(axis=0),
        "p5": np.percentile(post, 5, axis=0),
        "p95": np.percentile(post, 95, axis=0),
    }
    out["range"] = out["p95"] - out["p5"]
    if keep_trace:
        out["trace"] = rec
    return out


def _commands_at_zero(params) -> dict:
    from organism.mn_joint_bridge import MNJointBridge

    return MNJointBridge().commands(np.zeros(4310), params)


def _body_job(args):
    rep, u, pdict, keep = args
    _threads1()
    from organism.flybody_tethered import TetheredFlyBody
    from organism.mn_joint_bridge import BridgeParams

    if "body" not in _W:
        _W["body"] = TetheredFlyBody("position")
    return rep, body_replay(_W["body"], u, BridgeParams.from_dict(pdict), keep_trace=keep)


def median_stats(per_rep: dict[int, dict]) -> dict:
    keys = ("mean", "range", "p5", "p95")
    return {k: np.median(np.stack([r[k] for r in per_rep.values()]), axis=0) for k in keys}


def acceptance(measured: dict, targets: dict, driven: np.ndarray) -> dict:
    """Pass/fail per leg and accepted DOF (design §5.1)."""
    out = {"criteria": {}, "accepted": True}
    for li, leg in enumerate(LEGS):
        for name, j in ACCEPT_DOFS.items():
            m_mean, m_range = float(measured["mean"][li, j]), float(measured["range"][li, j])
            t_mean, t_range = float(targets["mean"][j]), float(targets["range"][j])
            mean_ok = abs(m_mean - t_mean) <= ACCEPT_MEAN_RAD
            range_ok = abs(m_range / t_range - 1.0) <= ACCEPT_RANGE_REL
            row = {
                "driven": bool(driven[li, j]),
                "median_mean_rad": m_mean,
                "target_mean_rad": t_mean,
                "mean_error_rad": m_mean - t_mean,
                "mean_pass": bool(mean_ok),
                "median_range_rad": m_range,
                "target_range_rad": t_range,
                "range_ratio": m_range / t_range,
                "range_pass": bool(range_ok),
            }
            row["pass"] = bool(mean_ok and range_ok and driven[li, j])
            out["criteria"][f"{leg}:{name}"] = row
            out["accepted"] &= row["pass"]
    out["accepted"] = bool(out["accepted"])
    return out


# ----------------------------------------------------------------------------- calibration
def calibrate(replicates=CALIBRATION_REPLICATES, *, T: float = T_S, workers: int = 9, verify=VERIFY_REPLICATES,
              log=print, max_corrections: int = MAX_CORRECTIONS) -> dict:
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs

    reps = check_replicates(replicates)
    verify = tuple(r for r in verify if r in reps)
    _threads1()
    t0 = time.time()
    clip = load_walking_clip_front_legs()
    targets = pooled_targets(clip)
    probe = TetheredFlyBody("position")
    desc = probe.describe()["dofs"]
    lower = np.array([[desc[n]["range_rad"][0] for n in probe.dof_names[BODY_LEG[leg]]] for leg in LEGS])
    upper = np.array([[desc[n]["range_rad"][1] for n in probe.dof_names[BODY_LEG[leg]]] for leg in LEGS])

    # 1-2. network pass
    us, net_info = {}, {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for rep, u, info in pool.map(_network_job, [(r, T) for r in reps]):
            us[rep] = u
            net_info[rep] = info
    t_net = time.time() - t0
    log(f"network pass: {len(reps)} replicates in {t_net:.0f} s")
    ds = drive_stats(us)

    # 3-4. body passes
    mu = np.tile(targets["mean"], (2, 1))
    Delta = np.tile(targets["range"] / 2, (2, 1))
    iterations = []
    for it in range(max_corrections + 1):
        params = make_params(mu, Delta, ds["c"], ds["s"], lower, upper, ds["driven"])
        pd = params.to_dict()
        per_rep = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for rep, st in pool.map(_body_job, [(r, us[r], pd, r in verify) for r in reps]):
                per_rep[rep] = st
        med = median_stats(per_rep)
        acc = acceptance(med, targets, ds["driven"])
        iterations.append({
            "iteration": it,
            "params": pd,
            "median_mean_rad": med["mean"].tolist(),
            "median_range_rad": med["range"].tolist(),
            "acceptance": acc,
        })
        log(f"body pass {it}: accepted={acc['accepted']} "
            + " ".join(f"{k}: mean {v['mean_error_rad']:+.3f} range x{v['range_ratio']:.2f}" for k, v in acc["criteria"].items()))
        if acc["accepted"] or it == max_corrections:
            break
        # Correct driven DOFs only: mu additively, Delta multiplicatively.
        drv = ds["driven"]
        mu = np.where(drv, mu + (targets["mean"][None, :] - med["mean"]), mu)
        ratio = np.divide(targets["range"][None, :], med["range"], out=np.ones_like(Delta), where=med["range"] > 1e-9)
        Delta = np.where(drv, Delta * ratio, Delta)

    final = iterations[-1]
    traces = {r: per_rep[r]["trace"] for r in verify}

    # 6. equivalence with the real co-simulation (deafferented)
    equivalence = {}
    if verify:
        from experiment.paper2_cosim import Network, RunConfig, run
        from organism.mn_joint_bridge import BridgeParams, MNJointBridge

        net = Network.load(SEED, PARAM_BLOCK)
        p = BridgeParams.from_dict(final["params"])
        for r in verify:
            res = run(RunConfig("deafferented", r, T=T), net=net, params=p, body=probe, bridge=MNJointBridge())
            same = bool(np.array_equal(res["angle"], traces[r]))
            equivalence[str(r)] = {"bit_identical_angles": same, "max_abs_diff": float(np.max(np.abs(res["angle"] - traces[r])))}
    equiv_ok = all(v["bit_identical_angles"] for v in equivalence.values()) if equivalence else None

    return {
        "replicates": list(reps),
        "T_s": T,
        "max_corrections": int(max_corrections),
        "corrections_used": len(iterations) - 1,
        "targets": {k: np.asarray(v).tolist() for k, v in targets.items()},
        "drive_stats": {
            "centring": "median drive of every channel (amendment A5)",
            "c_median_hz": ds["c_median"].tolist(),
            "c_mean_hz_reference_only": ds["c_mean"].tolist(),
            "s_raw_hz": ds["s_raw"].tolist(),
            "driven": ds["driven"].tolist(),
            "c_hz": ds["c"].tolist(),
            "s_hz": ds["s"].tolist(),
            "s_fallback_hz": ds["s_fallback_hz"],
        },
        "network": {"all_finite": all(v["finite"] for v in net_info.values()),
                    "max_rate_hz": max(v["max_rate_hz"] for v in net_info.values())},
        "iterations": iterations,
        "final_params": final["params"],
        "acceptance": final["acceptance"],
        "equivalence_with_cosim": {"replicates": list(verify), "per_replicate": equivalence, "pass": equiv_ok},
        "seconds": {"network_pass": round(t_net, 1), "total": round(time.time() - t0, 1)},
    }


# ----------------------------------------------------------------------------- freeze
def canonical_hash(obj: dict) -> str:
    body = {k: v for k, v in obj.items() if k != "content_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def frozen_document(cal: dict) -> dict:
    import importlib.metadata as md

    from flybrain.authors_mcns import W_PATH, WTABLE_PATH
    from organism.proprio_transduction import FACTORIAL, gain_for_rate

    data = ROOT / "data" / "paper2"
    clip = Path(__import__("flygym_demo").__file__).resolve().parent / "ball_flybody_data" / "assets" / "ball_flybody_clip.npz"
    status = "accepted" if (cal["acceptance"]["accepted"] and cal["equivalence_with_cosim"]["pass"] is not False) else "failed"
    doc = {
        "what": "Stage 2B frozen bridge (design §5.3; amendments A1-A6; plan step 7)",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "drive": {"bodyIds": [10045, 10056], "amplitude": 400.0, "pulse_s": [0.02, 1.999]},
        "integrator": {"method": "fixed-step RK4, inputs held per step (flybrain/rate_stepper.py)", "h_s": H, "T_s": T_S,
                       "record_every_ticks": RECORD_EVERY, "transient_s": TRANSIENT_TICKS * H},
        "parameter_block": {"seed": SEED, "n_replicates": PARAM_BLOCK},
        "replicates": {"calibration": [CALIBRATION_REPLICATES[0], CALIBRATION_REPLICATES[-1]],
                       "calibration_used": cal["replicates"],
                       "test": [TEST_REPLICATES[0], TEST_REPLICATES[-1]]},
        "body": {"model": "FlyGym 2.1 FlyBody, TetheredWorld, legs in air (A1.2)", "actuation": "position (primary)",
                 "R1_torque": "not calibrated here (robustness arm; separate frozen file when run)"},
        "bridge_params": cal["final_params"],
        "bridge_scheme": "command_j = offset_j + agonist_angle_sign_j * scale_j * (u_j - c_j) / s_j, clipped to the joint range (organism/mn_joint_bridge.py)",
        "sensory": {
            "r_star_hz": R_STAR_HZ,
            "g_c": gain_for_rate(R_STAR_HZ),
            "current": "I_i = g_c * 7.5 * s_i * u (organism/proprio_transduction.py)",
            "normalisation": "data/paper2/stage2B_sensory_norm.json (pooled L/R, A4)",
            "hair_plate_factorial": list(FACTORIAL),
            "hair_plate_primary": None,
            "hair_plate_note": "no verified source at freeze: all four combinations are run and reported, none primary (A1.5)",
            "R2_r_star_hz": {"lower": R_STAR_R2_HZ[0], "upper": R_STAR_R2_HZ[1],
                             "g_c": {"lower": gain_for_rate(R_STAR_R2_HZ[0]), "upper": gain_for_rate(R_STAR_R2_HZ[1])},
                             "note": "amendment A5: upper value 150 Hz; the design's 200 Hz equals the nominal rate cap, where g_c diverges"},
        },
        "conditions": {
            "open_loop": "no body, no afferent input",
            "deafferented": "body moves, no afferent input (bit-identical to open loop, A1.4)",
            "closed_loop": "each leg's afferents read that leg",
            "mismatched": "left/right swapped (A1.3)",
            "scrambled": "multivariate Fourier phase randomisation of this replicate's closed-loop body state, every tick; rng = numpy.default_rng([seed, replicate, 0x5C]); angles clipped to joint ranges",
        },
        "calibration": cal,
        "versions": {"flygym": md.version("flygym"), "mujoco": md.version("mujoco"), "numpy": md.version("numpy"),
                     "scipy": md.version("scipy")},
        "data_sha256": {
            "wTable": _sha256(WTABLE_PATH),
            "W": _sha256(W_PATH),
            "sensory_table": _sha256(data / "stage2B_sensory_table.json"),
            "sensory_norm": _sha256(data / "stage2B_sensory_norm.json"),
            "mn_joint_table": _sha256(data / "stage2B_mn_joint_table.json"),
            "body_transfer": _sha256(data / "stage2B_body_transfer.json"),
            "walking_clip": _sha256(clip),
        },
        "code_sha256": {p: _sha256(ROOT / p) for p in (
            "flybrain/rate_stepper.py", "organism/flybody_tethered.py", "organism/mn_joint_bridge.py",
            "organism/proprio_transduction.py", "experiment/paper2_cosim.py", "experiment/paper2_calibrate_2B.py")},
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
    p.add_argument("--replicates", type=str, default="960:1024", help="start:stop within 960-1023")
    p.add_argument("--T", type=float, default=T_S)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    lo, hi = (int(x) for x in a.replicates.split(":"))
    cal = calibrate(range(lo, hi), T=a.T, workers=a.workers)
    doc = frozen_document(cal)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {out} status={doc['status']} file sha256={_sha256(out)}")
    return 0 if doc["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())

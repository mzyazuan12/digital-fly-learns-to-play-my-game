"""Chirp transfer function of each front-leg DOF of the tethered FlyBody (stage 2B, plan step 3).

For each front-leg actuation model (position = primary, torque = robustness
arm R1) and each of the 14 front-leg DOFs, one DOF at a time:

- Operating point: every front-leg DOF at the recorded walking mean of its own
  leg (bundled NeuroMechFly v1 clip, see
  organism.flybody_tethered.load_walking_clip_front_legs). Where that mean is
  closer than 0.15 rad to a FlyBody joint limit, it is moved inward to 0.15 rad
  from the limit, so the test signal does not hit the limit (recorded in the
  output). The joints start at the operating angles. Position mode commands
  those angles. Torque mode commands the bias torque stiffness * (angle -
  springref), the passive equilibrium if gravity is ignored. The other four
  legs are held at neutral by their servos.
- Baseline: one unperturbed run per actuation model, from the same reset, is
  subtracted sample by sample from every perturbed run. The simulation is
  deterministic, so this removes slow settling (torque mode's passive time
  constant is about 1 s) and gravity sag exactly, leaving the response to the
  test signal.
- Input: a logarithmic chirp of 1 -> 40 Hz over 4 s added to that DOF's
  command, with 0.2 s cosine tapers. Amplitude 0.1 rad (position) or 0.3 model
  torque units (torque). The record runs 0.1 s before and 0.4 s after the
  chirp.
- Output: that DOF's joint angle minus the baseline run's angle.
- Estimate: band-averaged H1 = sum(Y conj(U)) / sum(|U|^2) over FFT bins
  within +-max(5%, 0.3 Hz) of each reported frequency. The record is deterministic and
  noise-free.
- Check at 11.5 Hz: a 1 s sine of the same amplitude, fitted by least squares
  (sin, cos, offset, slope) over its last 0.5 s.

Gain is output / input: rad/rad for position, rad per torque unit for torque.
Phase lag is in degrees, positive when the angle lags the command.

Writes data/paper2/stage2B_body_transfer.json and refuses to overwrite it
unless --overwrite is given.

  .venv/bin/python -m experiment.paper2_body_transfer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "paper2" / "stage2B_body_transfer.json"
REPORT_FREQS_HZ = [1, 1.5, 2, 3, 4, 6, 8, 10, 11.5, 13, 16, 20, 25, 30]
CHIRP = {"f0_hz": 1.0, "f1_hz": 40.0, "duration_s": 4.0, "taper_s": 0.2, "pre_s": 0.1, "post_s": 0.4}
AMPLITUDE = {"position": 0.1, "torque": 0.3}
SETTLE_S = 0.3
SINE = {"freq_hz": 11.5, "duration_s": 1.0, "fit_last_s": 0.5}


def chirp_signal(h: float) -> tuple[np.ndarray, np.ndarray]:
    """(time, unit-amplitude input) over pre + chirp + post, sampled every h."""
    c = CHIRP
    n_pre, n_ch, n_post = (int(round(c[k] / h)) for k in ("pre_s", "duration_s", "post_s"))
    t = np.arange(n_ch) * h
    k = np.log(c["f1_hz"] / c["f0_hz"]) / c["duration_s"]
    phase = 2 * np.pi * c["f0_hz"] * (np.exp(k * t) - 1.0) / k
    x = np.sin(phase)
    taper = np.ones(n_ch)
    nt = int(round(c["taper_s"] / h))
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(nt) / nt))
    taper[:nt] = ramp
    taper[-nt:] = ramp[::-1]
    u = np.concatenate([np.zeros(n_pre), x * taper, np.zeros(n_post)])
    return np.arange(u.size) * h, u


def band_h1(u: np.ndarray, y: np.ndarray, h: float, freqs) -> list[complex]:
    U = np.fft.rfft(u)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(u.size, h)
    out = []
    for fc in freqs:
        half = max(0.05 * fc, 0.3)
        sel = (f >= fc - half) & (f <= fc + half)
        out.append(complex(np.sum(Y[sel] * np.conj(U[sel])) / np.sum(np.abs(U[sel]) ** 2)))
    return out


def sine_fit(y: np.ndarray, t: np.ndarray, f: float) -> complex:
    A = np.column_stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t), np.ones_like(t), t - t.mean()])
    a, b, _, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return complex(a, b)  # y = Re(H) sin + Im(H) cos  <=>  H = a + i b relative to sin input


LIMIT_MARGIN_RAD = 0.15


def operating_point(body, clip) -> dict:
    """Per front leg: operating angles (recorded mean, kept 0.15 rad inside the limits) and command."""
    desc = body.describe()["dofs"]
    out = {}
    for leg in ("lf", "rf"):
        names = body.dof_names[leg]
        mean = clip[leg].mean(axis=0)
        lo = np.array([desc[n]["range_rad"][0] for n in names]) + LIMIT_MARGIN_RAD
        hi = np.array([desc[n]["range_rad"][1] for n in names]) - LIMIT_MARGIN_RAD
        angle = np.clip(mean, lo, hi)
        if body.front_actuation == "position":
            cmd = angle.copy()
        else:
            k = np.array([desc[n]["stiffness"] for n in names])
            ref = np.array([desc[n]["springref_rad"] for n in names])
            cmd = k * (angle - ref)
        out[leg] = {"command": cmd, "angle": angle, "recorded_mean": mean}
    return out


def _reset(body, op, settle_s: float) -> None:
    body.reset(
        dict(zip(body.dof_names["lf"], op["lf"]["command"])),
        dict(zip(body.dof_names["rf"], op["rf"]["command"])),
        warmup_s=settle_s,
        left_angles=dict(zip(body.dof_names["lf"], op["lf"]["angle"])),
        right_angles=dict(zip(body.dof_names["rf"], op["rf"]["angle"])),
    )


def drive(body, op, leg: str | None, dof_index: int, signal: np.ndarray) -> np.ndarray:
    """Reset to the operating point, add ``signal`` to one DOF's command (or none
    if ``leg`` is None), and return all 14 front-leg angles per step (n, 2, 7)."""
    _reset(body, op, SETTLE_S)
    cmd = {lg: op[lg]["command"].copy() for lg in ("lf", "rf")}
    out = np.empty((signal.size, 2, 7))
    for i in range(signal.size):
        if leg is not None:
            cmd[leg][dof_index] = op[leg]["command"][dof_index] + signal[i]
        body.set_front_leg_targets_array(cmd["lf"], cmd["rf"])
        body.step()
        s = body.front_leg_arrays()
        out[i, 0] = s["lf"]["angle"]
        out[i, 1] = s["rf"]["angle"]
    return out


def measure(body, op, baseline: np.ndarray, leg: str, dof_index: int, amplitude: float) -> dict:
    h = body.timestep
    li = ("lf", "rf").index(leg)
    t, u = chirp_signal(h)
    y = drive(body, op, leg, dof_index, amplitude * u)[:, li, dof_index] - baseline[: u.size, li, dof_index]
    H = band_h1(amplitude * u, y, h, REPORT_FREQS_HZ)
    n = int(round(SINE["duration_s"] / h))
    ts = np.arange(n) * h
    x = amplitude * np.sin(2 * np.pi * SINE["freq_hz"] * ts)
    ys = drive(body, op, leg, dof_index, x)[:, li, dof_index] - baseline[:n, li, dof_index]
    m = int(round(SINE["fit_last_s"] / h))
    Hs = sine_fit(ys[-m:], ts[-m:], SINE["freq_hz"]) / amplitude
    return {
        "recorded_mean_angle_rad": float(op[leg]["recorded_mean"][dof_index]),
        "operating_angle_rad": float(op[leg]["angle"][dof_index]),
        "moved_inside_limits": bool(abs(op[leg]["angle"][dof_index] - op[leg]["recorded_mean"][dof_index]) > 1e-12),
        "operating_command": float(op[leg]["command"][dof_index]),
        "baseline_angle_at_chirp_start_rad": float(baseline[int(round(CHIRP["pre_s"] / h)), li, dof_index]),
        "max_response_rad": float(np.max(np.abs(y))),
        "freq_hz": list(REPORT_FREQS_HZ),
        "gain": [float(abs(z)) for z in H],
        "phase_lag_deg": [float(-np.degrees(np.angle(z))) for z in H],
        "sine_11p5hz": {"gain": float(abs(Hs)), "phase_lag_deg": float(-np.degrees(np.angle(Hs)))},
    }


def run(out: Path = OUT, overwrite: bool = False) -> dict:
    import flygym  # noqa: F401
    import importlib.metadata as md

    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs

    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {out}; pass --overwrite")
    t0 = time.time()
    clip = load_walking_clip_front_legs()
    report = {
        "what": "Chirp transfer function of each front-leg DOF of the tethered FlyBody (stage 2B design §3.2, plan step 3)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "experiment/paper2_body_transfer.py",
        "flygym_version": md.version("flygym"),
        "mujoco_version": md.version("mujoco"),
        "chirp": CHIRP,
        "amplitude": {"position": "0.1 rad", "torque": "0.3 model torque units"},
        "settle_s": SETTLE_S,
        "sine_check": SINE,
        "estimator": "band-averaged H1 over FFT bins within +-5% of each frequency; deterministic noise-free record",
        "units": {"position": "gain rad/rad", "torque": "gain rad per model torque unit"},
        "phase_convention": "phase_lag_deg > 0: joint angle lags the command",
        "operating_point": "recorded walking mean per DOF (ball_flybody_clip, own leg), moved to >= 0.15 rad inside FlyBody joint limits where needed; joints start there; torque mode commands bias torque stiffness*(angle-springref)",
        "baseline": "one unperturbed run per actuation model subtracted sample by sample (deterministic simulation)",
        "results": {},
        "body": {},
    }
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        report["git_commit"] = None
    for mode in ("position", "torque"):
        body = TetheredFlyBody(mode)
        report["body"][mode] = body.describe()
        op = operating_point(body, clip)
        baseline = drive(body, op, None, 0, np.zeros(chirp_signal(body.timestep)[1].size))
        res = {}
        for leg in ("lf", "rf"):
            for j, name in enumerate(body.dof_names[leg]):
                res[name] = measure(body, op, baseline, leg, j, AMPLITUDE[mode])
                s = res[name]["sine_11p5hz"]
                print(f"[{mode}] {name:40s} 11.5 Hz sine: gain {s['gain']:.3f} lag {s['phase_lag_deg']:.1f} deg", flush=True)
        report["results"][mode] = res
    report["seconds"] = round(time.time() - t0, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    r = run(a.out, a.overwrite)
    print(f"wrote {a.out} in {r['seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure whether identified VNC CPG cells oscillate.

This is a readout. It does not retune weights, time constants, or the
graph when the trace is tonic. The Pugliese et al. 2025 bioRxiv walking
rhythms are ~7–15 Hz. The score is compared to that preprint's 0.5
threshold; failing it is a dynamics result, not a reason to edit anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from organism.walking_pathways import CPG_ROLES, LEG_SLOTS, WalkingCircuit

PUBLISHED_WALK_HZ = (7.0, 15.0)
SEARCH_BAND_HZ = (5.0, 20.0)
RHYTHMICITY_THRESHOLD = 0.5
PREVIEW_POINTS = 60
# LIF rest is −52 mV. |V| far outside this range is not a walking rhythm.
EXPLOSION_ABS_MV = 150.0


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = int(x.size)
    if n < 8:
        return np.zeros(max(n, 1), dtype=np.float64)
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conjugate(spec), n=2 * n)[:n].real
    if ac[0] <= 0:
        return np.zeros(n, dtype=np.float64)
    return ac / ac[0]


def rhythmicity_score(
    trace: np.ndarray,
    dt_ms: float,
    *,
    band: tuple[float, float] = SEARCH_BAND_HZ,
) -> dict:
    """Autocorr peak in `band` plus a tonic/silent classifier.

    score in [0, 1] is the autocorrelation at the strongest lag in-band.
    """
    x = np.asarray(trace, dtype=np.float64)
    dt_s = float(dt_ms) / 1000.0
    out = {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
        "cv": 0.0,
        "score": 0.0,
        "peak_hz": None,
        "fft_hz": None,
        "band_power_frac": 0.0,
        "tonic_plateau": False,
        "silent_or_flat": True,
        "oscillatory": False,
        "exploding": False,
        "in_published_walk_band": False,
    }
    if x.size < 16:
        return out
    mean = float(x.mean())
    std = float(x.std())
    out["mean"] = mean
    out["std"] = std
    out["min"] = float(x.min())
    out["max"] = float(x.max())
    out["cv"] = float(std / abs(mean)) if abs(mean) > 1e-9 else 0.0
    span = float(x.max() - x.min())
    finite = np.isfinite(x)
    exploding = bool(
        (not np.all(finite))
        or abs(out["min"]) >= EXPLOSION_ABS_MV
        or abs(out["max"]) >= EXPLOSION_ABS_MV
        or (np.isfinite(x).any() and (np.max(np.abs(x[finite])) >= EXPLOSION_ABS_MV))
    )
    out["exploding"] = exploding
    flat = std < 1e-8 or span < 1e-8
    out["silent_or_flat"] = bool(flat and not exploding)
    if exploding:
        return out
    if flat:
        out["tonic_plateau"] = bool(mean > 0.25)
        return out

    freqs = np.fft.rfftfreq(x.size, d=dt_s)
    spec = np.abs(np.fft.rfft(x - mean)) ** 2
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total = float(spec[1:].sum()) if spec.size > 1 else 0.0
    if total > 0 and np.any(band_mask):
        peak_i = int(np.argmax(spec[band_mask]))
        fft_hz = float(freqs[band_mask][peak_i])
        out["fft_hz"] = fft_hz
        out["band_power_frac"] = float(spec[band_mask][peak_i] / total)

    ac = _autocorr(x)
    lag_min = max(2, int(round(1.0 / band[1] / dt_s)))
    lag_max = min(max(lag_min + 1, x.size // 3), int(round(1.0 / band[0] / dt_s)))
    if lag_max <= lag_min:
        return out
    segment = ac[lag_min : lag_max + 1]
    rel = int(np.argmax(segment))
    score = float(max(0.0, segment[rel]))
    lag = lag_min + rel
    peak_hz = 1.0 / (lag * dt_s) if lag > 0 else None
    out["score"] = score
    out["peak_hz"] = float(peak_hz) if peak_hz is not None else None
    out["tonic_plateau"] = bool(mean > 0.35 and out["cv"] < 0.15 and score < 0.35)
    published = PUBLISHED_WALK_HZ
    in_band = bool(peak_hz is not None and published[0] <= peak_hz <= published[1])
    out["in_published_walk_band"] = in_band
    out["oscillatory"] = bool(score >= RHYTHMICITY_THRESHOLD and not out["tonic_plateau"])
    return out


def downsample_preview(trace: np.ndarray, n: int = PREVIEW_POINTS) -> list[float]:
    x = np.asarray(trace, dtype=np.float64)
    if x.size == 0:
        return []
    if x.size <= n:
        return [float(v) for v in x.tolist()]
    idx = np.linspace(0, x.size - 1, n).round().astype(int)
    return [float(v) for v in x[idx].tolist()]


def cell_signal(net, index: int) -> float:
    i = int(index)
    if bool(net.is_graded[i]):
        return float(net.graded_output[i])
    return float(net.last_spikes[i])


def population_signal(net, indices: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int32)
    if idx.size == 0:
        return 0.0
    graded = net.is_graded[idx]
    values = np.empty(idx.size, dtype=np.float64)
    if np.any(graded):
        values[graded] = net.graded_output[idx[graded]]
    if np.any(~graded):
        values[~graded] = net.last_spikes[idx[~graded]].astype(np.float64)
    return float(values.mean())


@dataclass
class RhythmRecording:
    t_ms: np.ndarray
    dng100: np.ndarray
    dng100_v: np.ndarray
    dng100_l: np.ndarray
    dng100_r: np.ndarray
    dng100_l_v: np.ndarray
    dng100_r_v: np.ndarray
    legs: dict[str, dict[str, np.ndarray]]

    def summary(self, dt_ms: float, stim_onset_ms: float) -> dict:
        stim = self.t_ms >= stim_onset_ms
        dng_spikes = _window_score(self.dng100, stim, dt_ms)
        dng_v = self.dng100_v[stim] if stim.size == self.dng100_v.size and np.any(stim) else self.dng100_v
        report = {
            "DNg100": dng_spikes,
            "DNg100_L": _window_score(self.dng100_l, stim, dt_ms),
            "DNg100_R": _window_score(self.dng100_r, stim, dt_ms),
            "DNg100_v": {
                "mean": float(dng_v.mean()) if dng_v.size else 0.0,
                "std": float(dng_v.std()) if dng_v.size else 0.0,
                "min": float(dng_v.min()) if dng_v.size else 0.0,
                "max": float(dng_v.max()) if dng_v.size else 0.0,
                "exploding": bool(
                    dng_v.size
                    and (abs(float(dng_v.min())) >= EXPLOSION_ABS_MV or abs(float(dng_v.max())) >= EXPLOSION_ABS_MV)
                ),
            },
            "legs": {},
        }
        any_osc = False
        any_tonic = False
        any_explode = bool(report["DNg100"].get("exploding") or report["DNg100_v"]["exploding"])
        n_osc_legs = 0
        for slot in LEG_SLOTS:
            traces = self.legs.get(slot) or {}
            row = {}
            leg_osc = False
            for role, trace in traces.items():
                scored = _window_score(trace, stim, dt_ms)
                row[role] = scored
                if scored.get("exploding"):
                    any_explode = True
                if role in {"E1", "E2", "I1", "I2", "MN"} and scored.get("oscillatory"):
                    leg_osc = True
                if role in {"E1", "E2", "I1"} and scored.get("tonic_plateau"):
                    any_tonic = True
            row["oscillatory"] = leg_osc
            if leg_osc:
                n_osc_legs += 1
                any_osc = True
            report["legs"][slot] = row
        report["n_oscillatory_legs"] = n_osc_legs
        report["any_leg_oscillatory"] = any_osc
        report["any_exploding"] = any_explode
        report["core_tonic_plateau"] = bool(any_tonic and not any_osc and not any_explode)
        return report

    def previews(self) -> dict:
        out = {
            "DNg100": downsample_preview(self.dng100),
            "DNg100_v": downsample_preview(self.dng100_v),
            "DNg100_L": downsample_preview(self.dng100_l),
            "DNg100_R": downsample_preview(self.dng100_r),
            "legs": {},
        }
        for slot, traces in self.legs.items():
            out["legs"][slot] = {role: downsample_preview(tr) for role, tr in traces.items()}
        return out


def _window_score(trace: np.ndarray, stim: np.ndarray, dt_ms: float) -> dict:
    x = np.asarray(trace, dtype=np.float64)
    if stim.size == x.size and np.any(stim):
        x = x[stim]
    return rhythmicity_score(x, dt_ms)


def allocate_traces(circuit: WalkingCircuit, n_steps: int) -> RhythmRecording:
    t = np.zeros(n_steps, dtype=np.float64)
    dng = np.zeros(n_steps, dtype=np.float64)
    dng_v = np.zeros(n_steps, dtype=np.float64)
    legs: dict[str, dict[str, np.ndarray]] = {}
    roles = list(CPG_ROLES) + ["MN"]
    for slot in LEG_SLOTS:
        legs[slot] = {role: np.zeros(n_steps, dtype=np.float64) for role in roles}
    return RhythmRecording(
        t_ms=t,
        dng100=dng,
        dng100_v=dng_v,
        dng100_l=np.zeros(n_steps, dtype=np.float64),
        dng100_r=np.zeros(n_steps, dtype=np.float64),
        dng100_l_v=np.zeros(n_steps, dtype=np.float64),
        dng100_r_v=np.zeros(n_steps, dtype=np.float64),
        legs=legs,
    )


def _dng100_by_side(circuit: WalkingCircuit) -> dict[str, np.ndarray]:
    connectome = circuit.connectome
    idx = circuit.indices("DNg100")
    left = [int(i) for i in idx.tolist() if str(connectome.side[int(i)]).upper().startswith("L")]
    right = [int(i) for i in idx.tolist() if str(connectome.side[int(i)]).upper().startswith("R")]
    return {
        "L": np.asarray(left, dtype=np.int32),
        "R": np.asarray(right, dtype=np.int32),
    }


def record_tick(rec: RhythmRecording, net, circuit: WalkingCircuit, t: int, t_ms: float) -> None:
    rec.t_ms[t] = t_ms
    dng_idx = circuit.indices("DNg100")
    rec.dng100[t] = population_signal(net, dng_idx)
    rec.dng100_v[t] = float(np.mean(net.v[dng_idx])) if dng_idx.size else 0.0
    sides = _dng100_by_side(circuit)
    rec.dng100_l[t] = population_signal(net, sides["L"])
    rec.dng100_r[t] = population_signal(net, sides["R"])
    rec.dng100_l_v[t] = float(np.mean(net.v[sides["L"]])) if sides["L"].size else 0.0
    rec.dng100_r_v[t] = float(np.mean(net.v[sides["R"]])) if sides["R"].size else 0.0
    for slot, copy in circuit.legs.items():
        for role, idx in copy.cells.items():
            rec.legs[slot][role][t] = 0.0 if idx is None else cell_signal(net, idx)
        rec.legs[slot]["MN"][t] = population_signal(net, copy.motor_indices)


def interpret_intact(summary: dict) -> dict:
    n_osc = int(summary.get("n_oscillatory_legs") or 0)
    tonic = bool(summary.get("core_tonic_plateau"))
    exploding = bool(summary.get("any_exploding"))
    reproduced = n_osc >= 1 and not exploding
    if exploding:
        answer = "no"
        next_step = (
            "DNg100 stimulation produced an unstable explosion or "
            "non-physiological voltage, not a 7–15 Hz rhythm. Do not change "
            "the graph. Compare MixedDynamicsNetwork equations/gains to the "
            "authors' MANC rate-ODE VNC simulator."
        )
    elif reproduced:
        answer = "yes"
        next_step = (
            "Oscillation is present under DNg100 current. Compare E1/E2/I1 "
            "lesions to Pugliese et al. 2025 bioRxiv, then consider MN→FlyBody."
        )
    elif tonic:
        answer = "no"
        next_step = (
            "Sustained DNg100 produced a tonic plateau, not a 7–15 Hz rhythm. "
            "Do not change the graph. Investigate assumed dynamics, time "
            "constants, and gains against the authors' working VNC simulator."
        )
    else:
        answer = "no"
        next_step = (
            "DNg100 stimulation did not produce oscillatory CPG/MN activity. "
            "Do not change the graph. Compare neuron equations/parameters to "
            "the authors' DNg100_Stim simulator instead of retuning functional_gain."
        )
    return {
        "question": (
            "Does sustained DNg100 activation produce oscillatory activity "
            "downstream of E1/E2/inhibition and leg motor neurons?"
        ),
        "answer": answer,
        "oscillation_reproduced": reproduced,
        "graph_modified": False,
        "dynamics_retuned": False,
        "next_step": next_step,
    }

"""Measure whether identified VNC CPG cells oscillate.

This is a readout. It does not retune weights, time constants, or the
graph when the trace is tonic. The Pugliese et al. 2025 bioRxiv walking
rhythms are ~7–15 Hz. The score is compared to that preprint's 0.5
threshold; failing it is a dynamics result, not a reason to edit anatomy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from flybrain.neurons import (
    EXPLOSION_ABS_MV,
    PHYSIOLOGICAL_V_LOWER_MV,
    PHYSIOLOGICAL_V_UPPER_MV,
    SHIU_LIF_SANITY_MODEL,
    valid_dynamics,
    voltage_is_physiological,
    voltages_finite,
)
from organism.walking_pathways import CPG_ROLES, LEG_SLOTS, WalkingCircuit

PUBLISHED_WALK_HZ = (7.0, 15.0)
SEARCH_BAND_HZ = (5.0, 20.0)
RHYTHMICITY_THRESHOLD = 0.5
PREVIEW_POINTS = 60


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


def derive_rhythm_permissions(
    *,
    dng_voltage_physiological: bool,
    dng_dynamics_valid: bool,
    all_network_voltages_valid: bool,
) -> dict:
    """Rhythm/lesion flags are derived. They never start as True."""
    dng_v_exploding = not (
        bool(dng_voltage_physiological) and bool(dng_dynamics_valid)
    )
    valid_for_rhythm_analysis = (
        bool(dng_dynamics_valid)
        and not dng_v_exploding
        and bool(all_network_voltages_valid)
    )
    return {
        "dng100_voltage_physiological": bool(dng_voltage_physiological),
        "dng100_dynamics_valid": bool(dng_dynamics_valid),
        "dng100_voltage_exploding": bool(dng_v_exploding),
        "all_network_voltages_valid": bool(all_network_voltages_valid),
        "valid_for_rhythm_analysis": bool(valid_for_rhythm_analysis),
        "allow_lesions": bool(valid_for_rhythm_analysis),
        "fft_allowed": bool(valid_for_rhythm_analysis),
    }


def _trace_stats(trace: np.ndarray) -> dict:
    """Numerical/membrane census. Never FFT."""
    x = np.asarray(trace, dtype=np.float64)
    finite = voltages_finite(x)
    phys = voltage_is_physiological(x)
    dyn = valid_dynamics(x)
    exploding = bool(x.size) and not (phys and dyn)
    mean = float(x.mean()) if x.size else 0.0
    std = float(x.std()) if x.size else 0.0
    return {
        "n": int(x.size),
        "mean": mean,
        "std": std,
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
        "cv": float(std / abs(mean)) if x.size and abs(mean) > 1e-9 else 0.0,
        "score": None,
        "rhythmicity_score": None,
        "peak_hz": None,
        "dominant_frequency": None,
        "fft_hz": None,
        "spectral_peak_power": None,
        "band_power_frac": None,
        "autocorrelation_peak": None,
        "tonic_plateau": False,
        "silent_or_flat": True,
        "oscillatory": False,
        "exploding": exploding,
        "voltage_finite": finite,
        "voltage_physiological": phys,
        "valid_dynamics": dyn,
        "valid_for_rhythm_analysis": False,
        "allow_lesions": False,
        "fft_executed": False,
        "in_published_walk_band": False,
        "model_id": SHIU_LIF_SANITY_MODEL,
    }


def rhythmicity_score(
    trace: np.ndarray,
    dt_ms: float,
    *,
    band: tuple[float, float] = SEARCH_BAND_HZ,
) -> dict:
    """Unrestricted spectral/autocorrelation peaks plus a tonic/silent classifier.

    score in [0, 1] is the strongest positive local autocorrelation peak.
    FFT does not run on exploding or non-physiological traces.
    Rhythm/lesion permissions are not granted here.
    """
    x = np.asarray(trace, dtype=np.float64)
    dt_s = float(dt_ms) / 1000.0
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_ms must be finite and positive")
    out = _trace_stats(x)
    if x.size < 16:
        return out
    mean = out["mean"]
    std = out["std"]
    span = float(out["max"] - out["min"])
    exploding = bool(out["exploding"])
    flat = std < 1e-8 or span < 1e-8
    out["silent_or_flat"] = bool(flat and not exploding)
    if exploding:
        return out
    if flat:
        out["score"] = 0.0
        out["rhythmicity_score"] = 0.0
        out["autocorrelation_peak"] = 0.0
        out["spectral_peak_power"] = 0.0
        out["band_power_frac"] = 0.0
        out["tonic_plateau"] = bool(mean > 0.25)
        return out

    out["fft_executed"] = True
    freqs = np.fft.rfftfreq(x.size, d=dt_s)
    spec = np.abs(np.fft.rfft(x - mean)) ** 2
    total = float(spec[1:].sum()) if spec.size > 1 else 0.0
    if total > 0:
        peak_i = 1 + int(np.argmax(spec[1:]))
        fft_hz = float(freqs[peak_i])
        out["fft_hz"] = fft_hz
        out["spectral_peak_power"] = float(spec[peak_i])
        out["band_power_frac"] = float(spec[peak_i] / total)

    ac = _autocorr(x)
    # Search observed autocorrelation peaks, without a published-frequency prior.
    lag_max = x.size // 3
    candidates = np.flatnonzero((ac[1:lag_max] > ac[:lag_max-1]) & (ac[1:lag_max] >= ac[2:lag_max+1])) + 1
    if candidates.size == 0:
        out["dominant_frequency"] = out.get("fft_hz")
        return out
    lag = int(candidates[np.argmax(ac[candidates])])
    score = float(max(0.0, ac[lag]))
    peak_hz = 1.0 / (lag * dt_s)
    out["score"] = score
    out["rhythmicity_score"] = score
    out["peak_hz"] = float(peak_hz) if peak_hz is not None else None
    out["dominant_frequency"] = out.get("fft_hz")
    out["autocorrelation_peak"] = score
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
    network_voltage_valid: np.ndarray | None = None
    dng_voltage_valid: np.ndarray | None = None

    def summary(self, dt_ms: float, stim_onset_ms: float) -> dict:
        stim = self.t_ms >= stim_onset_ms
        dng_v = self.dng100_v[stim] if stim.size == self.dng100_v.size and np.any(stim) else self.dng100_v
        dng_l_v = self.dng100_l_v[stim] if stim.size == self.dng100_l_v.size and np.any(stim) else self.dng100_l_v
        dng_r_v = self.dng100_r_v[stim] if stim.size == self.dng100_r_v.size and np.any(stim) else self.dng100_r_v
        dng_voltage_physiological = bool(voltage_is_physiological(dng_v) and (self.dng_voltage_valid is None or np.all(self.dng_voltage_valid)))
        dng_dynamics_valid = bool(voltages_finite(dng_v) and dng_voltage_physiological)
        dng_voltage_finite = bool(not dng_v.size or voltages_finite(dng_v))
        left_phys = bool(not dng_l_v.size or voltage_is_physiological(dng_l_v))
        right_phys = bool(not dng_r_v.size or voltage_is_physiological(dng_r_v))
        all_network_voltages_valid = bool(
            self.network_voltage_valid is not None
            and self.network_voltage_valid.size > 0
            and np.all(self.network_voltage_valid)
            and dng_voltage_physiological and left_phys and right_phys
        )
        permissions = derive_rhythm_permissions(
            dng_voltage_physiological=dng_voltage_physiological,
            dng_dynamics_valid=dng_dynamics_valid,
            all_network_voltages_valid=all_network_voltages_valid,
        )
        dng_v_exploding = bool(permissions["dng100_voltage_exploding"])
        recruited = {role: any(np.any(_windowed(traces.get(role, np.array([])), stim) > 0) for traces in self.legs.values()) for role in ("E1", "E2", "I1", "I2", "MN")}
        recruited["DNg100"] = bool(np.any(_windowed(self.dng100, stim) > 0))
        fft_ok = bool(permissions["fft_allowed"] and all(recruited.values()))
        permissions.update(valid_for_rhythm_analysis=fft_ok, allow_lesions=fft_ok, fft_allowed=fft_ok)
        dng_spikes = _window_score(self.dng100, stim, dt_ms) if fft_ok else _window_census(self.dng100, stim)
        report = {
            "model_id": SHIU_LIF_SANITY_MODEL,
            "DNg100": dng_spikes,
            "DNg100_L": _window_score(self.dng100_l, stim, dt_ms) if fft_ok else _window_census(self.dng100_l, stim),
            "DNg100_R": _window_score(self.dng100_r, stim, dt_ms) if fft_ok else _window_census(self.dng100_r, stim),
            "DNg100_v": {
                "mean": float(dng_v.mean()) if dng_v.size else 0.0,
                "std": float(dng_v.std()) if dng_v.size else 0.0,
                "min": float(dng_v.min()) if dng_v.size else 0.0,
                "max": float(dng_v.max()) if dng_v.size else 0.0,
                "exploding": dng_v_exploding,
                "finite": dng_voltage_finite,
                "physiological": dng_voltage_physiological,
                "valid_dynamics": dng_dynamics_valid,
            },
            "recruitment": recruited,
            "dng_finite": dng_voltage_finite,
            "dng_physiological": dng_voltage_physiological,
            "dng_dynamics_valid": dng_dynamics_valid,
            "network_dynamics_valid": all_network_voltages_valid,
            "all_dynamics_valid": bool(dng_dynamics_valid and all_network_voltages_valid),
            "dominant_frequency": None,
            "rhythmicity_score": None,
            "legs": {},
        }
        report.update(permissions)
        any_osc = False
        any_tonic = False
        any_explode = dng_v_exploding or not all_network_voltages_valid
        n_osc_legs = 0
        for slot in LEG_SLOTS:
            traces = self.legs.get(slot) or {}
            row = {}
            leg_osc = False
            for role, trace in traces.items():
                scored = _window_score(trace, stim, dt_ms) if fft_ok else _window_census(trace, stim)
                row[role] = scored
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
        report["valid_dynamics"] = bool(dng_dynamics_valid and not any_explode)
        report["fft_executed"] = bool(dng_spikes.get("fft_executed") or any(metric.get("fft_executed", False) for row in report["legs"].values() for metric in row.values() if isinstance(metric, dict)))
        report["core_tonic_plateau"] = bool(any_tonic and not any_osc and not any_explode)
        report["physiological_v_lower"] = PHYSIOLOGICAL_V_LOWER_MV
        report["physiological_v_upper"] = PHYSIOLOGICAL_V_UPPER_MV
        report["physiological_bound_is_debug_guardrail"] = True
        report["explosion_abs_mv"] = EXPLOSION_ABS_MV
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


def _windowed(trace: np.ndarray, stim: np.ndarray) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    if stim.size == x.size and np.any(stim):
        x = x[stim]
    return x


def _window_census(trace: np.ndarray, stim: np.ndarray) -> dict:
    return _trace_stats(_windowed(trace, stim))


def _window_score(trace: np.ndarray, stim: np.ndarray, dt_ms: float) -> dict:
    return rhythmicity_score(_windowed(trace, stim), dt_ms)


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
        network_voltage_valid=np.zeros(n_steps, dtype=bool),
        dng_voltage_valid=np.zeros(n_steps, dtype=bool),
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
    if rec.network_voltage_valid is not None:
        rec.network_voltage_valid[t] = voltage_is_physiological(net.v)
    dng_idx = circuit.indices("DNg100")
    if rec.dng_voltage_valid is not None:
        rec.dng_voltage_valid[t] = bool(dng_idx.size and voltage_is_physiological(net.v[dng_idx]))
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


def _role_flag(summary: dict, role: str, field: str) -> bool:
    legs = summary.get("legs") or {}
    return any(bool((row.get(role) or {}).get(field)) for row in legs.values())


def interpret_intact(summary: dict) -> dict:
    n_osc = int(summary.get("n_oscillatory_legs") or 0)
    tonic = bool(summary.get("core_tonic_plateau"))
    exploding = bool(summary.get("any_exploding") or summary.get("dng100_voltage_exploding"))
    e1_osc = _role_flag(summary, "E1", "oscillatory")
    e2_osc = _role_flag(summary, "E2", "oscillatory")
    mn_osc = _role_flag(summary, "MN", "oscillatory")
    valid = bool(summary.get("valid_for_rhythm_analysis", False)) and not exploding
    # MN-only oscillation is not the published E1/E2 CPG mechanism.
    reproduced = bool(valid and n_osc >= 1 and (e1_osc and e2_osc))
    if exploding or not valid:
        answer = "no"
        next_step = (
            "DNg100 stimulation produced an unstable explosion or "
            "non-physiological voltage, not a 7–15 Hz rhythm. dominant_frequency "
            "is invalid. Do not run lesions. Run isolated DNg100 LIF sanity "
            "(W=0) before another MaleCNS load. Do not change the graph."
        )
    elif reproduced:
        answer = "yes"
        next_step = (
            "Oscillation is present under DNg100 current in both E1 and E2. Compare "
            "E1/E2/I1/I2 lesions to Pugliese et al. 2025 bioRxiv, then consider MN→FlyBody."
        )
    elif mn_osc and not (e1_osc and e2_osc):
        answer = "no"
        next_step = (
            "Some MN traces are non-flat, but E1 and E2 rhythmicity are both "
            "zero. That is not evidence the published E1/E2 CPG generated a "
            "rhythm. Do not interpret dominant_frequency. Do not change the graph."
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
        "valid_dynamics": bool(summary.get("all_dynamics_valid", summary.get("valid_dynamics", False))) and not exploding,
        "valid_for_rhythm_analysis": valid,
        "allow_lesions": valid,
        "e1_rhythmic": e1_osc,
        "e2_rhythmic": e2_osc,
        "graph_modified": False,
        "dynamics_retuned": False,
        "next_step": next_step,
    }

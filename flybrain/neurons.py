"""Modeling constants that are NOT measured by the MaleCNS wiring graph.

The Feather files give topology, synapse counts, annotations, and predicted
transmitters. They do not specify membrane time constants, reversal potentials,
synaptic gain, delays, or receptor identity. Values below follow the publicly
documented coarse LIF used by Shiu et al. / DoomFly so the simulator is
reproducible, not because they are MaleCNS ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Drosophila central-synapse sign convention used by several connectome LIF
# models: acetylcholine excitatory; GABA, glutamate, and histamine inhibitory.
# Unclear / modulator-only / missing transmitters default to excitatory.
# This is a modeling policy, not a receptor-level measurement.
INHIBITORY_NT = frozenset({"gaba", "glutamate", "histamine"})
EXCITATORY_NT = frozenset(
    {"acetylcholine", "dopamine", "octopamine", "serotonin", "unclear"}
)


def nt_sign(name: str | None) -> int:
    key = str(name or "missing").strip().lower()
    if key in INHIBITORY_NT:
        return -1
    return 1


NT_SIGN = nt_sign


# SHIU_LIF_SANITY_MODEL.
# Current-based LIF expressed entirely in mV/ms.
# Used for numerical and connectome-integration sanity checks.
# NOT the Pugliese CPG dynamical model.
#
# MixedDynamicsNetwork stores voltage in millivolts. The Shiu/DoomFly
# reference implementation stores volts internally and multiplies by 1e3
# only when plotting. Never mix -52e-3 into this solver.
SHIU_LIF_SANITY_MODEL = "shiu_lif_sanity_v1"
PUGLIESE_CPG_MODEL = "pugliese_cpg_v1"

VOLTAGE_UNIT = "mV"
V_REST_MV = -52.0
V_RESET_MV = -52.0
V_THRESHOLD_MV = -45.0
TAU_M_MS = 20.0
TAU_SYN_MS = 5.0
T_REF_MS = 2.2
DELAY_MS = 1.8
# Millivolts per anatomical synapse. Connectivity weight is synapse *count*.
WSYN_MV = 0.275
SYNAPTIC_STEP_MV = WSYN_MV
# PUGLIESE_CPG_MODEL rate-ODE stimulus. Not a SHIU_LIF current and not mV.
PUGLIESE_CPG_STIM_AMPLITUDE = 250.0

# Debug guardrail for SHIU_LIF_SANITY_MODEL. Not a measured Drosophila bound.
PHYSIOLOGICAL_V_LOWER_MV = -100.0
PHYSIOLOGICAL_V_UPPER_MV = 40.0
EXPLOSION_ABS_MV = 150.0


def voltages_finite(v) -> bool:
    """Numerical validity: no NaN/Inf."""
    x = np.asarray(v, dtype=np.float64)
    if x.size == 0:
        return True
    return bool(np.all(np.isfinite(x)))


def voltage_is_physiological(v) -> bool:
    """Membrane validity: finite and inside the Shiu-LIF debug band (mV)."""
    x = np.asarray(v, dtype=np.float64)
    if x.size == 0:
        return True
    if not voltages_finite(x):
        return False
    return bool(float(np.min(x)) >= PHYSIOLOGICAL_V_LOWER_MV and float(np.max(x)) <= PHYSIOLOGICAL_V_UPPER_MV)


def valid_dynamics(v) -> bool:
    """Numerical dynamics: finite and |V| below the explosion abs cutoff."""
    x = np.asarray(v, dtype=np.float64)
    if x.size == 0:
        return True
    if not voltages_finite(x):
        return False
    return bool(float(np.max(np.abs(x))) < EXPLOSION_ABS_MV)


@dataclass(frozen=True)
class LIFParams:
    """SHIU_LIF_SANITY_MODEL current-based LIF.

    Units: millivolts and milliseconds. Not PUGLIESE_CPG_MODEL (that JAX/ODE
    model uses gain, threshold, tau, firing-rate cap, and cell-size
    normalization). Rest and reset are both −52 mV; threshold is −45 mV.
    `contact_gain` is Wsyn = 0.275 mV per anatomical synapse.
    """

    v_rest: float = V_REST_MV
    v_threshold: float = V_THRESHOLD_MV
    tau_m: float = TAU_M_MS
    tau_g: float = TAU_SYN_MS
    t_ref: float = T_REF_MS
    delay: float = DELAY_MS
    contact_gain: float = WSYN_MV
    dt: float = 0.1

    def __post_init__(self) -> None:
        if abs(self.v_rest) < 1.0 or abs(self.v_threshold) < 1.0:
            raise ValueError(
                "LIFParams are millivolts (V_rest=-52.0), not volts (-0.052). "
                "Do not mix Shiu plotting units into the solver."
            )
        if self.dt <= 0 or self.tau_m <= 0 or self.tau_g <= 0:
            raise ValueError("Time constants and dt must be positive.")
        if self.t_ref < 0 or self.delay < 0:
            raise ValueError("Refractory period and delay cannot be negative.")
        if self.contact_gain <= 0:
            raise ValueError("Contact gain must be positive.")

    @property
    def voltage_unit(self) -> str:
        return VOLTAGE_UNIT

    @property
    def wsyn_mv(self) -> float:
        return float(self.contact_gain)

    @property
    def model_id(self) -> str:
        return SHIU_LIF_SANITY_MODEL

    @property
    def model_label(self) -> str:
        return SHIU_LIF_SANITY_MODEL

    @property
    def alpha_v(self) -> float:
        return float(__import__("math").exp(-self.dt / self.tau_m))

    @property
    def alpha_g(self) -> float:
        return float(__import__("math").exp(-self.dt / self.tau_g))

    @property
    def coupling(self) -> float:
        # Analytic current-based synapse coupling used by the DoomFly kernel.
        return (self.alpha_v - self.alpha_g) / (self.tau_m / self.tau_g - 1.0) * (
            self.tau_m / self.tau_g
        ) if abs(self.tau_m - self.tau_g) > 1e-9 else self.dt * self.alpha_v

    @property
    def delay_slots(self) -> int:
        return max(1, int(round(self.delay / self.dt)))

    @property
    def refractory_steps(self) -> int:
        return max(0, int(round(self.t_ref / self.dt)))


# Default published coarse constants (modeling assumption).
LIF_PARAMS = LIFParams()

# DoomFly uses coupling = (exp(-dt/20)-exp(-dt/5))/3 at dt=0.1, tau_m=20, tau_g=5.
# That equals (av-ag)/(tau_m/tau_g - 1) * something? They hardcode /3 because
# tau_m/tau_g = 4, and (4-1)=3. Keep an identical schedule when those taus match.


def shiu_lif_params(*, dt: float = 1.0) -> LIFParams:
    """Organism-step Shiu LIF (mV). dt=1 ms matches the closed-loop tick."""
    return LIFParams(
        v_rest=V_REST_MV,
        v_threshold=V_THRESHOLD_MV,
        tau_m=TAU_M_MS,
        tau_g=TAU_SYN_MS,
        t_ref=T_REF_MS,
        delay=DELAY_MS,
        contact_gain=WSYN_MV,
        dt=dt,
    )


def shiu_coupling(params: LIFParams) -> float:
    """Match the audited DoomFly/Shiu current-based coupling at these taus."""
    import math

    av = math.exp(-params.dt / params.tau_m)
    ag = math.exp(-params.dt / params.tau_g)
    ratio = params.tau_m / params.tau_g
    return (av - ag) / (ratio - 1.0) if abs(ratio - 1.0) > 1e-9 else params.dt * av

"""Modeling constants that are NOT measured by the MaleCNS wiring graph.

The Feather files give topology, synapse counts, annotations, and predicted
transmitters. They do not specify membrane time constants, reversal potentials,
synaptic gain, delays, or receptor identity. Values below follow the publicly
documented coarse LIF used by Shiu et al. / DoomFly so the simulator is
reproducible, not because they are MaleCNS ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True)
class LIFParams:
    """Current-based leaky-integrate-and-fire parameters.

    Units: millivolts and milliseconds. `dt` is the integration step.
    """

    v_rest: float = -52.0
    v_threshold: float = -45.0
    tau_m: float = 20.0
    tau_g: float = 5.0
    t_ref: float = 2.2
    delay: float = 1.8
    contact_gain: float = 0.275
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.tau_m <= 0 or self.tau_g <= 0:
            raise ValueError("Time constants and dt must be positive.")
        if self.t_ref < 0 or self.delay < 0:
            raise ValueError("Refractory period and delay cannot be negative.")
        if self.contact_gain <= 0:
            raise ValueError("Contact gain must be positive.")

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


def shiu_coupling(params: LIFParams) -> float:
    """Match the audited DoomFly/Shiu current-based coupling at these taus."""
    import math

    av = math.exp(-params.dt / params.tau_m)
    ag = math.exp(-params.dt / params.tau_g)
    ratio = params.tau_m / params.tau_g
    return (av - ag) / (ratio - 1.0) if abs(ratio - 1.0) > 1e-9 else params.dt * av

"""MaleCNS graph loading and an approximate LIF simulator.

The connectome specifies directed wiring and synapse counts. Membrane
constants, signs, and the keyboard interface are modeling choices — see
ASSUMPTIONS.md.
"""

from flybrain.loader import Connectome, import_malecns, load_connectome
from flybrain.network import LIFNetwork, LIFParams
from flybrain.neurons import NT_SIGN, LIF_PARAMS
from flybrain.plasticity import RewardModulatedPlasticity

__all__ = [
    "Connectome",
    "LIFNetwork",
    "LIFParams",
    "NT_SIGN",
    "LIF_PARAMS",
    "RewardModulatedPlasticity",
    "import_malecns",
    "load_connectome",
]

"""MaleCNS graph loading and an approximate LIF simulator.

The connectome specifies directed wiring and synapse counts. Membrane
constants, signs, and the keyboard interface are modeling choices — see
ASSUMPTIONS.md.
"""

from flybrain.loader import Connectome, import_malecns, load_connectome
from flybrain.mushroom_body import MushroomBodyLearning
from flybrain.network import LIFNetwork, LIFParams
from flybrain.neuron_model import NeuronKind, ParameterProvenance
from flybrain.neurons import NT_SIGN, LIF_PARAMS
from flybrain.plasticity import RewardModulatedPlasticity

__all__ = [
    "Connectome",
    "LIFNetwork",
    "LIFParams",
    "NT_SIGN",
    "LIF_PARAMS",
    "RewardModulatedPlasticity",
    "MushroomBodyLearning",
    "NeuronKind",
    "ParameterProvenance",
    "import_malecns",
    "load_connectome",
]

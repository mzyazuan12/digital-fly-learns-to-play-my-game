from flybrain.neuron_model import NeuronKind, assign_neuron_models
from organism.toy import miniature_connectome
from flybrain.network import LIFNetwork, LIFParams


def test_not_every_cell_is_identical_lif():
    graph = miniature_connectome(0)
    table = assign_neuron_models(graph)
    assert table.graded_mask().sum() >= 1
    assert table.spiking_mask().sum() >= 1
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    assert net.is_graded.any()
    assert net.models.notes["identical_lif_for_all_cells"] is False
    assert NeuronKind.GRADED_RATE.value in set(table.kind)

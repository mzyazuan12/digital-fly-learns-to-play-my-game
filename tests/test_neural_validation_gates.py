import numpy as np
import pytest
from flybrain.lif_sanity import synapse_count_scaling
from organism.cpg_rhythm import allocate_traces, rhythmicity_score
from organism.toy import miniature_connectome
from organism.walking_pathways import WalkingCircuit


def recording():
    rec = allocate_traces(WalkingCircuit(miniature_connectome(1)), 400)
    rec.t_ms[:] = np.arange(400)
    rec.dng100_v[:] = rec.dng100_l_v[:] = rec.dng100_r_v[:] = -52
    rec.network_voltage_valid[:] = True
    rec.dng_voltage_valid[:] = True
    signal = .5 + .4*np.sin(2*np.pi*10*np.arange(400)/1000)
    rec.dng100[:] = signal
    for row in rec.legs.values():
        for values in row.values(): values[:] = signal
    return rec


@pytest.mark.parametrize('invalid', ['network', 'dng', 'silent_e2'])
def test_invalid_or_unrecruited_circuit_never_calls_fft(monkeypatch, invalid):
    rec = recording()
    if invalid == 'network': rec.network_voltage_valid[0] = False
    if invalid == 'dng': rec.dng_voltage_valid[10] = False
    if invalid == 'silent_e2':
        for row in rec.legs.values(): row['E2'][:] = 0
    def forbidden(*a, **kw): raise AssertionError('FFT before validation')
    monkeypatch.setattr(np.fft, 'rfft', forbidden)
    report = rec.summary(1, 20)
    assert report['valid_for_rhythm_analysis'] is False
    assert report['allow_lesions'] is False
    assert report['fft_executed'] is False
    assert report['dominant_frequency'] is None


def test_valid_recruited_circuit_can_be_measured():
    report = recording().summary(1, 0)
    assert report['valid_for_rhythm_analysis'] is True
    assert report['fft_executed'] is True


def test_frequency_outside_expected_band_is_not_forced():
    trace = .5+.4*np.sin(2*np.pi*40*np.arange(2000)/1000)
    assert rhythmicity_score(trace, 1)['dominant_frequency'] == pytest.approx(40)


def test_scaling_is_subthreshold_at_identical_kernel_time():
    report = synapse_count_scaling()
    for rec in report['by_transmitter'].values():
        assert rec['subthreshold'] and rec['same_kernel_time']
        assert 1.5 < rec['ratio_10_over_5'] < 2.5
        assert 1.5 < rec['ratio_20_over_10'] < 2.5


def test_full_graph_refused_before_loading(monkeypatch):
    from experiment import dng100_cpg_rhythm as experiment
    monkeypatch.setattr(experiment, 'load_graph', lambda *a: pytest.fail('full graph loaded'))
    with pytest.raises(RuntimeError, match='locked'):
        experiment.run(connectome='malecns')

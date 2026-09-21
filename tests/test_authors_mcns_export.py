"""The shipped 4,310-row export is the mCNS reproduction source."""

import json
from pathlib import Path

import numpy as np
import pytest

from flybrain.authors_mcns import (
    NOT_THE_TABLE_MEDIAN,
    POPULATION_AUTHORS,
    POPULATION_RECONSTRUCTION,
    STIMULUS_AUTHORS_MCNS,
    STIMULUS_MANC_AMPLITUDE_CONTROL,
    WTABLE_PATH,
    export_summary,
    load_authors_wtable,
)
from flybrain.neurons import PUGLIESE_THRESHOLD_MEAN


pytestmark = pytest.mark.skipif(not WTABLE_PATH.exists(), reason="authors mCNS export is not in this checkout")


def test_shipped_table_median_is_not_body_20129_size():
    summary = export_summary()
    assert summary["population"] == POPULATION_AUTHORS
    assert summary["not_population"] == POPULATION_RECONSTRUCTION
    assert summary["n"] == 4310
    assert summary["size"]["positive_nonmissing"] == 4310
    assert summary["size"]["missing_or_nonfinite"] == 0
    assert summary["size"]["zero"] == 0
    assert summary["size"]["nanmedian"] == pytest.approx(998_603_428.5)
    assert summary["size"]["nanmedian"] != pytest.approx(float(NOT_THE_TABLE_MEDIAN))
    counts = summary["consensusNt_counts"]
    assert counts["acetylcholine"] == 2493
    assert counts["gaba"] == 1011
    assert counts["glutamate"] == 606
    assert counts["unclear"] == 193
    assert counts["<blank>"] == 1
    assert counts["histamine"] == 4
    assert counts["serotonin"] == 2
    assert "acetylcholine" not in summary["membership_rule"] or "not a" in summary["membership_rule"]
    right = summary["dng100"]["10056"]
    left = summary["dng100"]["10045"]
    assert right["instance"] == "DNg100_R"
    assert right["matrix_index"] == 9
    assert right["size"] == pytest.approx(16_542_412_146)
    assert left["size"] == pytest.approx(16_688_976_137)
    assert right["normalized_size"] == pytest.approx(16_542_412_146 / 998_603_428.5)
    assert right["mean_threshold_after_set_sizes"] == pytest.approx(
        PUGLIESE_THRESHOLD_MEAN * right["normalized_size"]
    )
    assert 250.0 > right["mean_threshold_after_set_sizes"]
    unclear = summary["outgoing_by_consensusNt"]["unclear"]
    blank = summary["outgoing_by_consensusNt"]["<blank>"]
    assert unclear["outgoing_positive"] == 0 and unclear["outgoing_negative"] == 0
    assert blank["outgoing_positive"] == 0 and blank["outgoing_negative"] == 0
    assert summary["W_n_negative"] > 0
    assert summary["W_n_positive"] > 0


def test_set_sizes_on_the_shipped_column_uses_its_nanmedian():
    from flybrain.pugliese_rate import set_sizes

    table = load_authors_wtable()
    sizes = table["size"].to_numpy(dtype=np.float64)
    gain = np.ones((1, sizes.shape[0]))
    threshold = np.full((1, sizes.shape[0]), PUGLIESE_THRESHOLD_MEAN)
    _, scaled = set_sizes(sizes, gain, threshold)
    index = int(np.flatnonzero(table["bodyId"].to_numpy() == 10056)[0])
    assert scaled[0, index] == pytest.approx(PUGLIESE_THRESHOLD_MEAN * sizes[index] / np.nanmedian(sizes))


def test_i250_port_check_is_a_manc_amplitude_control(tmp_path):
    from experiment.pugliese_rate_mcns_portcheck import preflight, run

    summary = export_summary()
    header = preflight(summary, stim_amplitude=250.0)
    assert header["stimulus_source"] == STIMULUS_MANC_AMPLITUDE_CONTROL
    assert header["stimulus_source"] != STIMULUS_AUTHORS_MCNS
    assert header["is_authors_mcns_reproduction"] is False
    assert header["population"] == POPULATION_AUTHORS
    assert header["mean_threshold_versus_stimulus"]["I_exceeds_mean_threshold"] is True
    manc = tmp_path / "manc"
    manc.mkdir()
    (manc / "report.json").write_text(json.dumps({"port_ok": True, "status": "manc_port_ok"}) + "\n")
    out = tmp_path / "mcns"
    report = run(out, manc_portcheck=manc, n_replicates=1, integrate=False)
    saved = json.loads((out / "report.json").read_text())
    assert saved["stimulus_source"] == STIMULUS_MANC_AMPLITUDE_CONTROL
    assert saved["is_authors_mcns_reproduction"] is False
    assert saved["status"] == "preflight_only"
    assert report["sampled_stimulated_threshold"]["I_exceeds_sampled_threshold_replicate_0"] is True
    with pytest.raises(ValueError, match="I=250"):
        preflight(summary, stim_amplitude=400.0)


def test_body_id_diff_uses_the_shipped_ids():
    path = Path("data/malecns_v1/mcns_body_id_diff.json")
    if not path.exists():
        pytest.skip("body-id diff has not been written")
    diff = json.loads(path.read_text())
    assert diff["authors_population"] == POPULATION_AUTHORS
    assert diff["reconstruction_population"] == POPULATION_RECONSTRUCTION
    assert diff["authors_n"] == 4310
    assert diff["reproduction_source"] == POPULATION_AUTHORS
    assert diff["reconstruction_is_denominator"] is False
    assert diff["intersection"] + len(diff["authors_only"]) == 4310
    assert diff["intersection"] + len(diff["ours_only"]) == diff["reconstruction_n"]

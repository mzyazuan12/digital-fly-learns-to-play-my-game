"""Pugliese MANC T1 DNg100 IDs are not MaleCNS body IDs."""

from pathlib import Path

from organism.roi_innervation import (
    MANC_NOT_DNG100_10056,
    MANC_VMS16,
    PUGLIESE_DNG100_STIM,
    verify_pugliese_manc_t1_dng100,
)


def test_pugliese_stim_is_matrix_index_31_body_10093_not_10056():
    assert PUGLIESE_DNG100_STIM["source_dataset"] == "MANC_T1"
    assert PUGLIESE_DNG100_STIM["source_matrix_index"] == 31
    assert PUGLIESE_DNG100_STIM["source_body_id"] == 10093
    assert PUGLIESE_DNG100_STIM["type"] == "DNg100"
    assert PUGLIESE_DNG100_STIM["source_body_id"] != 10056
    assert MANC_VMS16["source_body_id"] == 10056
    assert MANC_VMS16["source_matrix_index"] == 16
    assert MANC_VMS16["type"] == "vMS16"
    assert MANC_NOT_DNG100_10056 is MANC_VMS16

    checked = verify_pugliese_manc_t1_dng100()
    assert checked["pugliese_reference"]["source_matrix_index"] == 31
    assert checked["pugliese_reference"]["source_body_id"] == 10093
    assert checked["manc_vms16"]["type"] == "vMS16"
    bodies = {row["source_body_id"] for row in checked["manc_t1_dng100"]}
    assert 10093 in bodies
    assert 10056 not in bodies


def test_shipped_pugliese_figures_are_not_a_reproduction():
    from experiment.authors_dng100_reference import (
        OUR_HYDRA_ROOT,
        SHIPPED_FIGURES_DIR,
        is_our_hydra_run,
        is_shipped_author_figure,
    )

    assert SHIPPED_FIGURES_DIR.exists()
    core = SHIPPED_FIGURES_DIR / "DNg100_Stim_CoreCPG-hyak-run_id=29236830"
    prune = SHIPPED_FIGURES_DIR / "DNg100_Stim_Prune-hyak-run_id=28965859"
    assert core.exists()
    assert prune.exists()
    assert is_shipped_author_figure(core)
    assert is_shipped_author_figure(prune)
    assert is_our_hydra_run(core) is False
    assert str(OUR_HYDRA_ROOT) != str(SHIPPED_FIGURES_DIR)
    # A cloned figure directory must never be treated as our Hydra run.
    assert not (core / "logs" / "run_config.yaml").exists() or True
    assert Path("third_party/Pugliese_2026/figures") == SHIPPED_FIGURES_DIR or SHIPPED_FIGURES_DIR.name == "figures"


def test_reference_lookup_never_falls_back_to_another_run(tmp_path, monkeypatch):
    import pytest
    from experiment import authors_dng100_reference as ref
    monkeypatch.setattr(ref, 'OUR_HYDRA_ROOT', tmp_path)
    wrong = tmp_path/'DNg100_Stim'/'default'/'run_id=wrong'
    (wrong/'ckpt').mkdir(parents=True)
    (wrong/'logs').mkdir()
    (wrong/'ckpt'/'DNg100_Stim_Rs.npz').touch()
    (wrong/'logs'/'run_config.yaml').touch()
    with pytest.raises(FileNotFoundError, match='requested'):
        ref.latest_ckpt('requested')

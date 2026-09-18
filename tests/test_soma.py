from pathlib import Path

import numpy as np

from organism.soma import annotation_path, load_soma_xyz, lookup_xyz, to_display_xyz


def test_malecns_somata_are_em_voxels():
    if not annotation_path().exists():
        return
    soma = load_soma_xyz()
    assert soma["body_id"].size > 100_000
    assert soma["xyz"].shape[1] == 3
    assert np.isfinite(soma["xyz"]).all()
    # Giant Fiber / DNp01 body 10001 from the annotation table.
    xyz = lookup_xyz(np.array([10001], dtype=np.uint64), soma)
    assert np.isfinite(xyz).all()
    disp, meta = to_display_xyz(xyz)
    assert disp.shape == (1, 3)
    assert meta["voxel_nm"] == 8.0
    cache = Path(__file__).resolve().parents[1] / "data/malecns_v1/normalized/soma_xyz.npz"
    assert cache.exists()

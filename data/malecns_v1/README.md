# MaleCNS v1.0 inputs

Official source: https://male-cns.janelia.org/download/

Required files (already present in this checkout, also symlinked here):

- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`
- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `body-neurotransmitters-male-cns-v1.0.feather`

Optional anatomy (Janelia GCS, gitignored; `python scripts/download_malecns_anatomy.py`):

- `syn-points-male-cns-v1.0-minconf-0.5.feather` — reconstructed pre/post XYZ
- `syn-partners-male-cns-v1.0-minconf-0.5.feather` — pre→post partner pairs
- `skeletons-swc/` — SWC centerlines for retained MaleCNS body IDs

Do not download EM imagery or segmentation volumes for runtime.

The importer writes a compact sparse cache to `normalized/` (gitignored).
That cache is derived data, not a new biological measurement.

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

CPG leg identity is **not** in `graph.npz`. After
`python scripts/cache_malecns_roi_innervation.py`:

- `cpg_roi_innervation.parquet` — per-bodyId LegNp PreSyn/PostSyn counts
- `cpg_mapping.json` — provenance-heavy E1/E2/I1… records with
  `malecns_body_id`, per-segment `roi_counts`, `assigned_segment`,
  `fallback_used=false`
- `cpg_mapping.INVALID_pre_roi.json` — archived previous mapping, kept as a
  forensic trail (do not use)
- `neuron_metadata.parquet` — optional MaleCNS bodyId → graph index

Pugliese `DNg100_Stim` is MANC T1 **matrix index 31 / body 10093 / type DNg100**.
In that same MANC table, body 10056 is vMS16. MaleCNS DNg100 is resolved
independently from `annotations[type == "DNg100"]` (one left, one right).
The same integer can exist in both datasets as different cells.
I2 is `IN19B007`, not `IN19A007`.

Those files assign each neuron from its own synapses. Type-level ROI
pages pool T1+T2+T3 and must not be used. Soma-Z is not used.

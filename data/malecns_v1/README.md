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

CPG / DNg100 **identity** is the official annotation feather, not a derived
parquet:

```
body-annotations-male-cns-v1.0-minconf-0.5.feather
  column bodyId   →  malecns_body_id
syn-points-male-cns-v1.0-minconf-0.5.feather
  column body     →  malecns_body_id
```

`MaleCNS DNg100` is `annotations[type == "DNg100"]` (exactly two rows:
bodyId 10045 L, bodyId 10056 R). Curated `mancBodyid` / `mancType` are
correspondence to a different specimen, not integer identity. Pugliese
`DNg100_Stim` is MANC T1 **source_matrix_index 31 / source_body_id 10093 /
type DNg100** and lives only under `DNg100.pugliese_reference`. In that MANC
table, body 10056 is vMS16 — a different cell from MaleCNS DNg100_R 10056.
MaleCNS bodyId 10093 is Am1. I2 is `IN19A007` (six copies), not `IN19B007`.

After `python scripts/cache_malecns_roi_innervation.py --write-mapping`:

- `cpg_roi_innervation.parquet` — derived join of those two tables
- `cpg_mapping.json` — `malecns_body_id`, per-segment `roi_counts`,
  `assigned_segment`, `fallback_used=false`
- `cpg_mapping.INVALID_pre_roi.json` / `cpg_mapping.INVALID_pre_namespace_fix.json`
  / `male_cpg.INVALID_pre_namespace_fix.parquet` — forensic archives (do not use)
- `neuron_metadata.parquet` — optional MaleCNS bodyId → graph index

Those files assign each neuron from its own synapses. Type-level ROI
pages pool T1+T2+T3 and must not be used. Soma-Z is not used.

Pugliese rate-model size cache (`python scripts/cache_pugliese_rate_sizes.py`):

- `pugliese_rate_sizes.parquet` — neuPrint `Neuron.size` (voxel volume) for the
  408-cell restricted CPG graph, with `median_volume_reference` equal to the
  median of all male-cns:v1.0 neurons that have a positive size. Do not use SWC
  skeletons, synapse counts, cable length, soma size, or partner count as size.
- `pugliese_rate_sizes.meta.json` — query provenance for that parquet

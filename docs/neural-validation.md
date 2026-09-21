Neural validation workflow

The current experiment record is [the 19 September 2026 report](../outputs/neural_milestone_20260919/REPORT.md). It records successful numerical sanity and a small original-code Pugliese reproduction, alongside the failed left-side MaleCNS voltage gate. Full-network and embodiment work are not validated.

Run from the repository root using its Python environment:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiment.dng100_cpg_rhythm --sanity
.venv/bin/python -m experiment.validate_neural_milestone
```

The biological validator creates a new timestamped directory, scans raw synaptic ROI data in batches, saves the mapping, builds a restricted biological graph, and runs each DNg100 side separately. Exit status 2 means a scientific gate failed; inspect its report and preserved traces. No full connectome simulation is launched.

The MANC port of the half-tanh ODE has passed. The silent 408-cell MaleCNS run used the median of all male-cns neurons with a positive size. That is a control, not Pugliese's mCNS convention. Their `set_sizes()` uses `nanmedian` of the simulation table. That table is the shipped export `wTable_20260210_vncRoisOnly.csv` (4,310 rows, commit `10e7661`), paired with `W_20260210_vncRoisOnly.csv`. The MaleCNS v1.0 reconstruction is an audit, not this population.

The shipped size column has a positive value on every row. Its `nanmedian` is 998,603,428.5. Body 20129's size is 1,010,837,052; that number is not the table median.

```text
MANC PORT CHECK                         passed
PUGLIESE ODE IMPLEMENTATION             passed
MALECNS VOLUME PROPERTY                 passed
FULL-MALECNS-MEDIAN CONTROL             silent; valid control, not replication
AUTHOR mCNS 4,310 POPULATION            authors export, commit 10e7661
AUTHOR mCNS SIZE MEDIAN                 998,603,428.5
AUTHOR mCNS W MATRIX                    shipped, already signed, aligned
AUTHOR mCNS STIMULUS AMPLITUDE          unresolved
mCNS PORT @ I=250 CONTROL               10056 reaches 7.5 Hz; no E1/E2 rhythm; MANC amplitude control
EXACT mCNS PUBLISHED REPRODUCTION       locked until mCNS stimI is recovered
RESTRICTED-408 TRANSFER                 locked
CONNECTIVITY DIFFERENCE DISSECTION      locked
FULL MALECNS                            locked
BODY                                    locked
v1.0 RECONSTRUCTION AUDIT               not the reproduction source
```

```sh
.venv/bin/python scripts/cache_pugliese_rate_sizes.py
.venv/bin/python -m experiment.pugliese_rate_manc_portcheck
.venv/bin/python scripts/reconstruct_pugliese_mcns_population.py
.venv/bin/python -m experiment.pugliese_rate_mcns_portcheck --manc-portcheck outputs/pugliese_rate_manc_portcheck_20260919T133800Z
```

The restricted-408 runner stays on the full-median control unless `--size-norm` says otherwise. `pugliese_simulation_population_median` is refused on that 408-cell graph until the published mCNS stimulus amplitude is recovered. `restricted_408_median_control` is not a reference. MaleCNS 10056 is the stimulated cell because its curated `mancBodyid` is 10093 and its LegNp innervation is left, despite somaSide R. On the authors' export, I=250 is labeled `MANC amplitude control`, not `authors_mcns`. Existing output directories are refused. A passing control does not unlock full MaleCNS.

For the original-code reference, choose a new run ID:

```sh
.venv/bin/python -m experiment.authors_dng100_reference --n-replicates 2 --batch-size 1 --run-id YOUR_UNIQUE_RUN_ID
```

The original simulator runs in the existing vnc-sim environment; analysis runs in the project environment. Existing run directories are refused. Its default analysis folder is inside the new Hydra run. Read the reported model and dataset identifiers before comparing experiments.

`experiment.tiny_cpg_lesions` accepts an explicit control directory, Pugliese metrics path, stimulated MaleCNS body, and unused output directory. It refuses a control without lesion permission, checks the saved individual voltage traces, and requires an exactly matching intact replay. See `--help` for arguments. A failed left-side control cannot authorize right-side lesions or vice versa.

The synthetic seven-cell motif remains a plumbing test. It is not the biological MaleCNS subgraph and cannot unlock a full-network run.

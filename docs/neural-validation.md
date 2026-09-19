Neural validation workflow

The current experiment record is [the 19 September 2026 report](../outputs/neural_milestone_20260919/REPORT.md). It records successful numerical sanity and a small original-code Pugliese reproduction, alongside the failed left-side MaleCNS voltage gate. Full-network and embodiment work are not validated.

Run from the repository root using its Python environment:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m experiment.dng100_cpg_rhythm --sanity
.venv/bin/python -m experiment.validate_neural_milestone
```

The biological validator creates a new timestamped directory, scans raw synaptic ROI data in batches, saves the mapping, builds a restricted biological graph, and runs each DNg100 side separately. Exit status 2 means a scientific gate failed; inspect its report and preserved traces. No full connectome simulation is launched.

The next scientific gate transfers published Pugliese rate dynamics onto that restricted MaleCNS graph. It does not retune Shiu LIF:

```sh
.venv/bin/python -m experiment.pugliese_rate_malecns
.venv/bin/python -m experiment.pugliese_rate_malecns --shuffle
```

Primary stimulated cell is MaleCNS DNg100_R `10056` (MANC homolog `10093`). Existing output directories are refused. A passing transfer still does not unlock full MaleCNS.

For the original-code reference, choose a new run ID:

```sh
.venv/bin/python -m experiment.authors_dng100_reference --n-replicates 2 --batch-size 1 --run-id YOUR_UNIQUE_RUN_ID
```

The original simulator runs in the existing vnc-sim environment; analysis runs in the project environment. Existing run directories are refused. Its default analysis folder is inside the new Hydra run. Read the reported model and dataset identifiers before comparing experiments.

`experiment.tiny_cpg_lesions` accepts an explicit control directory, Pugliese metrics path, stimulated MaleCNS body, and unused output directory. It refuses a control without lesion permission, checks the saved individual voltage traces, and requires an exactly matching intact replay. See `--help` for arguments. A failed left-side control cannot authorize right-side lesions or vice versa.

The synthetic seven-cell motif remains a plumbing test. It is not the biological MaleCNS subgraph and cannot unlock a full-network run.

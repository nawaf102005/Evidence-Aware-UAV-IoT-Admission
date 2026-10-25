# Evidence-Aware UAV-IoT Admission — Final Main-Study Source

This repository contains **one maintained experiment entry point** for the final independent 200-layout confirmation of the research paper **“Evidence-Aware Admission of Advance Service Commitments in UAV-Enabled IoT Networks Under Packet-Loss Uncertainty.”**

It deliberately excludes earlier revisions, historical replay programs, plotting revisions, archived experimental outputs, and precomputed result tables. It contains only the final main-study driver (`run_latest.py`), its transitively imported implementation modules, and focused unit tests. The modules are retained because the final driver imports them; they are **not** alternate published algorithm versions.

## Installation

Python 3.10+ is recommended. From a virtual environment:

```bash
python -m pip install -r requirements.txt
```

## Tests

```bash
python -m unittest test_latest test_memory_certificates -q
```

## Quick smoke test (not for publication)

```bash
python run_latest.py --limit 1 --workers 1 --repetitions 8
```

This writes `results/pilot_admission_*.{csv,json}` and only tests executable pathways, not the published reliability conclusions.

## Final independent experiment

```bash
python run_latest.py --workers 4 --repetitions 2048
```

The declared 200 independently generated layouts use seeds 8000001–8000200. The script writes `results/final_admission_200_records.csv` and `results/final_admission_200_protocol.json`. These are new output filenames for the unchanged study records; do not combine pilot outputs with final study tables. Set `--workers` according to available memory. The full study is compute-intensive and has **not** been rerun in this release curation.

## Scope and limitations

- This source-only release contains **no generated results, plots, models, or raw third-party datasets**.
- The executable reproduces the **main independent confirmation cohort only**. It does **not** regenerate every table or historical supplementary experiment. Additional supplementary code would require a separate, fully validated release with its own fresh-input provenance.
- Statistical guarantees depend on the manuscript's independence, Gaussian-noise, packet-loss model, and drift conditions; the code does not validate them for field deployment.
- Backend modules retain some general-purpose functions originally developed in earlier research stages because the latest driver imports their public interfaces. No obsolete replay/experiment entry points are distributed.
- This archive adds no software license; the owner should select a license before granting public reuse rights.
- The author should compare full-run output against the published manuscript before advertising complete reproducibility.

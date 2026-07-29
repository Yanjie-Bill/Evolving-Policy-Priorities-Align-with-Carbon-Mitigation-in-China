# Evolving Policy Priorities and Carbon Mitigation in China

This folder is a cleaned reproducibility package for the manuscript
`Evolving Policy Priorities Align with Carbon Mitigation in China`.

The original working folder contains many exploratory scripts. This package keeps
the files needed to reproduce the manuscript-level analyses and figures from
frozen intermediate data.

## Folder structure

- `data/raw/`: source or semi-source data copied from the working folder.
- `data/processed/`: frozen analysis tables used by the manuscript figures.
- `scripts/`: path-independent reproduction scripts.
- `outputs/`: generated figures and tables after running the scripts.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py
```

If `geopandas` is not installed, the provincial map is skipped and the other
provincial typology outputs are still generated.

The optional embedding recomputation script requires `sentence-transformers`,
`torch`, and access to the model `BAAI/bge-large-zh-v1.5` through either a local
cache or network download.

## Reproduction map

- `scripts/01_national_policy_emission.py`
  - Reproduces the national policy-emission alignment figure.
  - Recomputes a t-SNE visualization from stored BGE embeddings.
  - Uses `data/processed/national/`.

- `scripts/00_optional_recompute_policy_embeddings.py`
  - Optional provenance step that recomputes BGE embeddings and carbon semantic
    similarity from `data/raw/combined_policy_data.xlsx`.
  - Not included in `run_all.py` because model download and hardware conditions
    are environment-dependent.

- `scripts/02_provincial_typology.py`
  - Reproduces typology radar fingerprints and carbon-intensity reduction
    boxplots.
  - Draws the typology map when `geopandas` is available.
  - Uses `data/processed/provincial/` and the province shapefile in `data/raw/`.

- `scripts/03_policy_effectiveness.py`
  - Summarizes and plots the DID policy-effect screening results.
  - Highlights the New Energy Vehicle Industry Development Plan row.
  - Uses `data/processed/did/`.

- `scripts/04_national_forecast.py`
  - Replots national CO2 forecasting scenarios from the ElasticNet outputs.
  - Uses `data/processed/forecast/`.

- `scripts/05_global_comparison.py`
  - Reproduces global carbon-intensity comparison panels and FAOLEX policy
    subject composition panels.
  - Uses `data/processed/carbon intensity.xlsx` and
    `data/processed/global_policy/`.

## Notes

The embedding-generation step is intentionally not part of the default one-click
pipeline because it depends on `BAAI/bge-large-zh-v1.5`, local model cache or
network access, and hardware. The frozen file
`data/processed/national/policies_with_cluster_labels.csv` contains the policy
texts, carbon semantic similarity scores, BGE embeddings, and cluster labels used
for the manuscript analyses.

Raw policy scraping and manual exception-handling scripts are treated as
provenance, not as the default analysis pipeline, because several source
government websites and downloaded files can change over time.

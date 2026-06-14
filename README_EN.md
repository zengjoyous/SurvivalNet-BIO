# SurvivalNet

SurvivalNet is a TCGA survival-analysis toolkit for expression-based modeling with three survival models:

- Cox
- LASSO-Cox
- DeepSurv

The repository follows a `src/` layout and is designed to be installable with `pip install -e .`.

## Features

- TCGA clinical/expression table loading
- patient-level merging and train/test splitting
- Cox, LASSO-Cox, and DeepSurv training
- repeated CV and one-SE selection
- summary tables, hazard-ratio tables, and risk-stratification plots

## Installation

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
```

## Example Run

```bash
make all
```

Or run the pipeline step by step:

```bash
make prepare
make cox
make lasso
make deepsurv
make summary
make plot
```

## Data

The default example uses the STAD cohort:

- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.clinical.tsv`
- `examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.star_tpm.tsv`

Other example cohorts are available in `examples/`.

## Output

Outputs are written to `output/<dataset name>/`:

- `process/merged.csv`
- `process/train.csv`
- `process/test.csv`
- `final/*_summary.csv`
- `final/*_hazard_ratios.csv`
- `final/*_test_scored.csv`
- `final/model_summary.csv`
- `final/model_comparison.png`

## Notebook

See `examples/demo.ipynb` for an end-to-end demonstration.

## Testing

```bash
pytest
```

## Environment

Use `environment.yml` for a reproducible environment.

## Project Structure

```
project_name/
├── pyproject.toml
├── README.md
├── src/
├── tests/
├── examples/
└── environment.yml
```

## Notes

- Clinical columns can be merged into the analysis table when present.
- DeepSurv uses a smaller, more stable search space in this repository.


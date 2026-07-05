# Workflow

The standard execution order is:

1. `prepare`
2. `cox`
3. `lasso`
4. `deepsurv`
5. `summary`
6. `plot`

The workflow is defined in `Snakefile` and uses values from `config.yaml`.

## Outputs

- `process/merged.csv`
- `process/train.csv`
- `process/test.csv`
- `final/*_summary.csv`
- `final/*_hazard_ratios.csv`
- `final/*_test_scored.csv`
- `final/model_summary.csv`
- `final/model_comparison.png`

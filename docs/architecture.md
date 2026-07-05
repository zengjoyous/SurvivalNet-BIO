# Architecture

The repository uses a layered structure:

- `scripts/` provides command-line entry points
- `src/survivalnet/` contains reusable library code
- `Snakefile` orchestrates the workflow
- `examples/` provides demo data and a notebook
- `tests/` contains pytest coverage

## Core design choices

- Normalize clinical and expression data into a patient-level analysis table
- Filter noisy high-dimensional features before model fitting
- Keep Cox, LASSO-Cox, and DeepSurv on a shared evaluation interface
- Save intermediate and final outputs in separate folders

## No core-code impact

These docs and container files are added alongside the package and do not modify the modeling logic in `src/`.

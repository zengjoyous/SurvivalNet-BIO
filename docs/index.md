# SurvivalNet

SurvivalNet is a TCGA-oriented survival analysis toolkit for comparing Cox, LASSO-Cox, and DeepSurv models.

## What it does

- Load TCGA clinical and expression tables
- Merge them at the patient level
- Split data into train and test sets
- Train Cox, LASSO-Cox, and DeepSurv models
- Select hyperparameters with repeated CV and one-SE rules
- Export summaries, hazard ratios, and survival plots

## Quick Start

```bash
conda env create -f environment.yml
conda activate survivalnet
pip install -e .
snakemake --cores 1
```

## Document Map

- `architecture.md`: implementation overview
- `workflow.md`: end-to-end execution flow
- `api.md`: main public interfaces

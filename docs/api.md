# API

The main public interfaces are exposed from `survivalnet.__init__`.

## Data utilities

- `load_table`
- `load_clinical_data`
- `normalize_survival_data`
- `prepare_feature_matrix`
- `prepare_survival_dataset`
- `split_train_test`
- `split_train_val_test`

## Core survival helpers

- `c_index`
- `fit_km`
- `logrank_p_value`
- `run_logrank_test`
- `split_risk_group`

## Models

- `CoxModel`
- `LassoCoxModel`
- `DeepSurvModel`

## Workflow helpers

- `build_analysis_table`
- `load_input_tables`
- `save_split_tables`
- `summarize_model_results`
- `train_baseline_models`

The full implementation details live in the source files under `src/survivalnet/`.

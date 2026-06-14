"""End-to-end workflow helpers for survival analysis projects."""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import (
    load_expression_matrix,
    load_table,
    prepare_feature_matrix,
    prepare_survival_dataset,
    split_train_test,
)


@dataclass
class WorkflowPaths:
    """Standard output locations for a stomach cancer analysis."""

    output_dir: Path

    @property
    def process_dir(self) -> Path:
        return self.output_dir / "process"

    @property
    def final_dir(self) -> Path:
        return self.output_dir / "final"

    def ensure(self) -> "WorkflowPaths":
        self.process_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        return self


def infer_output_dir(
    reference_path: str | Path,
    *,
    output_root: str | Path,
) -> Path:
    """Derive an output folder name from an input file or directory path."""
    reference = Path(reference_path)
    return Path(output_root) / reference.parent.name


def load_input_tables(
    clinical_path: str | Path,
    expression_path: str | Path,
    clinical_sep: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load clinical and expression tables from disk."""
    clinical_df = load_table(clinical_path, sep=clinical_sep)
    expression_df = load_expression_matrix(expression_path)
    return clinical_df, expression_df


def build_analysis_table(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    *,
    time_col: str = "OS.time",
    status_col: str = "OS",
) -> pd.DataFrame:
    return prepare_survival_dataset(
        clinical_df,
        expression_df,
        time_col=time_col,
        status_col=status_col,
    )


def save_split_tables(
    merged_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    test_size: float = 0.30,
    max_features: int | None = 200,
    stratify_col: str | None = "event",
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split the merged table and persist train/test CSV files."""
    paths = WorkflowPaths(Path(output_dir)).ensure()
    train_df, test_df = split_train_test(
        merged_df,
        test_size=test_size,
        stratify_col=stratify_col,
        random_state=random_state,
    )

    feature_cols: list[str] = []
    if max_features is not None:
        train_model_df, feature_cols = prepare_feature_matrix(
            train_df,
            "duration",
            "event",
            max_features=max_features,
        )
        train_df = train_model_df.reset_index(drop=True)
        test_df = test_df[[col for col in ["duration", "event"] + feature_cols if col in test_df.columns]].copy().reset_index(drop=True)
        merged_df = merged_df[[col for col in ["duration", "event"] + feature_cols if col in merged_df.columns]].copy().reset_index(drop=True)

    merged_df.to_csv(paths.process_dir / "merged.csv", index=False)
    train_df.to_csv(paths.process_dir / "train.csv", index=False)
    test_df.to_csv(paths.process_dir / "test.csv", index=False)

    return {
        "merged": merged_df,
        "train": train_df,
        "test": test_df,
        "selected_features": pd.DataFrame({"feature": feature_cols}) if feature_cols else pd.DataFrame(),
    }


def _select_cox_penalizer_by_cv(
    train_df: pd.DataFrame,
    penalizers: list[float] | None,
    *,
    duration_col: str,
    event_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Pick the Cox penalizer using repeated CV with a coarse-to-fine search."""

    coarse_candidates = penalizers or _default_cox_candidates()
    coarse_candidates = sorted({float(p) for p in coarse_candidates if float(p) > 0})
    repeated_splits = [
        list(
            _iter_stratified_folds(
                train_df,
                n_splits=n_splits,
                stratify_col=event_col,
                random_state=random_state + repeat,
            )
        )
        for repeat in range(3)
    ]

    coarse_results, coarse_fold_results = _evaluate_cox_candidates(
        train_df,
        coarse_candidates,
        duration_col=duration_col,
        event_col=event_col,
        repeated_splits=repeated_splits,
        stage="coarse",
    )

    if coarse_results.empty:
        raise RuntimeError("No Cox penalizer candidate could be fitted successfully.")

    anchor = float(coarse_results.iloc[0]["penalizer"])
    fine_candidates = _build_cox_refinement_grid(anchor)
    fine_candidates = sorted({float(p) for p in fine_candidates if float(p) > 0 and float(p) not in set(coarse_candidates)})

    if fine_candidates:
        fine_results, fine_fold_results = _evaluate_cox_candidates(
            train_df,
            fine_candidates,
            duration_col=duration_col,
            event_col=event_col,
            repeated_splits=repeated_splits,
            stage="fine",
        )
        cv_results = pd.concat([coarse_results, fine_results], ignore_index=True)
        fold_results = pd.concat([coarse_fold_results, fine_fold_results], ignore_index=True)
    else:
        cv_results = coarse_results.copy()
        fold_results = coarse_fold_results.copy()

    cv_results = cv_results.drop_duplicates(subset=["penalizer"], keep="first").reset_index(drop=True)
    cv_results = _rank_cox_results(cv_results)
    best_row, selected_row = _choose_cox_row(cv_results, use_one_se_rule=True)
    selected_penalizer = float(selected_row["penalizer"])

    feature_frequency_source = fold_results[
        (fold_results["penalizer"] == selected_penalizer) & (fold_results["status"] == "ok")
    ].copy()
    feature_frequency_source["selected_features"] = feature_frequency_source["selected_features"].fillna("")
    freq_counter: Counter[str] = Counter()
    for feature_string in feature_frequency_source["selected_features"].tolist():
        if feature_string:
            freq_counter.update([feature for feature in feature_string.split(";") if feature])
    if freq_counter:
        total_success = int(feature_frequency_source.shape[0])
        feature_frequency_table = pd.DataFrame(
            [
                {
                    "feature": feature,
                    "frequency": count,
                    "selection_rate": count / total_success if total_success else np.nan,
                }
                for feature, count in freq_counter.most_common()
            ]
        )
    else:
        feature_frequency_table = pd.DataFrame(columns=["feature", "frequency", "selection_rate"])

    selection_summary: dict[str, Any] = {
        "selected_penalizer": selected_penalizer,
        "selected_rule": "one_se",
        "selected_mean_c_index": float(selected_row["mean_c_index"]),
        "selected_sem_c_index": float(selected_row["sem_c_index"]),
        "selected_mean_selected_features": float(selected_row.get("mean_selected_features", np.nan)),
        "best_penalizer_by_mean": float(best_row["penalizer"]),
        "best_mean_c_index": float(best_row["mean_c_index"]),
        "best_mean_selected_features": float(best_row.get("mean_selected_features", np.nan)),
    }

    return selected_penalizer, cv_results, fold_results, feature_frequency_table, selection_summary


def _iter_stratified_folds(
    data: pd.DataFrame,
    *,
    n_splits: int = 5,
    stratify_col: str | None = "event",
    random_state: int = 42,
):
    """Yield stratified train/validation index splits without sklearn."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(data) < n_splits:
        raise ValueError("n_splits cannot exceed sample size.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(data))

    if stratify_col is None or stratify_col not in data.columns:
        rng.shuffle(indices)
        chunks = np.array_split(indices, n_splits)
        for i in range(n_splits):
            val_idx = chunks[i]
            train_idx = np.concatenate([chunks[j] for j in range(n_splits) if j != i])
            yield train_idx, val_idx
        return

    labels = data[stratify_col].fillna("__nan__").astype(str).to_numpy()
    fold_bins: list[list[int]] = [[] for _ in range(n_splits)]
    for label in pd.unique(labels):
        label_idx = indices[labels == label]
        rng.shuffle(label_idx)
        for fold, chunk in enumerate(np.array_split(label_idx, n_splits)):
            fold_bins[fold].extend(chunk.tolist())

    all_indices = np.arange(len(data))
    for fold in range(n_splits):
        val_idx = np.array(sorted(fold_bins[fold]), dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        yield train_idx, val_idx


def _default_cox_candidates() -> list[float]:
    return [
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
        3e-1,
        1.0,
    ]


def _build_cox_refinement_grid(anchor: float) -> list[float]:
    if anchor <= 0:
        return []

    factors = np.geomspace(0.3, 3.0, num=9)
    grid = {float(anchor * factor) for factor in factors}
    return sorted(round(value, 6) for value in grid if value > 0)


def _rank_cox_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    ranked = results.copy()
    if "mean_selected_features" not in ranked.columns:
        ranked["mean_selected_features"] = np.nan
    return ranked.sort_values(
        ["mean_c_index", "mean_selected_features", "penalizer"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def _choose_cox_row(
    cv_results: pd.DataFrame,
    *,
    use_one_se_rule: bool,
) -> tuple[pd.Series, pd.Series]:
    if cv_results.empty:
        raise RuntimeError("Cross-validation failed.")

    ranked = _rank_cox_results(cv_results)
    best_row = ranked.iloc[0]

    if not use_one_se_rule:
        return best_row, best_row

    threshold = float(best_row["mean_c_index"]) - float(best_row["sem_c_index"])
    eligible = ranked[ranked["mean_c_index"] >= threshold].copy()
    if eligible.empty:
        return best_row, best_row

    eligible = eligible.sort_values(
        ["mean_selected_features", "penalizer", "mean_c_index"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return best_row, eligible.iloc[0]


def _evaluate_cox_candidates(
    train_df: pd.DataFrame,
    candidate_list: list[float],
    *,
    duration_col: str,
    event_col: str,
    repeated_splits: list[list[tuple[np.ndarray, np.ndarray]]],
    stage: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from lifelines.exceptions import ConvergenceWarning
    from lifelines.utils import concordance_index
    from .models import CoxModel

    aggregate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for penalizer in candidate_list:
        fold_scores: list[float] = []
        fold_selected_feature_counts: list[int] = []
        selected_feature_hits: Counter[str] = Counter()
        failed_folds = 0

        for repeat_idx, splits in enumerate(repeated_splits):
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_train = train_df.iloc[train_idx]
                fold_val = train_df.iloc[val_idx]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", ConvergenceWarning)
                        candidate = CoxModel(penalizer=penalizer).fit(fold_train, duration_col, event_col)
                    risk_scores = candidate.predict_risk_score(fold_val)
                    score = concordance_index(
                        fold_val[duration_col],
                        -risk_scores.values,
                        fold_val[event_col],
                    )
                    selected_features = candidate.fitter.params_[candidate.fitter.params_.abs() > 1e-4].index.tolist()
                    fold_scores.append(float(score))
                    fold_selected_feature_counts.append(len(selected_features))
                    selected_feature_hits.update(selected_features)
                    fold_rows.append(
                        {
                            "stage": stage,
                            "penalizer": penalizer,
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "c_index": float(score),
                            "status": "ok",
                            "n_selected_features": len(selected_features),
                            "selected_features": ";".join(selected_features),
                        }
                    )
                except Exception:
                    failed_folds += 1
                    fold_rows.append(
                        {
                            "stage": stage,
                            "penalizer": penalizer,
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "c_index": np.nan,
                            "status": "failed",
                            "n_selected_features": np.nan,
                            "selected_features": "",
                        }
                    )

        valid_scores = np.asarray(fold_scores, dtype=float)
        n_scores = int(valid_scores.size)
        mean_score = float(valid_scores.mean()) if n_scores else float("-inf")
        std_score = float(valid_scores.std(ddof=1)) if n_scores > 1 else 0.0
        sem_score = float(std_score / np.sqrt(n_scores)) if n_scores else float("inf")
        mean_selected = float(np.mean(fold_selected_feature_counts)) if fold_selected_feature_counts else float("nan")
        aggregate_rows.append(
            {
                "stage": stage,
                "penalizer": penalizer,
                "mean_c_index": mean_score,
                "std_c_index": std_score,
                "sem_c_index": sem_score,
                "n_scores": n_scores,
                "failed_folds": failed_folds,
                "mean_selected_features": mean_selected,
                "unique_selected_features": len(selected_feature_hits),
                "total_selected_feature_hits": int(sum(selected_feature_hits.values())),
            }
        )

    agg = _rank_cox_results(pd.DataFrame(aggregate_rows))
    folds = pd.DataFrame(fold_rows)
    return agg, folds


def train_baseline_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    duration_col: str = "duration",
    event_col: str = "event",
    cox_penalizers: list[float] | None = None,
    lasso_penalizers: list[float] | None = None,
    deep_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train Cox, LASSO-Cox, and optionally DeepSurv models."""
    from .models import CoxModel, DeepSurvModel, LassoCoxModel

    results: dict[str, Any] = {}

    selected_penalizer, cox_search, cox_folds, cox_feature_frequency, cox_selection_summary = _select_cox_penalizer_by_cv(
        train_df,
        cox_penalizers,
        duration_col=duration_col,
        event_col=event_col,
    )
    cox_model = CoxModel(penalizer=selected_penalizer).fit(train_df, duration_col, event_col)
    results["cox"] = {
        "model": cox_model,
        "train_c_index": cox_model.score(train_df),
        "test_c_index": cox_model.score(test_df),
        "selected_penalizer": selected_penalizer,
        "search": cox_search,
        "search_folds": cox_folds,
        "feature_frequency": cox_feature_frequency,
        "selection_summary": cox_selection_summary,
    }

    lasso_model = LassoCoxModel().fit_cv(
        train_df,
        duration_col,
        event_col,
        penalizers=lasso_penalizers,
    )
    results["lasso"] = {
        "model": lasso_model,
        "train_c_index": lasso_model.score(train_df),
        "test_c_index": lasso_model.score(test_df),
        "selected_features": lasso_model.selected_features,
        "cv_results": getattr(lasso_model, "cv_results_", pd.DataFrame()),
        "cv_fold_results": getattr(lasso_model, "cv_fold_results_", pd.DataFrame()),
        "feature_frequency": getattr(lasso_model, "feature_frequency_table_", pd.DataFrame()),
        "selection_summary": getattr(lasso_model, "selection_summary_", {}),
    }

    if DeepSurvModel is not None:
        kwargs = deep_kwargs or {
            "hidden_dim1": 64,
            "hidden_dim2": 32,
            "dropout": 0.2,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 100,
            "validation_split": 0.2,
            "random_state": 42,
        }
        deep_model = DeepSurvModel(**kwargs).fit(train_df, duration_col, event_col)
        results["deep"] = {
            "model": deep_model,
            "train_c_index": deep_model.score(train_df),
            "test_c_index": deep_model.score(test_df),
        }

    return results


def summarize_model_results(
    results: dict[str, Any],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Write a compact summary table for the trained models."""
    paths = WorkflowPaths(Path(output_dir)).ensure()
    rows: list[dict[str, Any]] = []

    for name, item in results.items():
        rows.append(
            {
                "model": name,
                "train_c_index": item.get("train_c_index"),
                "test_c_index": item.get("test_c_index"),
                "selected_penalizer": item.get("selected_penalizer"),
                "selected_features": len(item.get("selected_features", []))
                if isinstance(item.get("selected_features"), list)
                else None,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(paths.final_dir / "model_summary.csv", index=False)
    return summary

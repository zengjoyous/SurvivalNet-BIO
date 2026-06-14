"""LASSO-regularized Cox proportional hazards model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.utils import concordance_index

from ..exceptions import DataValidationError, ModelNotFittedError
from ..io import prepare_feature_matrix


@dataclass
class LassoCoxModel:
    """
    LASSO-Cox proportional hazards model.

    Parameters
    ----------
    penalizer : float
        L1 penalty coefficient (lambda).
    cv : int
        Number of folds for cross-validation.
    random_state : int
        Random seed for reproducibility.
    """

    penalizer: float = 0.1

    cv: int = 5

    random_state: int = 42

    fitter: CoxPHFitter | None = None

    duration_col: str | None = None

    event_col: str | None = None

    feature_cols: list[str] | None = None

    @staticmethod
    def _default_coarse_candidates() -> list[float]:
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

    @staticmethod
    def _make_folds(
        n_samples: int,
        cv: int,
        random_state: int = 42,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        if cv < 2:
            raise ValueError(
                "cv must be at least 2."
            )

        if n_samples < cv:
            raise ValueError(
                "cv cannot exceed sample size."
            )

        rng = np.random.default_rng(
            random_state
        )

        indices = np.arange(
            n_samples
        )

        rng.shuffle(
            indices
        )

        folds = np.array_split(
            indices,
            cv,
        )

        splits = []

        for i in range(cv):

            test_idx = folds[i]

            train_idx = np.concatenate(
                [
                    folds[j]
                    for j in range(cv)
                    if j != i
                ]
            )

            splits.append(
                (
                    train_idx,
                    test_idx,
                )
            )

        return splits

    def _fit_single_penalizer(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        penalizer: float,
    ) -> CoxPHFitter:

        fitter = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=1.0,
        )

        fitter.fit(
            data,
            duration_col=duration_col,
            event_col=event_col,
        )

        return fitter

    @staticmethod
    def _iter_stratified_folds(
        data: pd.DataFrame,
        cv: int,
        random_state: int,
        stratify_col: str = "event",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if cv < 2:
            raise ValueError("cv must be at least 2.")
        if len(data) < cv:
            raise ValueError("cv cannot exceed sample size.")

        rng = np.random.default_rng(random_state)
        indices = np.arange(len(data))
        labels = data[stratify_col].fillna("__nan__").astype(str).to_numpy()

        fold_bins: list[list[int]] = [[] for _ in range(cv)]
        for label in pd.unique(labels):
            label_idx = indices[labels == label]
            rng.shuffle(label_idx)
            for fold, chunk in enumerate(np.array_split(label_idx, cv)):
                fold_bins[fold].extend(chunk.tolist())

        all_indices = np.arange(len(data))
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for fold in range(cv):
            test_idx = np.array(sorted(fold_bins[fold]), dtype=int)
            train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
            splits.append((train_idx, test_idx))
        return splits

    @staticmethod
    def _build_refinement_grid(anchor: float) -> list[float]:
        if anchor <= 0:
            return []

        factors = np.geomspace(0.3, 3.0, num=9)
        grid = {float(anchor * factor) for factor in factors}
        grid = {value for value in grid if value > 0}
        return sorted(round(value, 6) for value in grid)

    @staticmethod
    def _rank_cv_results(results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results.copy()

        ranked = results.copy()
        if "mean_selected_features" not in ranked.columns:
            ranked["mean_selected_features"] = np.nan
        return ranked.sort_values(
            [
                "mean_c_index",
                "mean_selected_features",
                "penalizer",
            ],
            ascending=[False, True, False],
        ).reset_index(drop=True)

    @staticmethod
    def _choose_row(
        cv_results: pd.DataFrame,
        *,
        use_one_se_rule: bool,
    ) -> tuple[pd.Series, pd.Series]:
        if cv_results.empty:
            raise DataValidationError("Cross-validation failed.")

        ranked = LassoCoxModel._rank_cv_results(cv_results)
        best_row = ranked.iloc[0]

        if not use_one_se_rule:
            return best_row, best_row

        threshold = float(best_row["mean_c_index"]) - float(best_row["sem_c_index"])
        eligible = ranked[ranked["mean_c_index"] >= threshold].copy()
        if eligible.empty:
            return best_row, best_row

        eligible = eligible.sort_values(
            [
                "mean_selected_features",
                "penalizer",
                "mean_c_index",
            ],
            ascending=[True, False, False],
        ).reset_index(drop=True)
        return best_row, eligible.iloc[0]

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "LassoCoxModel":

        model_data, feature_cols = prepare_feature_matrix(data, duration_col, event_col)

        self.feature_cols = (
            feature_cols
        )

        self.fitter = (
            self._fit_single_penalizer(
                model_data,
                duration_col,
                event_col,
                self.penalizer,
            )
        )

        self.duration_col = (
            duration_col
        )

        self.event_col = (
            event_col
        )

        return self

    def fit_cv(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        penalizers: list[float] | None = None,
        *,
        n_repeats: int = 3,
        use_one_se_rule: bool = True,
        refinement_grid: list[float] | None = None,
    ) -> "LassoCoxModel":

        model_data, feature_cols = prepare_feature_matrix(data, duration_col, event_col)

        self.feature_cols = (
            feature_cols
        )

        coarse_candidates = penalizers or [
            *self._default_coarse_candidates(),
        ]
        coarse_candidates = sorted({float(p) for p in coarse_candidates if float(p) > 0})

        repeated_splits = [
            self._iter_stratified_folds(
                model_data,
                cv=self.cv,
                random_state=self.random_state + repeat,
                stratify_col=event_col,
            )
            for repeat in range(n_repeats)
        ]

        def evaluate_candidates(candidate_list: list[float], stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
            aggregate_rows: list[dict[str, object]] = []
            fold_rows: list[dict[str, object]] = []

            for penalizer in candidate_list:
                fold_scores: list[float] = []
                fold_selected_feature_counts: list[int] = []
                selected_feature_hits: Counter[str] = Counter()
                failed_folds = 0

                for repeat_idx, splits in enumerate(repeated_splits):
                    for fold_idx, (train_idx, test_idx) in enumerate(splits):
                        train_fold = model_data.iloc[train_idx]
                        test_fold = model_data.iloc[test_idx]

                        try:
                            fitter = self._fit_single_penalizer(
                                train_fold,
                                duration_col,
                                event_col,
                                penalizer,
                            )
                            risk_scores = fitter.predict_partial_hazard(test_fold)
                            score = concordance_index(
                                test_fold[duration_col],
                                -risk_scores.values,
                                test_fold[event_col],
                            )
                            selected_features = fitter.params_[fitter.params_.abs() > 1e-4].index.tolist()
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

            agg = pd.DataFrame(aggregate_rows)
            agg = self._rank_cv_results(agg)
            folds = pd.DataFrame(fold_rows)
            return agg, folds

        coarse_results, coarse_fold_results = evaluate_candidates(coarse_candidates, "coarse")

        if coarse_results.empty:
            raise DataValidationError("Cross-validation failed.")

        anchor = float(coarse_results.iloc[0]["penalizer"])
        fine_candidates = refinement_grid if refinement_grid is not None else self._build_refinement_grid(anchor)
        fine_candidates = sorted({float(p) for p in fine_candidates if float(p) > 0 and float(p) not in set(coarse_candidates)})
        if fine_candidates:
            fine_results, fine_fold_results = evaluate_candidates(fine_candidates, "fine")
            cv_results = pd.concat([coarse_results, fine_results], ignore_index=True)
            fold_results = pd.concat([coarse_fold_results, fine_fold_results], ignore_index=True)
        else:
            cv_results = coarse_results.copy()
            fold_results = coarse_fold_results.copy()

        cv_results = cv_results.drop_duplicates(subset=["penalizer"], keep="first").reset_index(drop=True)
        cv_results = self._rank_cv_results(cv_results)

        best_row, selected_row = self._choose_row(
            cv_results,
            use_one_se_rule=use_one_se_rule,
        )

        selected_penalizer = float(selected_row["penalizer"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            final_fit = self._fit_single_penalizer(
                model_data,
                duration_col,
                event_col,
                selected_penalizer,
            )

        self.penalizer = selected_penalizer
        self.fitter = final_fit
        self.duration_col = duration_col
        self.event_col = event_col
        self.cv_results_ = cv_results
        self.cv_fold_results_ = fold_results
        self.selection_summary_ = {
            "selected_penalizer": selected_penalizer,
            "selected_mean_c_index": float(selected_row["mean_c_index"]),
            "selected_sem_c_index": float(selected_row["sem_c_index"]),
            "selected_mean_selected_features": float(selected_row.get("mean_selected_features", np.nan)),
            "selected_rule": "one_se" if use_one_se_rule else "best_mean",
            "best_penalizer_by_mean": float(best_row["penalizer"]),
            "best_mean_c_index": float(best_row["mean_c_index"]),
            "best_mean_selected_features": float(best_row.get("mean_selected_features", np.nan)),
        }
        final_selected = final_fit.params_[final_fit.params_.abs() > 1e-4].index.tolist()
        self.feature_frequency_ = (
            fold_results[
                (fold_results["penalizer"] == selected_penalizer)
                & (fold_results["status"] == "ok")
            ]
            .assign(
                selected_features=lambda frame: frame["selected_features"].fillna("")
            )
        )
        freq_counter: Counter[str] = Counter()
        successful_rows = self.feature_frequency_["selected_features"].tolist()
        for feature_string in successful_rows:
            if not feature_string:
                continue
            freq_counter.update([feature for feature in feature_string.split(";") if feature])
        if freq_counter:
            total_success = int(self.feature_frequency_.shape[0])
            feature_rows = []
            for feature, count in freq_counter.most_common():
                feature_rows.append(
                    {
                        "feature": feature,
                        "frequency": count,
                        "selection_rate": count / total_success if total_success else np.nan,
                    }
                )
            self.feature_frequency_table_ = pd.DataFrame(feature_rows)
        else:
            self.feature_frequency_table_ = pd.DataFrame(columns=["feature", "frequency", "selection_rate"])
        self.selected_features_ = final_selected

        return self

    @property
    def summary(self) -> pd.DataFrame:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        return self.fitter.summary

    @property
    def hazard_ratios(self) -> pd.DataFrame:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        summary = self.fitter.summary

        return summary[
            [
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        ].rename(
            columns={
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "CI_lower",
                "exp(coef) upper 95%": "CI_upper",
                "p": "p_value",
            }
        )

    @property
    def selected_features(
        self,
    ) -> list[str]:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        params = self.fitter.params_

        return params[
            params.abs() > 1e-4
        ].index.tolist()

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        if self.feature_cols is None:

            raise ModelNotFittedError(
                "Feature columns unavailable."
            )

        X = data[self.feature_cols].copy()
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))

        return self.fitter.predict_partial_hazard(X).rename("lasso_risk_score")

    def score(
        self,
        data: pd.DataFrame,
    ) -> float:

        if self.duration_col is None:

            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        if self.event_col is None:

            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        scoring_data = data.copy()
        scoring_data = scoring_data.dropna(subset=[self.duration_col, self.event_col]).reset_index(drop=True)
        risk_scores = self.predict_risk_score(scoring_data)

        return float(
            concordance_index(
                scoring_data[self.duration_col],
                -risk_scores,
                scoring_data[self.event_col],
            )
        )

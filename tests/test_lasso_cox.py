import math

import pandas as pd

from survivalnet.models import LassoCoxModel


def test_default_coarse_candidates_match_log_grid():
    assert LassoCoxModel._default_coarse_candidates() == [
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


def test_refinement_grid_spans_nearby_log_range():
    grid = LassoCoxModel._build_refinement_grid(0.1)

    assert grid[0] == 0.03
    assert grid[-1] == 0.3
    assert all(left < right for left, right in zip(grid, grid[1:]))
    assert len(grid) >= 7


def test_one_se_rule_prefers_sparser_model_within_threshold():
    results = pd.DataFrame(
        [
            {
                "penalizer": 0.03,
                "mean_c_index": 0.80,
                "std_c_index": 0.03,
                "sem_c_index": 0.02,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 8.0,
                "unique_selected_features": 8,
                "total_selected_feature_hits": 120,
            },
            {
                "penalizer": 0.1,
                "mean_c_index": 0.79,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 3.0,
                "unique_selected_features": 3,
                "total_selected_feature_hits": 45,
            },
            {
                "penalizer": 0.3,
                "mean_c_index": 0.74,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 1.0,
                "unique_selected_features": 1,
                "total_selected_feature_hits": 15,
            },
        ]
    )

    best_row, selected_row = LassoCoxModel._choose_row(results, use_one_se_rule=True)

    assert math.isclose(float(best_row["penalizer"]), 0.03)
    assert math.isclose(float(selected_row["penalizer"]), 0.1)
    assert float(selected_row["mean_selected_features"]) < float(best_row["mean_selected_features"])

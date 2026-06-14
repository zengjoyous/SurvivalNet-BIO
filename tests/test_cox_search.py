import math

import pandas as pd

from survivalnet.workflow import _build_cox_refinement_grid, _choose_cox_row, _default_cox_candidates


def test_default_cox_candidates_match_log_grid():
    assert _default_cox_candidates() == [
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


def test_cox_refinement_grid_spans_nearby_log_range():
    grid = _build_cox_refinement_grid(0.1)

    assert grid[0] == 0.03
    assert grid[-1] == 0.3
    assert all(left < right for left, right in zip(grid, grid[1:]))


def test_one_se_prefers_more_conservative_penalizer():
    results = pd.DataFrame(
        [
            {
                "penalizer": 0.03,
                "mean_c_index": 0.81,
                "std_c_index": 0.03,
                "sem_c_index": 0.02,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 12.0,
                "unique_selected_features": 12,
                "total_selected_feature_hits": 180,
            },
            {
                "penalizer": 0.1,
                "mean_c_index": 0.80,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 6.0,
                "unique_selected_features": 6,
                "total_selected_feature_hits": 90,
            },
            {
                "penalizer": 0.3,
                "mean_c_index": 0.79,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 15,
                "failed_folds": 0,
                "mean_selected_features": 2.0,
                "unique_selected_features": 2,
                "total_selected_feature_hits": 30,
            },
        ]
    )

    best_row, selected_row = _choose_cox_row(results, use_one_se_rule=True)

    assert math.isclose(float(best_row["penalizer"]), 0.03)
    assert math.isclose(float(selected_row["penalizer"]), 0.1)
    assert float(selected_row["mean_selected_features"]) < float(best_row["mean_selected_features"])

import math

import pandas as pd

from scripts.train_deepsurv import (
    choose_row,
    network_complexity,
    parse_float_grid,
    parse_int_grid,
)


def test_parse_grids_deduplicate_and_sort():
    assert parse_int_grid("64,32,64,128") == [32, 64, 128]
    assert parse_float_grid("0.2,0.1,0.2") == [0.1, 0.2]


def test_network_complexity_grows_with_width():
    small = network_complexity(32, 16, 100, batch_norm=False)
    large = network_complexity(128, 64, 100, batch_norm=False)

    assert large > small


def test_one_se_rule_prefers_simpler_deepsurv():
    results = pd.DataFrame(
        [
            {
                "hidden_dim1": 128,
                "hidden_dim2": 64,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 32,
                "batch_norm": True,
                "mean_c_index": 0.81,
                "std_c_index": 0.03,
                "sem_c_index": 0.02,
                "n_scores": 12,
                "failed_folds": 0,
                "n_parameters": 10000,
            },
            {
                "hidden_dim1": 64,
                "hidden_dim2": 32,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "batch_norm": True,
                "mean_c_index": 0.80,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 12,
                "failed_folds": 0,
                "n_parameters": 3000,
            },
            {
                "hidden_dim1": 32,
                "hidden_dim2": 16,
                "dropout": 0.3,
                "learning_rate": 0.0003,
                "batch_size": 64,
                "batch_norm": True,
                "mean_c_index": 0.79,
                "std_c_index": 0.02,
                "sem_c_index": 0.01,
                "n_scores": 12,
                "failed_folds": 0,
                "n_parameters": 900,
            },
        ]
    )

    best_row, selected_row = choose_row(results, use_one_se_rule=True)

    assert math.isclose(float(best_row["mean_c_index"]), 0.81)
    assert math.isclose(float(selected_row["n_parameters"]), 3000)
    assert float(selected_row["n_parameters"]) < float(best_row["n_parameters"])

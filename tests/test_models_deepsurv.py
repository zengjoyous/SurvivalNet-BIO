from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


pytest.importorskip("pycox")
pytest.importorskip("torchtuples")


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "scripts" / "_bootstrap.py"
SCRIPT_PATH = ROOT / "scripts" / "train_deepsurv.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.modules.setdefault("_bootstrap", _load_module("_bootstrap", BOOTSTRAP_PATH))
train_deepsurv = _load_module("train_deepsurv_for_tests", SCRIPT_PATH)


def test_parse_grids_deduplicate_and_sort():
    assert train_deepsurv.parse_int_grid("64,32,64,128") == [32, 64, 128]
    assert train_deepsurv.parse_float_grid("0.2,0.1,0.2") == [0.1, 0.2]


def test_network_complexity_grows_with_width():
    small = train_deepsurv.network_complexity(32, 16, 100, batch_norm=False)
    large = train_deepsurv.network_complexity(128, 64, 100, batch_norm=False)

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

    best_row, selected_row = train_deepsurv.choose_row(results, use_one_se_rule=True)

    assert pytest.approx(float(best_row["mean_c_index"])) == 0.81
    assert pytest.approx(float(selected_row["n_parameters"])) == 900
    assert float(selected_row["n_parameters"]) < float(best_row["n_parameters"])

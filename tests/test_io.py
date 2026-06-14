from pathlib import Path

import pandas as pd

from survivalnet import (
    load_table,
    normalize_survival_data,
    prepare_feature_matrix,
    prepare_survival_dataset,
    split_train_test,
)


def test_load_table(tmp_path: Path):
    path = tmp_path / "clinical.tsv"
    path.write_text("# comment\nsample\tOS.time\tOS\nA\t1\t1\n")

    df = load_table(path)

    assert list(df.columns) == ["sample", "OS.time", "OS"]
    assert len(df) == 1


def test_normalize_survival_data():
    df = pd.DataFrame(
        {
            "sample": ["A", "B"],
            "time": [10, 20],
            "status": ["Dead", "Alive"],
            "gene1": [1.2, 3.4],
        }
    )

    out = normalize_survival_data(df, "time", "status", id_col="sample")

    assert list(out.columns[:3]) == ["sample", "duration", "event"]
    assert out["event"].tolist() == [1, 0]


def test_prepare_survival_dataset_primary_tumor_only():
    clinical = pd.DataFrame(
        {
            "_PATIENT": ["TCGA-A", "TCGA-B"],
            "OS.time": [10, 20],
            "OS": [1, 0],
        }
    )
    expr = pd.DataFrame(
        {
            "sample": ["TCGA-A-01A", "TCGA-A-11A", "TCGA-B-01A"],
            "_PATIENT": ["TCGA-A", "TCGA-A", "TCGA-B"],
            "gene1": [1.0, 2.0, 3.0],
        }
    )

    merged = prepare_survival_dataset(clinical, expr)

    assert len(merged) == 2
    assert merged["_PATIENT"].tolist() == ["TCGA-A", "TCGA-B"]
    assert merged["gene1"].tolist() == [1.0, 3.0]


def test_split_train_test():
    df = pd.DataFrame(
        {
            "_PATIENT": [f"P{i}" for i in range(20)],
            "event": [0] * 10 + [1] * 10,
            "x1": range(20),
        }
    )

    train, test = split_train_test(df, test_size=0.3, random_state=0)

    assert len(train) + len(test) == len(df)
    assert len(train) > len(test) > 0


def test_prepare_feature_matrix_applies_log_transform_and_filters():
    df = pd.DataFrame(
        {
            "duration": [10, 11, 12, 13],
            "event": [1, 0, 1, 0],
            "gene_low": [0, 0, 0, 1],
            "gene_high": [1, 2, 4, 8],
            "gene_constant": [5, 5, 5, 5],
        }
    )

    model_data, features = prepare_feature_matrix(
        df,
        "duration",
        "event",
        max_features=None,
    )

    assert "gene_constant" not in features
    assert "gene_low" not in features
    assert "gene_high" in features
    assert model_data["gene_high"].iloc[0] == 1.0

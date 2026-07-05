from pathlib import Path

import pandas as pd

from survivalnet.workflow import infer_output_dir, load_input_tables, save_split_tables


def test_infer_output_dir_uses_parent_directory_name(tmp_path: Path):
    ref = tmp_path / "examples" / "GDC TCGA Stomach Cancer (STAD)" / "TCGA-STAD.clinical.tsv"

    out = infer_output_dir(ref, output_root=tmp_path / "output")

    assert out == tmp_path / "output" / "GDC TCGA Stomach Cancer (STAD)"


def test_load_input_tables_reads_both_tables(tmp_path: Path):
    clinical = tmp_path / "clinical.tsv"
    expression = tmp_path / "expression.tsv"
    clinical.write_text("sample\tOS.time\tOS\nA\t1\t1\n")
    expression.write_text("gene\tsample\nx\tA-01A\n")

    clinical_df, expression_df = load_input_tables(clinical, expression)

    assert clinical_df.shape == (1, 3)
    assert "sample" in expression_df.columns


def test_save_split_tables_writes_process_outputs(tmp_path: Path):
    merged = pd.DataFrame(
        {
            "_PATIENT": [f"P{i}" for i in range(10)],
            "duration": list(range(10, 20)),
            "event": [0, 1] * 5,
            "gene1": [float(i) for i in range(10)],
            "gene2": [float(i + 1) for i in range(10)],
        }
    )

    result = save_split_tables(merged, tmp_path, test_size=0.3, max_features=1, random_state=0)

    assert (tmp_path / "process" / "merged.csv").exists()
    assert (tmp_path / "process" / "train.csv").exists()
    assert (tmp_path / "process" / "test.csv").exists()
    assert "selected_features" in result
    assert len(result["train"]) + len(result["test"]) == len(merged)

from pathlib import Path

from survivalnet.workflow import infer_output_dir


def test_infer_output_dir_uses_parent_directory_name(tmp_path: Path):
    ref = tmp_path / "examples" / "GDC TCGA Stomach Cancer (STAD)" / "TCGA-STAD.clinical.tsv"

    out = infer_output_dir(ref, output_root=tmp_path / "output")

    assert out == tmp_path / "output" / "GDC TCGA Stomach Cancer (STAD)"

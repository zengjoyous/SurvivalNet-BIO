from pathlib import Path

from survivalnet.workflow import infer_output_dir


def test_infer_output_dir_uses_parent_folder_name():
    output_dir = infer_output_dir(
        Path("/tmp/examples/GDC TCGA Stomach Cancer (STAD)/TCGA-STAD.star_tpm.tsv"),
        output_root="/tmp/output",
    )

    assert output_dir == Path("/tmp/output/GDC TCGA Stomach Cancer (STAD)")

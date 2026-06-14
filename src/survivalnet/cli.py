"""Command line interface for survivalnet."""

from pathlib import Path

import click

from .workflow import (
    build_analysis_table,
    infer_output_dir,
    load_input_tables,
    save_split_tables,
    summarize_model_results,
    train_baseline_models,
)


@click.group()
def main() -> None:
    """Top-level survivalnet command."""


@main.command()
def version() -> None:
    """Print package version."""
    from . import __version__

    click.echo(__version__)


@main.command()
@click.option("--clinical", "clinical_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Clinical survival TSV/CSV.")
@click.option("--expression", "expression_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="TCGA expression matrix with genes as rows and samples as columns.")
@click.option("--output", "output_dir", type=click.Path(file_okay=False, path_type=Path), default=None, help="Output directory root. Defaults to output/<clinical folder name>.")
@click.option("--clinical-sep", default=None, help="Delimiter for clinical table.")
@click.option("--test-size", default=0.30, show_default=True, type=float)
def run(
    clinical_path: Path,
    expression_path: Path,
    output_dir: Path | None,
    clinical_sep: str | None,
    test_size: float,
) -> None:
    """Run the standard survival workflow."""
    clinical_df, expression_df = load_input_tables(
        clinical_path,
        expression_path,
        clinical_sep=clinical_sep,
    )
    merged_df = build_analysis_table(clinical_df, expression_df)
    resolved_output_dir = output_dir or infer_output_dir(clinical_path, output_root="output")
    splits = save_split_tables(
        merged_df,
        resolved_output_dir,
        test_size=test_size,
    )
    results = train_baseline_models(splits["train"], splits["test"])
    summary = summarize_model_results(results, resolved_output_dir)

    click.echo(f"Saved outputs to: {resolved_output_dir.resolve()}")
    click.echo(summary.to_string(index=False))


if __name__ == "__main__":
    main()

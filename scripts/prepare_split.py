"""Prepare merged data and split it into train/test."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from survivalnet.workflow import build_analysis_table, infer_output_dir, load_input_tables, save_split_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory root. Defaults to output/<clinical folder name>.",
    )
    parser.add_argument("--clinical-sep", default=None)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-features", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clinical_df, expression_df = load_input_tables(
        args.clinical,
        args.expression,
        clinical_sep=args.clinical_sep,
    )
    merged_df = build_analysis_table(clinical_df, expression_df)
    output_dir = args.output or infer_output_dir(args.clinical, output_root=ROOT / "output")
    splits = save_split_tables(
        merged_df,
        output_dir,
        test_size=args.test_size,
        max_features=args.max_features,
    )
    print(f"saved: {output_dir.resolve()}")
    for name, df in splits.items():
        print(f"{name}: {df.shape}")


if __name__ == "__main__":
    main()

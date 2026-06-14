"""Collect model summaries into a single final table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "stomach_cancer",
        help="Output directory root that contains the final/ folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_dir = args.output / "final"
    frames = []
    for name in ["cox", "lasso", "deepsurv"]:
        path = final_dir / f"{name}_summary.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No summary files found in {final_dir}")
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(final_dir / "model_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

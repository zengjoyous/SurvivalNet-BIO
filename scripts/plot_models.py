"""Plot comparable visualizations for Cox, LASSO-Cox, and DeepSurv outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from survivalnet.core import split_risk_group
from survivalnet.io import load_table
from survivalnet.visualize import (
    plot_feature_importance,
    plot_grouped_km_with_pvalue,
    plot_risk_score_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "stomach_cancer",
        help="Project output root containing the final/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults to final/model_comparison.png.",
    )
    parser.add_argument("--sep", default=None, help="Delimiter for CSV/TSV files.")
    parser.add_argument("--duration-col", default="duration")
    parser.add_argument("--event-col", default="event")
    parser.add_argument("--group-a", default="low")
    parser.add_argument("--group-b", default="high")
    return parser.parse_args()


def load_model_tables(output_dir: Path, sep: str | None = None) -> dict[str, pd.DataFrame]:
    final_dir = output_dir / "final"
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in {
        "cox": "cox_test_scored.csv",
        "lasso": "lasso_test_scored.csv",
        "deep": "deepsurv_test_scored.csv",
    }.items():
        path = final_dir / filename
        if path.exists():
            tables[name] = load_table(path, sep=sep, comment="#")
    return tables


def load_hazard_table(output_dir: Path, name: str, sep: str | None = None) -> pd.DataFrame | None:
    path = output_dir / "final" / f"{name}_hazard_ratios.csv"
    if not path.exists():
        return None
    return load_table(path, sep=sep, comment="#")


def summarize_row(output_dir: Path, name: str, sep: str | None = None) -> str:
    path = output_dir / "final" / f"{name}_summary.csv"
    if not path.exists():
        return ""
    df = load_table(path, sep=sep, comment="#")
    if df.empty:
        return ""
    row = df.iloc[0].to_dict()
    fields = [
        f"train_c_index={row.get('train_c_index')}",
        f"test_c_index={row.get('test_c_index')}",
        f"selected_penalizer={row.get('selected_penalizer')}",
        f"selected_rule={row.get('selected_rule')}",
    ]
    return ", ".join(field for field in fields if field and field.split("=", 1)[1] != "None")


def plot_single_model(
    ax_km,
    ax_dist,
    ax_imp,
    df: pd.DataFrame,
    *,
    model_name: str,
    duration_col: str,
    event_col: str,
    group_a: str,
    group_b: str,
    hazard_df: pd.DataFrame | None,
):
    score_col = "deep_risk_score" if model_name == "deep" else "risk_score"
    if score_col not in df.columns:
        raise KeyError(f"Missing score column for {model_name}: {score_col}")

    df = df.copy()
    df["risk_group"] = split_risk_group(df[score_col])

    plot_grouped_km_with_pvalue(
        df,
        duration_col=duration_col,
        event_col=event_col,
        group_col="risk_group",
        group_a=group_a,
        group_b=group_b,
        ax=ax_km,
    )
    ax_km.set_title(f"{model_name.title()} KM risk stratification", fontweight="bold")

    plot_risk_score_distribution(df, score_col, group_col="risk_group", ax=ax_dist)
    ax_dist.set_title(f"{model_name.title()} risk score distribution")

    if model_name in {"cox", "lasso"} and hazard_df is not None and not hazard_df.empty:
        plot_feature_importance(hazard_df, ax=ax_imp, top_n=12)
        ax_imp.set_title(f"{model_name.title()} top feature effects")
    else:
        ax_imp.axis("off")
        ax_imp.text(
            0.5,
            0.5,
            "No hazard-ratio forest plot\navailable for DeepSurv.",
            ha="center",
            va="center",
            fontsize=12,
        )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    final_dir = output_dir / "final"
    output_path = args.output or (final_dir / "model_comparison.png")

    tables = load_model_tables(output_dir, sep=args.sep)
    if not tables:
        raise FileNotFoundError(f"No scored model tables found in {final_dir}")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(len(tables), 3, hspace=0.35, wspace=0.28)

    order = [name for name in ["cox", "lasso", "deep"] if name in tables]
    for row_idx, name in enumerate(order):
        df = tables[name]
        ax_km = fig.add_subplot(gs[row_idx, 0])
        ax_dist = fig.add_subplot(gs[row_idx, 1])
        ax_imp = fig.add_subplot(gs[row_idx, 2])
        hazard_df = load_hazard_table(output_dir, name, sep=args.sep)
        plot_single_model(
            ax_km,
            ax_dist,
            ax_imp,
            df,
            model_name=name,
            duration_col=args.duration_col,
            event_col=args.event_col,
            group_a=args.group_a,
            group_b=args.group_b,
            hazard_df=hazard_df,
        )
        footer = summarize_row(output_dir, name, sep=args.sep)
        if footer:
            fig.text(0.5, 0.02 + row_idx * 0.01, f"{name}: {footer}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"saved plot: {output_path.resolve()}")


if __name__ == "__main__":
    main()

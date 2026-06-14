"""Plot Kaplan-Meier curves and compact model reports from survival results."""

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
    plot_grouped_km,
    plot_grouped_km_with_pvalue,
    plot_km_curve,
    plot_risk_score_distribution,
)
from survivalnet.core import fit_km


MODEL_FILE_MAP = {
    "cox": "cox_test_scored.csv",
    "lasso": "lasso_test_scored.csv",
    "deep": "deepsurv_test_scored.csv",
}

MODEL_RISK_COL_MAP = {
    "cox": "risk_score",
    "lasso": "risk_score",
    "deep": "deep_risk_score",
}

MODEL_HAZARD_MAP = {
    "cox": "cox_hazard_ratios.csv",
    "lasso": "lasso_hazard_ratios.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV/TSV table with duration/event columns.")
    parser.add_argument("--model", choices=["cox", "lasso", "deep"], default=None, help="Plot a saved model-scored test table.")
    parser.add_argument("--mode", choices=["km", "report"], default="km", help="Single KM plot or a richer report figure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "stomach_cancer",
        help="Project output root used with --model. Defaults to the existing stomach_cancer output tree.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "stomach_cancer" / "final" / "km_curve.png")
    parser.add_argument("--sep", default=None, help="Delimiter for the input table.")
    parser.add_argument("--duration-col", default="duration")
    parser.add_argument("--event-col", default="event")
    parser.add_argument("--group-col", default=None, help="Existing categorical group column to plot.")
    parser.add_argument("--risk-score-col", default=None, help="Numeric risk score column to convert into high/low groups.")
    parser.add_argument("--group-a", default="low")
    parser.add_argument("--group-b", default="high")
    parser.add_argument("--title", default="Kaplan-Meier Curve")
    parser.add_argument("--top-n", type=int, default=12, help="Number of features shown in report mode.")
    return parser.parse_args()


def load_model_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load the scored test table and optional feature-importance table."""
    if args.model is None:
        if args.input is None:
            raise ValueError("Either --input or --model must be provided.")
        return load_table(args.input, sep=args.sep, comment="#"), None

    scored_path = args.output_dir / "final" / MODEL_FILE_MAP[args.model]
    if not scored_path.exists():
        raise FileNotFoundError(f"Missing scored test table: {scored_path}")

    df = load_table(scored_path, sep=args.sep, comment="#")
    if args.risk_score_col is None:
        args.risk_score_col = MODEL_RISK_COL_MAP[args.model]

    hazard_df = None
    hazard_file = MODEL_HAZARD_MAP.get(args.model)
    if hazard_file is not None:
        hazard_path = args.output_dir / "final" / hazard_file
        if hazard_path.exists():
            hazard_df = load_table(hazard_path, sep=args.sep, comment="#")
    return df, hazard_df


def plot_km_panel(
    df: pd.DataFrame,
    args: argparse.Namespace,
    ax,
):
    if args.risk_score_col is not None:
        if args.risk_score_col not in df.columns:
            raise KeyError(f"Missing risk score column: {args.risk_score_col}")
        df = df.copy()
        df["risk_group"] = split_risk_group(df[args.risk_score_col])
        plot_grouped_km_with_pvalue(
            df,
            duration_col=args.duration_col,
            event_col=args.event_col,
            group_col="risk_group",
            group_a=args.group_a,
            group_b=args.group_b,
            ax=ax,
        )
    elif args.group_col is not None:
        if args.group_col not in df.columns:
            raise KeyError(f"Missing group column: {args.group_col}")
        plot_grouped_km(df, duration_col=args.duration_col, event_col=args.event_col, group_col=args.group_col, ax=ax)
    else:
        kmf = fit_km(df, args.duration_col, args.event_col)
        plot_km_curve(kmf, ax=ax)
    ax.set_title(args.title)
    return ax


def plot_report(
    df: pd.DataFrame,
    hazard_df: pd.DataFrame | None,
    args: argparse.Namespace,
):
    if args.risk_score_col is None:
        raise ValueError("Report mode requires a model-scored table.")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], hspace=0.35, wspace=0.25)
    ax_km = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_imp = fig.add_subplot(gs[1, 1])

    # KM panel
    df = df.copy()
    if args.risk_score_col not in df.columns:
        raise KeyError(f"Missing risk score column: {args.risk_score_col}")
    df["risk_group"] = split_risk_group(df[args.risk_score_col])
    plot_grouped_km_with_pvalue(
        df,
        duration_col=args.duration_col,
        event_col=args.event_col,
        group_col="risk_group",
        group_a=args.group_a,
        group_b=args.group_b,
        ax=ax_km,
    )
    ax_km.set_title(f"{args.model.title()} model: risk stratification on test set", fontsize=14, fontweight="bold")

    # Distribution panel
    plot_risk_score_distribution(df, args.risk_score_col, group_col="risk_group", ax=ax_dist)
    ax_dist.set_title("Risk score distribution by group")

    # Importance panel
    if hazard_df is not None and not hazard_df.empty and args.model in {"cox", "lasso"}:
        plot_feature_importance(hazard_df, ax=ax_imp, top_n=args.top_n)
        ax_imp.set_title(f"Top {min(args.top_n, len(hazard_df))} feature effects")
    else:
        ax_imp.axis("off")
        ax_imp.text(
            0.5,
            0.55,
            "No gene-level importance available\nfor this model.",
            ha="center",
            va="center",
            fontsize=12,
        )

    # Small metadata footer if summary exists.
    summary_path = args.output_dir / "final" / f"{args.model}_summary.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            row = summary.iloc[0].to_dict()
            footer = ", ".join(
                f"{key}={value}"
                for key, value in row.items()
                if key in {"train_c_index", "test_c_index", "selected_penalizer", "n_features"}
            )
            if footer:
                fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def main() -> None:
    args = parse_args()
    if args.output_dir is None and args.input is not None:
        args.output_dir = args.input.parent
    df, hazard_df = load_model_inputs(args)

    if args.mode == "report":
        if args.model is None:
            raise ValueError("Report mode requires --model.")
        fig = plot_report(df, hazard_df, args)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_km_panel(df, args, ax)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    print(f"saved plot: {args.output.resolve()}")


if __name__ == "__main__":
    main()

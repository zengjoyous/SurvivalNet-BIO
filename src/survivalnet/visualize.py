"""Plotting helpers for survival analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import fit_km, logrank_p_value


def plot_km_curve(km_fitter, label: str | None = None, ax=None):
    """Plot a Kaplan-Meier curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    km_fitter.plot_survival_function(ax=ax, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    return ax


def plot_grouped_km(data: pd.DataFrame, duration_col: str, event_col: str, group_col: str, ax=None):
    """Plot Kaplan-Meier curves for each group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for group_name, subset in data.groupby(group_col):
        kmf = fit_km(subset, duration_col=duration_col, event_col=event_col)
        kmf.plot_survival_function(ax=ax, label=str(group_name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    return ax


def plot_grouped_km_with_pvalue(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    group_a,
    group_b,
    ax=None,
):
    """Plot grouped KM curves and annotate the log-rank p-value."""
    ax = plot_grouped_km(data, duration_col=duration_col, event_col=event_col, group_col=group_col, ax=ax)
    p_value = logrank_p_value(data, group_col, duration_col, event_col, group_a, group_b)
    ax.text(0.98, 0.05, f"p = {p_value:.3g}", transform=ax.transAxes, ha="right", va="bottom")
    return ax


def plot_risk_score_distribution(
    data: pd.DataFrame,
    score_col: str,
    group_col: str = "risk_group",
    ax=None,
):
    """Plot the distribution of risk scores by group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if score_col not in data.columns:
        raise KeyError(f"Missing score column: {score_col}")
    if group_col not in data.columns:
        raise KeyError(f"Missing group column: {group_col}")

    order = [group for group in ["low", "high"] if group in set(data[group_col].astype(str))]
    if not order:
        order = sorted(data[group_col].dropna().astype(str).unique().tolist())

    grouped = [data.loc[data[group_col].astype(str) == group, score_col].dropna().astype(float).to_numpy() for group in order]
    box = ax.boxplot(grouped, labels=order, patch_artist=True, widths=0.55, showfliers=False)
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518"]
    for patch, color in zip(box["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    ax.set_ylabel(score_col)
    ax.set_title("Risk score distribution")
    ax.grid(True, axis="y", alpha=0.25)
    return ax


def plot_feature_importance(
    hazard_ratios: pd.DataFrame,
    ax=None,
    top_n: int = 12,
    sort_by: str = "p_value",
):
    """Plot a forest-style importance chart from hazard ratios."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if hazard_ratios is None or hazard_ratios.empty:
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
        return ax

    required = {"HR", "CI_lower", "CI_upper"}
    missing = required - set(hazard_ratios.columns)
    if missing:
        raise KeyError(f"Missing hazard ratio columns: {sorted(missing)}")

    df = hazard_ratios.copy()
    df = df.reset_index().rename(columns={"index": "feature"})
    if "feature" not in df.columns:
        df["feature"] = hazard_ratios.index.astype(str)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True)
    else:
        df = df.assign(_distance=np.abs(np.log(df["HR"].astype(float).clip(lower=1e-8))))
        df = df.sort_values("_distance", ascending=False)

    df = df.head(top_n).copy()
    df = df.iloc[::-1]

    x = df["HR"].astype(float).to_numpy()
    lower = df["CI_lower"].astype(float).to_numpy()
    upper = df["CI_upper"].astype(float).to_numpy()
    y = np.arange(len(df))
    xerr = np.vstack([x - lower, upper - x])

    ax.errorbar(x, y, xerr=xerr, fmt="o", color="#4C78A8", ecolor="gray", capsize=3, lw=1.2)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].astype(str))
    ax.set_xlabel("Hazard ratio (log scale)")
    ax.set_title(f"Top {len(df)} feature effects")
    ax.grid(True, axis="x", alpha=0.25)
    return ax

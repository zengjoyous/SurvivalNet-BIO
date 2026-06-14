"""Train a Cox model from the saved split tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from survivalnet.core import split_risk_group
from survivalnet.models import CoxModel
from survivalnet.workflow import _select_cox_penalizer_by_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output" / "stomach_cancer" / "process")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory root. Defaults to the parent of --input.",
    )
    parser.add_argument(
        "--penalizers",
        default="0.01,0.05,0.1,0.5,1.0",
        help="Comma-separated Cox penalizer candidates evaluated by internal CV on train.",
    )
    return parser.parse_args()


def parse_penalizers(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.input / "train.csv")
    test_df = pd.read_csv(args.input / "test.csv")
    output_dir = args.output or args.input.parent

    candidates = parse_penalizers(args.penalizers)
    (
        best_penalizer,
        search_results,
        search_fold_results,
        feature_frequency_table,
        selection_summary,
    ) = _select_cox_penalizer_by_cv(
        train_df,
        candidates,
        duration_col="duration",
        event_col="event",
    )
    model = CoxModel(penalizer=best_penalizer).fit(train_df, "duration", "event")

    summary = pd.DataFrame(
        [
            {
                "model": "cox",
                "selected_penalizer": best_penalizer,
                "selected_rule": selection_summary.get("selected_rule"),
                "train_c_index": model.score(train_df),
                "test_c_index": model.score(test_df),
                "n_features": len(model.feature_cols or []),
                "selected_cv_c_index": selection_summary.get("selected_mean_c_index"),
                "selected_cv_sem_c_index": selection_summary.get("selected_sem_c_index"),
                "selected_cv_mean_selected_features": selection_summary.get("selected_mean_selected_features"),
                "best_cv_c_index": selection_summary.get("best_mean_c_index"),
                "best_cv_mean_selected_features": selection_summary.get("best_mean_selected_features"),
            }
        ]
    )

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(final_dir / "cox_summary.csv", index=False)
    model.hazard_ratios.to_csv(final_dir / "cox_hazard_ratios.csv")
    search_results.to_csv(final_dir / "cox_penalizer_search.csv", index=False)
    search_fold_results.to_csv(final_dir / "cox_penalizer_search_folds.csv", index=False)
    feature_frequency_table.to_csv(final_dir / "cox_feature_frequency.csv", index=False)

    scored_test = test_df.copy()
    scored_test["risk_score"] = model.predict_risk_score(test_df).values
    scored_test["risk_group"] = split_risk_group(scored_test["risk_score"])
    scored_test.to_csv(final_dir / "cox_test_scored.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

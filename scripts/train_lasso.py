"""Train a LASSO-Cox model from the saved split tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from survivalnet.core import split_risk_group
from survivalnet.models import LassoCoxModel


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
        default="0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.3,1",
        help="Comma-separated coarse LASSO penalizer candidates for stage-1 CV.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-one-se", action="store_true", help="Disable the one-standard-error rule.")
    return parser.parse_args()


def parse_penalizers(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.input / "train.csv")
    test_df = pd.read_csv(args.input / "test.csv")
    output_dir = args.output or args.input.parent

    candidates = parse_penalizers(args.penalizers)
    model = LassoCoxModel(cv=args.folds, random_state=args.random_state).fit_cv(
        train_df,
        "duration",
        "event",
        penalizers=candidates,
        n_repeats=args.repeats,
        use_one_se_rule=not args.no_one_se,
    )

    selection_summary = getattr(model, "selection_summary_", {})
    summary = pd.DataFrame(
        [
            {
                "model": "lasso",
                "selected_penalizer": model.penalizer,
                "selected_rule": selection_summary.get("selected_rule"),
                "train_c_index": model.score(train_df),
                "test_c_index": model.score(test_df),
                "n_features": len(model.feature_cols or []),
                "selected_features": len(model.selected_features),
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
    summary.to_csv(final_dir / "lasso_summary.csv", index=False)
    model.hazard_ratios.to_csv(final_dir / "lasso_hazard_ratios.csv")
    pd.DataFrame(model.selected_features, columns=["feature"]).to_csv(final_dir / "lasso_selected_features.csv", index=False)
    getattr(model, "cv_results_", pd.DataFrame()).to_csv(final_dir / "lasso_penalizer_search.csv", index=False)
    getattr(model, "cv_fold_results_", pd.DataFrame()).to_csv(final_dir / "lasso_penalizer_search_folds.csv", index=False)
    getattr(model, "feature_frequency_table_", pd.DataFrame()).to_csv(final_dir / "lasso_feature_frequency.csv", index=False)

    scored_test = test_df.copy()
    scored_test["risk_score"] = model.predict_risk_score(test_df).values
    scored_test["risk_group"] = split_risk_group(scored_test["risk_score"])
    scored_test.to_csv(final_dir / "lasso_test_scored.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

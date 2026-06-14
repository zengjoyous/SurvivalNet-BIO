"""Train a DeepSurv model from the saved split tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from survivalnet.core import split_risk_group
from survivalnet.models import DeepSurvModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output" / "stomach_cancer" / "process")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory root. Defaults to the parent of --input.",
    )
    parser.add_argument("--hidden-dim1-grid", default="16,32,64", help="Comma-separated hidden_dim1 candidates.")
    parser.add_argument("--hidden-dim2-grid", default="8,16,32", help="Comma-separated hidden_dim2 candidates.")
    parser.add_argument("--dropout-grid", default="0.0,0.1,0.2", help="Comma-separated dropout candidates.")
    parser.add_argument("--learning-rate-grid", default="0.0001,0.0003,0.001", help="Comma-separated learning rate candidates.")
    parser.add_argument("--batch-size-grid", default="16,32", help="Comma-separated batch size candidates.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-batch-norm", action="store_true", help="Disable batch normalization in the network.")
    parser.add_argument(
        "--log-transform",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable log2(x + 1) preprocessing for DeepSurv.",
    )
    parser.add_argument("--max-features", type=int, default=120, help="Maximum number of expression features retained for DeepSurv.")
    parser.add_argument("--min-expression-rate", type=float, default=0.05, help="Minimum non-zero rate for expression features.")
    return parser.parse_args()


def parse_int_grid(raw: str) -> list[int]:
    return sorted({int(item) for item in raw.split(",") if item.strip()})


def parse_float_grid(raw: str) -> list[float]:
    return sorted({float(item) for item in raw.split(",") if item.strip()})


def build_candidate_rows(
    hidden_dim1_list: list[int],
    hidden_dim2_list: list[int],
    dropout_list: list[float],
    learning_rate_list: list[float],
    batch_size_list: list[int],
    *,
    batch_norm: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hidden_dim1 in hidden_dim1_list:
        for hidden_dim2 in hidden_dim2_list:
            for dropout in dropout_list:
                for learning_rate in learning_rate_list:
                    for batch_size in batch_size_list:
                        rows.append(
                            {
                                "hidden_dim1": hidden_dim1,
                                "hidden_dim2": hidden_dim2,
                                "dropout": dropout,
                                "learning_rate": learning_rate,
                                "batch_size": batch_size,
                                "batch_norm": batch_norm,
                            }
                        )
    return rows


def network_complexity(hidden_dim1: int, hidden_dim2: int, num_features: int, batch_norm: bool) -> int:
    base = (num_features + 1) * hidden_dim1 + (hidden_dim1 + 1) * hidden_dim2 + hidden_dim2 + 1
    if batch_norm:
        base += hidden_dim1 + hidden_dim2
    return int(base)


def rank_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    ranked = results.copy()
    return ranked.sort_values(
        ["mean_c_index", "n_parameters", "dropout", "learning_rate", "batch_size"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def choose_row(results: pd.DataFrame, *, use_one_se_rule: bool = True) -> tuple[pd.Series, pd.Series]:
    if results.empty:
        raise RuntimeError("No DeepSurv candidate could be fitted successfully.")

    ranked = rank_results(results)
    best_row = ranked.iloc[0]
    if not use_one_se_rule:
        return best_row, best_row

    threshold = float(best_row["mean_c_index"]) - float(best_row["sem_c_index"])
    eligible = ranked[ranked["mean_c_index"] >= threshold].copy()
    if eligible.empty:
        return best_row, best_row

    eligible = eligible.sort_values(
        ["n_parameters", "dropout", "learning_rate", "batch_size"],
        ascending=[True, False, True, False],
    ).reset_index(drop=True)
    return best_row, eligible.iloc[0]


def build_readable_search_table(
    search_results: pd.DataFrame,
    *,
    selected_row: pd.Series,
    best_row: pd.Series,
) -> pd.DataFrame:
    if search_results.empty:
        return search_results.copy()

    readable = search_results.copy().reset_index(drop=True)
    readable.insert(0, "rank", np.arange(1, len(readable) + 1))
    readable.insert(1, "is_selected", False)
    readable.insert(2, "is_best", False)

    selected_key = (
        float(selected_row["hidden_dim1"]),
        float(selected_row["hidden_dim2"]),
        float(selected_row["dropout"]),
        float(selected_row["learning_rate"]),
        float(selected_row["batch_size"]),
        bool(selected_row["batch_norm"]),
    )
    best_key = (
        float(best_row["hidden_dim1"]),
        float(best_row["hidden_dim2"]),
        float(best_row["dropout"]),
        float(best_row["learning_rate"]),
        float(best_row["batch_size"]),
        bool(best_row["batch_norm"]),
    )

    def _make_key(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            list(
                zip(
                    frame["hidden_dim1"].astype(float),
                    frame["hidden_dim2"].astype(float),
                    frame["dropout"].astype(float),
                    frame["learning_rate"].astype(float),
                    frame["batch_size"].astype(float),
                    frame["batch_norm"].astype(bool),
                )
            ),
            index=frame.index,
        )

    keys = _make_key(readable)
    readable.loc[keys == selected_key, "is_selected"] = True
    readable.loc[keys == best_key, "is_best"] = True

    readable = readable[
        [
            "rank",
            "is_selected",
            "is_best",
            "hidden_dim1",
            "hidden_dim2",
            "dropout",
            "learning_rate",
            "batch_size",
            "batch_norm",
            "mean_c_index",
            "sem_c_index",
            "n_scores",
            "failed_folds",
            "n_parameters",
        ]
    ].copy()

    readable["mean_c_index"] = readable["mean_c_index"].round(4)
    readable["sem_c_index"] = readable["sem_c_index"].round(4)
    readable = readable.sort_values(["is_selected", "mean_c_index", "n_parameters"], ascending=[False, False, True])
    readable = readable.reset_index(drop=True)
    readable["rank"] = np.arange(1, len(readable) + 1)
    return readable


def evaluate_candidates(
    train_df: pd.DataFrame,
    candidate_rows: list[dict[str, object]],
    *,
    duration_col: str,
    event_col: str,
    random_state: int,
    validation_split: float,
    epochs: int,
    patience: int,
    no_batch_norm: bool,
    log_transform: bool,
    max_features: int,
    min_expression_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(train_df))
    labels = train_df[event_col].fillna("__nan__").astype(str).to_numpy()
    fold_bins: list[list[int]] = [[] for _ in range(3)]
    for label in pd.unique(labels):
        label_idx = indices[labels == label]
        rng.shuffle(label_idx)
        for fold, chunk in enumerate(np.array_split(label_idx, 3)):
            fold_bins[fold].extend(chunk.tolist())

    repeated_splits = []
    for repeat in range(2):
        repeat_splits = []
        for fold in range(3):
            val_idx = np.array(sorted(fold_bins[fold]), dtype=int)
            train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)
            repeat_splits.append((train_idx, val_idx))
        repeated_splits.append(repeat_splits)

    for params in candidate_rows:
        fold_scores: list[float] = []
        failed_folds = 0
        hidden_dim1 = int(params["hidden_dim1"])
        hidden_dim2 = int(params["hidden_dim2"])
        dropout = float(params["dropout"])
        learning_rate = float(params["learning_rate"])
        batch_size = int(params["batch_size"])
        batch_norm = bool(params["batch_norm"]) and not no_batch_norm

        for repeat_idx, splits in enumerate(repeated_splits):
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_train = train_df.iloc[train_idx]
                fold_val = train_df.iloc[val_idx]
                try:
                    model = DeepSurvModel(
                        hidden_dim1=hidden_dim1,
                        hidden_dim2=hidden_dim2,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        batch_norm=batch_norm,
                        batch_size=batch_size,
                        epochs=epochs,
                        patience=patience,
                        validation_split=validation_split,
                        log_transform=log_transform,
                        max_features=max_features,
                        min_expression_rate=min_expression_rate,
                        random_state=random_state + repeat_idx * 100 + fold_idx,
                    ).fit(fold_train, duration_col, event_col)
                    score = model.score(fold_val)
                    fold_scores.append(float(score))
                    fold_rows.append(
                        {
                            **params,
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "c_index": float(score),
                            "status": "ok",
                            "n_parameters": model.n_parameters_,
                        }
                    )
                except Exception:
                    failed_folds += 1
                    fold_rows.append(
                        {
                            **params,
                            "repeat": repeat_idx,
                            "fold": fold_idx,
                            "c_index": np.nan,
                            "status": "failed",
                            "n_parameters": network_complexity(hidden_dim1, hidden_dim2, max_features, batch_norm),
                        }
                    )

        valid_scores = np.asarray(fold_scores, dtype=float)
        n_scores = int(valid_scores.size)
        mean_score = float(valid_scores.mean()) if n_scores else float("-inf")
        std_score = float(valid_scores.std(ddof=1)) if n_scores > 1 else 0.0
        sem_score = float(std_score / np.sqrt(n_scores)) if n_scores else float("inf")
        aggregate_rows.append(
            {
                **params,
                "mean_c_index": mean_score,
                "std_c_index": std_score,
                "sem_c_index": sem_score,
                "n_scores": n_scores,
                "failed_folds": failed_folds,
                "n_parameters": network_complexity(hidden_dim1, hidden_dim2, max_features, batch_norm),
            }
        )

    return rank_results(pd.DataFrame(aggregate_rows)), pd.DataFrame(fold_rows)


def main() -> None:
    args = parse_args()
    train_df = pd.read_csv(args.input / "train.csv")
    test_df = pd.read_csv(args.input / "test.csv")
    output_dir = args.output or args.input.parent

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    if DeepSurvModel is None:
        summary = pd.DataFrame(
            [
                {
                    "model": "deep",
                    "status": "skipped",
                    "reason": "DeepSurv dependencies are not installed",
                }
            ]
        )
        summary.to_csv(final_dir / "deepsurv_summary.csv", index=False)
        print(summary.to_string(index=False))
        return

    hidden_dim1_grid = parse_int_grid(args.hidden_dim1_grid)
    hidden_dim2_grid = parse_int_grid(args.hidden_dim2_grid)
    dropout_grid = parse_float_grid(args.dropout_grid)
    learning_rate_grid = parse_float_grid(args.learning_rate_grid)
    batch_size_grid = parse_int_grid(args.batch_size_grid)

    candidates = build_candidate_rows(
        hidden_dim1_grid,
        hidden_dim2_grid,
        dropout_grid,
        learning_rate_grid,
        batch_size_grid,
        batch_norm=not args.no_batch_norm,
    )
    search_results, search_folds = evaluate_candidates(
        train_df,
        candidates,
        duration_col="duration",
        event_col="event",
        random_state=args.random_state,
        validation_split=args.validation_split,
        epochs=args.epochs,
        patience=args.patience,
        no_batch_norm=args.no_batch_norm,
        log_transform=bool(args.log_transform),
        max_features=args.max_features,
        min_expression_rate=args.min_expression_rate,
    )

    if search_results.empty:
        raise RuntimeError("No DeepSurv candidate could be fitted successfully.")

    search_results = rank_results(search_results.drop_duplicates(
        subset=["hidden_dim1", "hidden_dim2", "dropout", "learning_rate", "batch_size", "batch_norm"],
        keep="first",
    ).reset_index(drop=True))

    best_row, selected_row = choose_row(search_results, use_one_se_rule=True)
    selected_hidden_dim1 = int(selected_row["hidden_dim1"])
    selected_hidden_dim2 = int(selected_row["hidden_dim2"])
    selected_dropout = float(selected_row["dropout"])
    selected_learning_rate = float(selected_row["learning_rate"])
    selected_batch_size = int(selected_row["batch_size"])
    selected_batch_norm = bool(selected_row["batch_norm"])

    model = DeepSurvModel(
        hidden_dim1=selected_hidden_dim1,
        hidden_dim2=selected_hidden_dim2,
        dropout=selected_dropout,
        learning_rate=selected_learning_rate,
        batch_size=selected_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        validation_split=args.validation_split,
        batch_norm=selected_batch_norm,
        log_transform=bool(args.log_transform),
        max_features=args.max_features,
        min_expression_rate=args.min_expression_rate,
        random_state=args.random_state,
    ).fit(train_df, "duration", "event")

    summary = pd.DataFrame(
        [
            {
                "model": "deep",
                "selected_rule": "one_se",
                "selected_hidden_dim1": selected_hidden_dim1,
                "selected_hidden_dim2": selected_hidden_dim2,
                "selected_dropout": selected_dropout,
                "selected_learning_rate": selected_learning_rate,
                "selected_batch_size": selected_batch_size,
                "selected_batch_norm": selected_batch_norm,
                "selected_cv_c_index": float(selected_row["mean_c_index"]),
                "selected_cv_sem_c_index": float(selected_row["sem_c_index"]),
                "selected_cv_n_parameters": int(selected_row["n_parameters"]),
                "best_cv_c_index": float(best_row["mean_c_index"]),
                "best_cv_n_parameters": int(best_row["n_parameters"]),
                "train_c_index": model.score(train_df),
                "test_c_index": model.score(test_df),
                "n_features": len(model.feature_cols or []),
                "status": "ok",
            }
        ]
    )

    readable_search = build_readable_search_table(
        search_results,
        selected_row=selected_row,
        best_row=best_row,
    )

    summary.to_csv(final_dir / "deepsurv_summary.csv", index=False)
    search_results.to_csv(final_dir / "deepsurv_search.csv", index=False)
    readable_search.to_csv(final_dir / "deepsurv_search_readable.csv", index=False)
    search_folds.to_csv(final_dir / "deepsurv_search_folds.csv", index=False)

    scored_test = test_df.copy()
    scored_test["deep_risk_score"] = model.predict_risk_score(test_df).values
    scored_test["risk_group"] = split_risk_group(scored_test["deep_risk_score"])
    scored_test.to_csv(final_dir / "deepsurv_test_scored.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

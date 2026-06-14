"""Data loading and preprocessing helpers for survival analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .exceptions import DataValidationError

__all__ = [
    "load_table",
    "load_clinical_data",
    "load_expression_matrix",
    "normalize_survival_data",
    "prepare_feature_matrix",
    "prepare_survival_dataset",
    "split_train_test",
    "split_train_val_test",
]


def load_table(file_path: str | Path, sep: str | None = None, comment: str | None = "#") -> pd.DataFrame:
    """Load a delimited text file into a DataFrame."""
    path = Path(file_path)
    if sep is None:
        suffix = path.suffix.lower()
        if suffix in {".tsv", ".txt"}:
            sep = "\t"
        else:
            sep = ","

    try:
        return pd.read_csv(path, sep=sep, comment=comment)
    except Exception as exc:  # pragma: no cover - pandas error details vary
        raise DataValidationError(f"Failed to read table from {path}: {exc}") from exc


def load_clinical_data(file_path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Load a clinical survival table."""
    return load_table(file_path, sep=sep, comment="#")


def _looks_like_sample_id(value: object) -> bool:
    text = str(value).strip()
    return bool(re.match(r"^TCGA-[A-Z0-9\-]+", text))


def _looks_like_gene_id(value: object) -> bool:
    text = str(value).strip()
    return bool(re.match(r"^(ENSG|gene|Gene|ENST|A_)", text))


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _first_non_missing_value(row: pd.Series, candidates: Iterable[str]) -> float:
    for candidate in candidates:
        if candidate not in row.index:
            continue
        value = row[candidate]
        if pd.notna(value) and str(value).strip() not in {"", "'--", "--"}:
            try:
                return float(value)
            except (TypeError, ValueError):
                numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.notna(numeric):
                    return float(numeric)
    return np.nan


def _get_case_patients_from_sample(sample_series: pd.Series) -> pd.Series:
    return (
        sample_series.astype(str).str.replace(r"-\d+[A-Z]?$", "", regex=True)
    )


def _encode_stage_value(value: object) -> float:
    mapping = {
        "STAGE 0": 0.0,
        "STAGE I": 1.0,
        "STAGE IA": 1.0,
        "STAGE IB": 1.0,
        "STAGE IC": 1.0,
        "STAGE II": 2.0,
        "STAGE IIA": 2.0,
        "STAGE IIB": 2.0,
        "STAGE IIC": 2.0,
        "STAGE III": 3.0,
        "STAGE IIIA": 3.0,
        "STAGE IIIB": 3.0,
        "STAGE IIIC": 3.0,
        "STAGE IV": 4.0,
        "STAGE IVA": 4.0,
        "STAGE IVB": 4.0,
    }
    text = str(value).strip().upper()
    if text in mapping:
        return mapping[text]
    return float("nan")


def _encode_gender_value(value: object) -> float:
    mapping = {
        "male": 0.0,
        "female": 1.0,
    }
    return float(mapping.get(str(value).strip().lower(), np.nan))


def _prepare_gdc_clinical_table(clinical_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a wide GDC clinical TSV to patient-level survival data."""
    cleaned = clinical_df.replace({"'--": np.nan, "--": np.nan}).copy()

    patient_col = _first_existing_column(
        cleaned,
        [
            "submitter_id",
            "cases.submitter_id",
            "case_submitter_id.annotations",
            "demographic.submitter_id",
            "sample",
            "PATIENT_ID",
            "patient_id",
            "_PATIENT",
        ],
    )
    if patient_col is None:
        raise DataValidationError("Could not infer patient column from GDC clinical table.")

    if patient_col == "sample":
        cleaned = _pick_primary_tumor_samples(cleaned, "sample")

    cleaned = cleaned.dropna(subset=[patient_col]).copy()
    cleaned = cleaned.groupby(patient_col, sort=False).first().reset_index()
    cleaned = cleaned.rename(columns={patient_col: "_PATIENT"})

    status_col = _first_existing_column(
        cleaned,
        [
            "vital_status.demographic",
            "demographic.vital_status",
            "vital_status",
            "status.demographic",
        ],
    )
    if status_col is None:
        raise DataValidationError("GDC clinical table is missing a vital-status column.")

    status_mapping = {
        "Alive": 0,
        "alive": 0,
        "Censored": 0,
        "censored": 0,
        "0": 0,
        0: 0,
        "LWT": 0,
        "0:LIVING": 0,
        "LIVING": 0,
        "Dead": 1,
        "dead": 1,
        "Event": 1,
        "event": 1,
        "1": 1,
        1: 1,
        "1:DECEASED": 1,
        "DECEASED": 1,
    }

    cleaned["event"] = cleaned[status_col].astype(str).str.strip().map(status_mapping)
    death_presence_col = _first_existing_column(
        cleaned,
        [
            "days_to_death.demographic",
            "demographic.days_to_death",
            "year_of_death.demographic",
        ],
    )
    cleaned["event"] = cleaned["event"].fillna(
        cleaned[death_presence_col].notna().astype(int) if death_presence_col is not None else np.nan
    )
    cleaned["event"] = pd.to_numeric(cleaned["event"], errors="coerce")

    death_candidates = [
        "days_to_death.demographic",
        "demographic.days_to_death",
        "days_to_diagnosis.diagnoses",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_last_known_disease_status",
        "cases.days_to_lost_to_followup",
    ]
    followup_candidates = [
        "days_to_last_follow_up.diagnoses",
        "days_to_last_follow_up.demographic",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.days_to_last_known_disease_status",
        "days_to_last_follow_up",
        "cases.days_to_lost_to_followup",
        "cases.days_to_consent",
    ]
    all_duration_candidates = death_candidates + followup_candidates

    duration = pd.Series(np.nan, index=cleaned.index, dtype=float)
    dead_mask = cleaned["event"] == 1
    alive_mask = cleaned["event"] == 0

    if dead_mask.any():
        duration.loc[dead_mask] = cleaned.loc[dead_mask].apply(
            lambda row: _first_non_missing_value(row, death_candidates),
            axis=1,
        )
    if alive_mask.any():
        duration.loc[alive_mask] = cleaned.loc[alive_mask].apply(
            lambda row: _first_non_missing_value(row, followup_candidates),
            axis=1,
        )

    missing_mask = duration.isna()
    if missing_mask.any():
        duration.loc[missing_mask] = cleaned.loc[missing_mask].apply(
            lambda row: _first_non_missing_value(row, all_duration_candidates),
            axis=1,
        )

    cleaned["duration"] = pd.to_numeric(duration, errors="coerce")

    age_col = _first_existing_column(
        cleaned,
        [
            "age_at_index.demographic",
            "demographic.age_at_index",
            "age_at_index",
            "age_at_diagnosis.diagnoses",
        ],
    )
    cleaned["age"] = pd.to_numeric(cleaned[age_col], errors="coerce") if age_col is not None else np.nan

    gender_col = _first_existing_column(
        cleaned,
        ["gender.demographic", "demographic.gender", "demographic.sex_at_birth", "gender"],
    )
    cleaned["gender_num"] = cleaned[gender_col].map(_encode_gender_value) if gender_col is not None else np.nan

    stage_col = _first_existing_column(
        cleaned,
        [
            "ajcc_pathologic_stage.diagnoses",
            "ajcc_clinical_stage.diagnoses",
            "uicc_pathologic_stage.diagnoses",
            "uicc_clinical_stage.diagnoses",
            "diagnoses.ajcc_pathologic_stage",
            "diagnoses.ajcc_clinical_stage",
            "diagnoses.uicc_pathologic_stage",
            "diagnoses.uicc_clinical_stage",
        ],
    )
    cleaned["stage_num"] = cleaned[stage_col].map(_encode_stage_value) if stage_col is not None else np.nan

    keep_cols = ["_PATIENT", "duration", "event", "age", "gender_num", "stage_num"]
    cleaned = cleaned[keep_cols].dropna(subset=["duration", "event"]).reset_index(drop=True)

    if len(cleaned) == 0:
        raise DataValidationError("No valid rows remain after normalizing GDC clinical table.")

    unique_events = pd.Series(cleaned["event"]).dropna().unique().tolist()
    if len(unique_events) < 2:
        raise DataValidationError(
            f"GDC clinical table must contain at least two event classes; got {unique_events}."
        )

    cleaned["event"] = cleaned["event"].astype(int)
    return cleaned


def load_expression_matrix(
    file_path: str | Path,
    gene_col: str | None = None,
) -> pd.DataFrame:
    """Load an expression matrix and normalize it to sample rows."""
    df = load_table(file_path, sep=None, comment="#")
    if gene_col is None:
        gene_col = df.columns[0]
    if gene_col not in df.columns:
        raise DataValidationError(f"Missing gene identifier column: {gene_col}")

    first_col = df.columns[0]
    sample_like_cols = sum(_looks_like_sample_id(col) for col in df.columns[1:])
    first_col_sample_like = df[first_col].astype(str).map(_looks_like_sample_id).mean()
    first_col_gene_like = df[first_col].astype(str).map(_looks_like_gene_id).mean()

    if gene_col == first_col and sample_like_cols > 0 and first_col_gene_like >= 0.3:
        expr = df.set_index(gene_col).T.reset_index(names="sample")
    elif first_col_sample_like >= 0.5:
        expr = df.copy()
        if first_col != "sample":
            expr = expr.rename(columns={first_col: "sample"})
    else:
        expr = df.set_index(gene_col).T.reset_index(names="sample")

    expr["_PATIENT"] = (
        expr["sample"]
        .astype(str)
        .str.replace(r"-\d+[A-Z]?$", "", regex=True)
    )
    return expr


def normalize_survival_data(
    df: pd.DataFrame,
    time_col: str,
    status_col: str,
    id_col: str | None = None,
) -> pd.DataFrame:
    """Standardize survival columns to `duration` and `event`."""
    processed_df = df.copy()

    if time_col not in processed_df.columns:
        raise DataValidationError(f"Missing time column: {time_col}")
    if status_col not in processed_df.columns:
        raise DataValidationError(f"Missing status column: {status_col}")

    processed_df["duration"] = pd.to_numeric(processed_df[time_col], errors="coerce")

    status_mapping = {
        "Alive": 0,
        "alive": 0,
        "Censored": 0,
        "censored": 0,
        "0": 0,
        0: 0,
        "LWT": 0,
        "0:LIVING": 0,
        "LIVING": 0,
        "Dead": 1,
        "dead": 1,
        "Event": 1,
        "event": 1,
        "1": 1,
        1: 1,
        "1:DECEASED": 1,
        "DECEASED": 1,
    }

    processed_df["event"] = processed_df[status_col].astype(str).str.strip().map(status_mapping)
    before_count = len(processed_df)
    processed_df = processed_df.dropna(subset=["duration", "event"]).copy()
    processed_df["event"] = processed_df["event"].astype(int)

    keep_cols = ["duration", "event"]
    if id_col and id_col in processed_df.columns:
        keep_cols.insert(0, id_col)

    feature_cols = [
        col
        for col in processed_df.columns
        if col not in {time_col, status_col, "duration", "event", id_col}
    ]

    processed_df = processed_df[keep_cols + feature_cols]
    processed_df = processed_df.reset_index(drop=True)

    if len(processed_df) == 0:
        raise DataValidationError(
            f"No valid survival rows remain after cleaning (removed {before_count} rows)."
        )

    return processed_df


def _infer_patient_column(df: pd.DataFrame) -> str:
    candidates = [
        "_PATIENT",
        "submitter_id",
        "cases.submitter_id",
        "demographic.submitter_id",
        "PATIENT_ID",
        "patient_id",
        "sample",
        "Sample",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise DataValidationError("Could not infer patient/sample ID column.")


def _pick_primary_tumor_samples(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    if sample_col != "sample":
        return df

    mask = df[sample_col].astype(str).str.contains(r"-01[A-Z]?$", regex=True, na=False)
    filtered = df.loc[mask].copy()
    return filtered if len(filtered) > 0 else df.copy()


def prepare_feature_matrix(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    *,
    excluded_cols: Iterable[str] | None = None,
    log_transform: bool = False,
    pseudo_count: float = 1.0,
    min_variance: float = 1e-8,
    min_expression_rate: float = 0.1,
    max_features: int | None = 200,
) -> tuple[pd.DataFrame, list[str]]:
    """Extract a numeric model matrix from a survival table.

    The default preprocessing keeps raw numeric features, filters low-expression
    columns, and keeps the highest-variance features. Set `log_transform=True`
    to apply `log2(x + pseudo_count)` before selection.
    """
    if duration_col not in data.columns:
        raise DataValidationError(f"Missing duration column: {duration_col}")
    if event_col not in data.columns:
        raise DataValidationError(f"Missing event column: {event_col}")

    excluded = {
        duration_col,
        event_col,
        "_PATIENT",
        "age",
        "gender_num",
        "stage_num",
        "PATIENT_ID",
        "patient_id",
        "ID",
        "id",
        "sample",
    }
    if excluded_cols is not None:
        excluded.update(excluded_cols)

    feature_cols: list[str] = []
    feature_series: list[pd.Series] = []

    for col in data.columns:
        if col in excluded:
            continue

        values = pd.to_numeric(data[col], errors="coerce")
        if values.notna().sum() == 0:
            continue
        if values.nunique(dropna=True) <= 1:
            continue

        is_non_negative = bool((values.dropna() >= 0).all())
        if log_transform and is_non_negative:
            values = pd.Series(
                np.log2(values.astype(float) + float(pseudo_count)),
                index=values.index,
                name=col,
            )
        elif log_transform and not is_non_negative:
            values = values.astype(float)

        if is_non_negative and float((values > 0).mean()) < min_expression_rate:
            continue

        feature_cols.append(col)
        feature_series.append(values.rename(col))

    if not feature_cols:
        raise DataValidationError("No usable numeric features found.")

    numeric_features = pd.concat(feature_series, axis=1)

    variances = numeric_features.var(axis=0, numeric_only=True).sort_values(ascending=False)
    variances = variances[variances > min_variance]
    if max_features is not None:
        variances = variances.head(max_features)
    if variances.empty:
        raise DataValidationError("No usable features remain after variance filtering.")

    selected_cols = variances.index.tolist()
    numeric_features = numeric_features[selected_cols]
    feature_cols = selected_cols

    model_data = pd.concat([data[[duration_col, event_col]], numeric_features], axis=1)
    model_data[duration_col] = pd.to_numeric(model_data[duration_col], errors="coerce")
    model_data[event_col] = pd.to_numeric(model_data[event_col], errors="coerce")
    model_data = model_data.dropna().reset_index(drop=True)

    if len(model_data) == 0:
        raise DataValidationError("No usable rows remain after cleaning.")

    feature_cols = [col for col in model_data.columns if col not in {duration_col, event_col}]
    if feature_cols:
        model_data[feature_cols] = model_data[feature_cols].apply(
            lambda col: col.fillna(col.median()) if pd.api.types.is_numeric_dtype(col) else col
        )

    unique_events = pd.Series(model_data[event_col]).dropna().unique().tolist()
    if len(unique_events) < 2:
        raise DataValidationError(
            f"Event column '{event_col}' must contain at least two classes; got {unique_events}."
        )

    return model_data, feature_cols


def prepare_survival_dataset(
    clinical_df: pd.DataFrame,
    expression_df: pd.DataFrame,
    *,
    clinical_patient_col: str | None = None,
    expression_patient_col: str | None = None,
    time_col: str = "OS.time",
    status_col: str = "OS",
    primary_tumor_only: bool = True,
) -> pd.DataFrame:
    """Merge clinical and expression tables into a patient-level analysis set."""
    gdc_clinical_style = (
        (
            "submitter_id" in clinical_df.columns
            or "cases.submitter_id" in clinical_df.columns
            or "sample" in clinical_df.columns
        )
        and (
            "vital_status.demographic" in clinical_df.columns
            or "demographic.vital_status" in clinical_df.columns
            or "vital_status" in clinical_df.columns
        )
        and (
            "days_to_death.demographic" in clinical_df.columns
            or "demographic.days_to_death" in clinical_df.columns
            or "days_to_last_follow_up.diagnoses" in clinical_df.columns
            or "diagnoses.days_to_last_follow_up" in clinical_df.columns
        )
    )

    if gdc_clinical_style:
        clinical = _prepare_gdc_clinical_table(clinical_df)
    else:
        if clinical_patient_col is None:
            clinical_patient_col = _infer_patient_column(clinical_df)
        clinical = normalize_survival_data(
            clinical_df,
            time_col=time_col,
            status_col=status_col,
            id_col=clinical_patient_col,
        ).copy()
        clinical = clinical.rename(columns={clinical_patient_col: "_PATIENT"})
        clinical = clinical.drop_duplicates(subset=["_PATIENT"], keep="first")

    expr = expression_df.copy()
    if primary_tumor_only and "sample" in expr.columns:
        expr = _pick_primary_tumor_samples(expr, "sample")

    if expression_patient_col is None:
        expression_patient_col = _infer_patient_column(expr)

    if "sample" in expr.columns and expression_patient_col == "sample":
        expr = expr.copy()
        expr["_PATIENT"] = _get_case_patients_from_sample(expr["sample"])
        expression_patient_col = "_PATIENT"

    if expression_patient_col not in expr.columns:
        raise DataValidationError(
            f"Expression table is missing patient column: {expression_patient_col}"
        )

    expr = expr.rename(columns={expression_patient_col: "_PATIENT"})
    expr = expr.drop_duplicates(subset=["_PATIENT"], keep="first")

    merged = clinical.merge(expr, on="_PATIENT", how="inner", validate="one_to_one")
    if len(merged) == 0:
        raise DataValidationError("No overlapping patients after merging clinical and expression tables.")

    leading_cols = [col for col in ["_PATIENT", "duration", "event", "age", "gender_num", "stage_num"] if col in merged.columns]
    trailing_cols = [col for col in merged.columns if col not in leading_cols]
    merged = merged[leading_cols + trailing_cols]

    return merged.reset_index(drop=True)


def split_train_test(
    data: pd.DataFrame,
    *,
    test_size: float = 0.30,
    stratify_col: str | None = "event",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a patient-level table into train and test sets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(data) < 2:
        raise ValueError("Need at least 2 rows to split into train/test.")

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(data))

    if stratify_col is None or stratify_col not in data.columns:
        rng.shuffle(indices)
        test_count = max(1, int(round(len(data) * test_size)))
        test_count = min(test_count, len(data) - 1)
        test_idx = indices[:test_count]
        train_idx = indices[test_count:]
    else:
        stratify_values = data[stratify_col].fillna("__nan__").astype(str).to_numpy()
        class_counts = pd.Series(stratify_values).value_counts()
        if class_counts.min() < 2:
            rng.shuffle(indices)
            test_count = max(1, int(round(len(data) * test_size)))
            test_count = min(test_count, len(data) - 1)
            test_idx = indices[:test_count]
            train_idx = indices[test_count:]
        else:
            train_parts: list[np.ndarray] = []
            test_parts: list[np.ndarray] = []
            for label in class_counts.index:
                label_idx = indices[stratify_values == label]
                rng.shuffle(label_idx)
                test_count = int(round(len(label_idx) * test_size))
                test_count = max(1, test_count)
                test_count = min(test_count, len(label_idx) - 1)
                test_parts.append(label_idx[:test_count])
                train_parts.append(label_idx[test_count:])
            train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
            test_idx = np.concatenate(test_parts) if test_parts else np.array([], dtype=int)
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)

    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    return train, test


def split_train_val_test(
    data: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_col: str | None = "event",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Backward-compatible helper that returns train/val/test."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")
    if len(data) < 3:
        raise ValueError("Need at least 3 rows to split into train/val/test.")

    train, test = split_train_test(
        data,
        test_size=test_size + val_size,
        stratify_col=stratify_col,
        random_state=random_state,
    )
    # Split the held-out portion again into validation and test using the same helper.
    val, test = split_train_test(
        test,
        test_size=test_size / (test_size + val_size),
        stratify_col=stratify_col,
        random_state=random_state,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

"""Merge GDC clinical data with TCGA survival data to generate an enhanced clinical TSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, required=True,
                        help="Path to GDC clinical TSV, e.g. TCGA-STAD.clinical.tsv")
    parser.add_argument("--survival", type=Path, required=True,
                        help="Path to the original TCGA survival TSV, e.g. TCGA-STAD.survival.tsv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output merged TSV path, e.g. TCGA-STAD.clinical_merged.tsv")
    return parser.parse_args()


def encode_stage(stage: str) -> float:
    """把分期字符串转成数值."""
    mapping = {
        "Stage I": 1, "Stage IA": 1, "Stage IB": 1,
        "Stage II": 2, "Stage IIA": 2, "Stage IIB": 2, "Stage IIC": 2,
        "Stage III": 3, "Stage IIIA": 3, "Stage IIIB": 3, "Stage IIIC": 3,
        "Stage IV": 4, "Stage IVA": 4, "Stage IVB": 4,
    }
    return float(mapping.get(str(stage).strip(), np.nan))


def encode_gender(gender: str) -> float:
    mapping = {"male": 0.0, "female": 1.0}
    return float(mapping.get(str(gender).strip().lower(), np.nan))


def encode_grade(grade: str) -> float:
    mapping = {"G1": 1, "G2": 2, "G3": 3, "G4": 4}
    return float(mapping.get(str(grade).strip(), np.nan))


def main() -> None:
    args = parse_args()

    # 读取文件
    print("Reading GDC clinical TSV...")
    clinical = pd.read_csv(args.clinical, sep="\t", low_memory=False)

    print("Reading survival TSV...")
    survival = pd.read_csv(args.survival, sep="\t")

    # 替换'--为NaN
    clinical = clinical.replace("'--", np.nan)

    # 每个患者只保留一行（去重）
    clinical = clinical.drop_duplicates(subset=["cases.submitter_id"], keep="first")

    # 提取有用的列
    cols = {
        "cases.submitter_id": "sample",
        "demographic.age_at_index": "age",
        "demographic.gender": "gender",
        "diagnoses.ajcc_pathologic_stage": "stage",
        "diagnoses.tumor_grade": "tumor_grade",
        "diagnoses.ajcc_pathologic_t": "T_stage",
        "diagnoses.ajcc_pathologic_n": "N_stage",
        "diagnoses.ajcc_pathologic_m": "M_stage",
    }
    available_cols = {k: v for k, v in cols.items() if k in clinical.columns}
    clinical_slim = clinical[list(available_cols.keys())].rename(columns=available_cols)

    # 编码
    if "stage" in clinical_slim.columns:
        clinical_slim["stage_num"] = clinical_slim["stage"].map(encode_stage)
    if "gender" in clinical_slim.columns:
        clinical_slim["gender_num"] = clinical_slim["gender"].map(encode_gender)
    if "tumor_grade" in clinical_slim.columns:
        clinical_slim["grade_num"] = clinical_slim["tumor_grade"].map(encode_grade)
    if "age" in clinical_slim.columns:
        clinical_slim["age"] = pd.to_numeric(clinical_slim["age"], errors="coerce")

    # 和survival合并
    print("Merging tables...")
    # 从sample ID提取患者ID（去掉-01A这样的后缀）
    survival["patient_id"] = survival["sample"].str.replace(r"-\d+[A-Z]+$", "", regex=True)
    clinical_slim = clinical_slim.rename(columns={"sample": "patient_id"})
    merged = survival.merge(clinical_slim, on="patient_id", how="inner")

    print(f"Merged sample count: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")

    # 统计各列缺失率
    print("\nMissing rate for clinical covariates:")
    for col in ["age", "stage_num", "gender_num", "grade_num"]:
        if col in merged.columns:
            missing = merged[col].isna().mean()
            print(f"  {col}: {missing:.1%} missing")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, sep="\t", index=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()

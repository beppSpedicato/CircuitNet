"""Merge CircuitNet-N28 and CircuitNet-N14 DRC datasets into a single training set.

Reads the per-dataset annotation CSVs (feature_rel_path, label_rel_path), drops
duplicate rows within each CSV and across the two datasets, copies the referenced
.npy files from each dataset's training-set folder into a unified output folder
(output_folder/feature/, output_folder/label/), and writes merged annotation
CSVs (merged_train.csv, merged_test.csv) whose paths are relative to
output_folder.
"""

import os
import os.path as osp
import shutil
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Setup (config variables)
# ---------------------------------------------------------------------------
N28_CSV_FILE_TRAIN = "../drc_prediction/files/train_N28.csv"
N28_CSV_FILE_TEST  = "../drc_prediction/files/test_N28.csv"
N14_CSV_FILE_TRAIN = "../drc_prediction/files/train_N14.csv"
N14_CSV_FILE_TEST  = "../drc_prediction/files/test_N14.csv"

N28_TRAINING_SET = "../drc_prediction/training_set/DRC"
N14_TRAINING_SET = "../drc_prediction/training_set/DRC_N14"

OUTPUT_FOLDER = "../drc_prediction/training_set/DRC_merged"

FEATURE_SUBDIR = "feature"
LABEL_SUBDIR   = "label"


# ---------------------------------------------------------------------------
# 2. CSV reading + duplicate removal
# ---------------------------------------------------------------------------
def read_csv(path: str) -> pd.DataFrame:
    """Read a CircuitNet annotation CSV: two comma-separated relative paths."""
    df = pd.read_csv(
        path,
        header=None,
        names=["feature_path", "label_path"],
        skip_blank_lines=True,
    )
    df["feature_path"] = df["feature_path"].str.strip()
    df["label_path"]   = df["label_path"].str.strip()
    return df


def drop_duplicates(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["feature_path", "label_path"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  [{tag}] dropped {dropped} duplicate row(s), kept {len(df)}")
    else:
        print(f"  [{tag}] no duplicates, {len(df)} rows")
    return df


# ---------------------------------------------------------------------------
# 3./4. Copy files + merge dataframes
# ---------------------------------------------------------------------------
def ensure_output_dirs(output_folder: str) -> Tuple[str, str]:
    feat_dir = osp.join(output_folder, FEATURE_SUBDIR)
    lbl_dir  = osp.join(output_folder, LABEL_SUBDIR)
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(lbl_dir,  exist_ok=True)
    return feat_dir, lbl_dir


def copy_pair(src_root: str, rel_path: str, dst_root: str) -> str:
    """Copy src_root/rel_path -> dst_root/rel_path, return the destination rel path."""
    src = osp.join(src_root, rel_path)
    dst = osp.join(dst_root, rel_path)
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if not osp.exists(src):
        raise FileNotFoundError(f"Missing source file: {src}")
    if not osp.exists(dst):
        shutil.copy2(src, dst)
    return rel_path


def merge_split(
    n28_df: pd.DataFrame,
    n14_df: pd.DataFrame,
    output_folder: str,
    split_name: str,
) -> pd.DataFrame:
    """Copy files from both datasets into output_folder and return merged CSV rows."""
    n28_df = drop_duplicates(n28_df, f"N28 {split_name}")
    n14_df = drop_duplicates(n14_df, f"N14 {split_name}")

    combined = pd.concat(
        [n28_df.assign(source="N28"), n14_df.assign(source="N14")],
        ignore_index=True,
    )
    combined = drop_duplicates(
        combined.drop(columns=["source"]).assign(source=combined["source"]),
        f"combined {split_name}",
    )

    for _, row in combined.iterrows():
        src_root = N28_TRAINING_SET if row["source"] == "N28" else N14_TRAINING_SET
        copy_pair(src_root, row["feature_path"], output_folder)
        copy_pair(src_root, row["label_path"],   output_folder)

    return combined[["feature_path", "label_path"]]


def write_merged_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, header=False, index=False)
    print(f"  wrote {len(df)} rows -> {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    script_dir = osp.dirname(osp.abspath(__file__))

    def _abs(p: str) -> str:
        return p if osp.isabs(p) else osp.abspath(osp.join(script_dir, p))

    n28_train_csv = _abs(N28_CSV_FILE_TRAIN)
    n28_test_csv  = _abs(N28_CSV_FILE_TEST)
    n14_train_csv = _abs(N14_CSV_FILE_TRAIN)
    n14_test_csv  = _abs(N14_CSV_FILE_TEST)
    output_folder = _abs(OUTPUT_FOLDER)

    globals().update(
        N28_TRAINING_SET=_abs(N28_TRAINING_SET),
        N14_TRAINING_SET=_abs(N14_TRAINING_SET),
    )

    print("=== Reading CSVs ===")
    n28_train = read_csv(n28_train_csv)
    n28_test  = read_csv(n28_test_csv)
    n14_train = read_csv(n14_train_csv)
    n14_test  = read_csv(n14_test_csv)

    print("=== Preparing output directories ===")
    ensure_output_dirs(output_folder)

    print("\n=== Merging train split ===")
    merged_train = merge_split(n28_train, n14_train, output_folder, "train")
    write_merged_csv(merged_train, osp.join(output_folder, "merged_train.csv"))

    print("\n=== Merging test split ===")
    merged_test = merge_split(n28_test, n14_test, output_folder, "test")
    write_merged_csv(merged_test, osp.join(output_folder, "merged_test.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()

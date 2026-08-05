import os
import os.path as osp
import shutil
from typing import List, Tuple
import pandas as pd

OUTPUT_FOLDER = "../routability_ir_drop_prediction/training_set_N14/DRC"

FEATURE_SUBDIR = "feature"
LABEL_SUBDIR   = "label"

CSV_FILE = "../drc_prediction/files/test_N14.csv"

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

def main():
    df = read_csv(CSV_FILE)
    print(f"Read {len(df)} rows from {CSV_FILE}")

    feature_dir = osp.join(OUTPUT_FOLDER, FEATURE_SUBDIR)
    label_dir   = osp.join(OUTPUT_FOLDER, LABEL_SUBDIR)

    for idx, row in df.iterrows():
        feature_src = row["feature_path"]
        label_src   = row["label_path"]

        feature_dst = osp.join(feature_dir, osp.basename(feature_src))
        label_dst   = osp.join(label_dir, osp.basename(label_src))

        if not osp.exists(feature_dst) or not osp.exists(label_dst):
            df.drop(idx, inplace=True)
            print(f"  [WARNING] Missing file(s) for row {idx}:")

    df.to_csv(CSV_FILE, index=False, header=False)

if __name__ == "__main__":
    main()

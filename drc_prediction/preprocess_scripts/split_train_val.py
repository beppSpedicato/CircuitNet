#!/usr/bin/env python3

import argparse
import os
import os.path as osp
import random
import re
from collections import defaultdict


_GROUP_RE = re.compile(r"^(?P<prefix>.+?)-(?P<macros>\d+)-c")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Split CircuitNet DRC train CSV into train/val using group-aware splitting "
            "based on the sample naming convention."
        )
    )
    p.add_argument(
        "--input_csv",
        default=osp.join("..", "files", "train_N28.csv"),
        help="Path to input CSV (feature_rel,label_rel).",
    )
    p.add_argument(
        "--output_train_csv",
        default=osp.join("..", "files", "train_N28_train.csv"),
        help="Where to write the train split CSV.",
    )
    p.add_argument(
        "--output_val_csv",
        default=osp.join("..", "files", "train_N28_val.csv"),
        help="Where to write the val split CSV.",
    )
    p.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of groups assigned to validation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    p.add_argument(
        "--group_by",
        choices=["design", "design_macros"],
        default="design",
        help=(
            "Grouping key. 'design' keeps all variants of a base design together; "
            "'design_macros' groups by (design, #macros)."
        ),
    )
    return p.parse_args()


def _extract_stem(rel_path: str) -> str:
    """Extract sample stem from a path like 'feature/<stem>.npy'."""
    base = osp.basename(rel_path)
    stem, _ = osp.splitext(base)
    return stem


def _group_key_from_stem(stem: str, group_by: str) -> str:
    """
    Expected stem format (per repo):
      <id>-<Design name>-<#Macros>-c<Clock>-u<Util>-m<MacroPl>-p<Power>-f<Filler>
    Examples:
      2187-RISCY-b-1-c2-u0.8-m1-p8-f1
      5107-RISCY-FPU-a-2-c20-u0.7-m1-p2-f1

    We group by base design name (e.g. 'RISCY-b' or 'RISCY-FPU-a') or by (design, macros).
    """
    # Drop numeric id prefix
    parts = stem.split("-", 1)
    name = parts[1] if len(parts) == 2 else stem

    m = _GROUP_RE.match(name)
    if not m:
        # Fallback: if pattern doesn't match, keep full name as its own group.
        return name

    design = m.group("prefix")
    macros = m.group("macros")

    if group_by == "design_macros":
        return f"{design}-{macros}"
    return design


def read_pairs(csv_path: str):
    pairs = []
    with open(csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            feat, lab = line.split(",")
            pairs.append((feat, lab))
    return pairs


def write_pairs(csv_path: str, pairs):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w") as f:
        for feat, lab in pairs:
            f.write(f"{feat},{lab}\n")


def main():
    args = parse_args()

    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in (0,1)")

    pairs = read_pairs(args.input_csv)

    grouped = defaultdict(list)
    for feat, lab in pairs:
        stem = _extract_stem(feat)
        gk = _group_key_from_stem(stem, args.group_by)
        grouped[gk].append((feat, lab))

    group_keys = list(grouped.keys())
    rng = random.Random(args.seed)
    rng.shuffle(group_keys)

    # We split by groups (not by rows) to avoid leakage. With few groups, the
    # realized val_ratio may be coarse (e.g. 1/4 = 25%). We choose the number of
    # groups that gets closest to the requested val_ratio.
    target = len(group_keys) * args.val_ratio
    n_val_groups = max(1, min(len(group_keys) - 1, int(round(target))))
    if n_val_groups == 0:
        n_val_groups = 1
    candidates = [n_val_groups]
    if n_val_groups - 1 >= 1:
        candidates.append(n_val_groups - 1)
    if n_val_groups + 1 <= len(group_keys) - 1:
        candidates.append(n_val_groups + 1)
    n_val_groups = min(candidates, key=lambda n: abs((n / len(group_keys)) - args.val_ratio))

    val_groups = set(group_keys[:n_val_groups])

    train_pairs, val_pairs = [], []
    for gk, items in grouped.items():
        if gk in val_groups:
            val_pairs.extend(items)
        else:
            train_pairs.extend(items)

    write_pairs(args.output_train_csv, train_pairs)
    write_pairs(args.output_val_csv, val_pairs)

    print(
        "Split complete:\n"
        f"  input: {args.input_csv} ({len(pairs)} samples)\n"
        f"  groups: {len(group_keys)} (group_by={args.group_by})\n"
        f"  train: {args.output_train_csv} ({len(train_pairs)} samples)\n"
        f"  val:   {args.output_val_csv} ({len(val_pairs)} samples)\n"
        f"  val_groups: {len(val_groups)} (~{args.val_ratio:.2f})"
    )


if __name__ == "__main__":
    main()

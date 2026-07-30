
"""
Prepare SWMH for the experiment.

Decisions encoded here, each with a reason:

1. Drop `self.offmychest`.
   Kermani et al. (2025) evaluate SWMH as 4-class (depression, anxiety,
   bipolar, suicidal ideation). Matching their formulation is the reason we
   chose this dataset: it gives us a published anxiety-class F1 (0.86
   fine-tuned, 0.74 zero-shot, LLaMA-3-8B) to anchor against. Keeping
   offmychest would make our numbers non-comparable.
   The 5-class version is written out too, as a secondary robustness run.

2. Strip the `self.` prefix from labels.
   Cosmetic, but the raw form leaks the collection method into prompts.

3. Remove posts that appear in more than one split.
   ~0.1-0.3% of rows. Small, but train/test contamination is indefensible in
   a paper whose whole point is a controlled comparison. Removed from the
   later split, keeping the earlier one (train > val > test priority).

4. Fixed ordering and a saved manifest.
   Every model must see byte-identical splits. The manifest records row
   counts and a content hash so we can prove it later.

Usage:
    python src/prepare_data.py --in-dir data/raw --out-dir data/processed
"""

import argparse
import hashlib
import json
import os

import pandas as pd

DROP_FOR_4CLASS = "offmychest"
SPLIT_PRIORITY = ["train", "val", "test"]  # earlier wins on duplicates


def load(in_dir):
    frames = {}
    for split in SPLIT_PRIORITY:
        path = os.path.join(in_dir, f"{split}.csv")
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}")
        df = pd.read_csv(path)
        df["label"] = df["label"].astype(str).str.replace("self.", "", regex=False)
        df["split"] = split
        frames[split] = df
    return frames


def drop_cross_split_duplicates(frames):
    seen = set()
    removed = {}
    for split in SPLIT_PRIORITY:
        df = frames[split]
        key = df["text"].astype(str).str.strip()
        dup = key.isin(seen)
        removed[split] = int(dup.sum())
        frames[split] = df[~dup].reset_index(drop=True)
        seen |= set(key[~dup])
    return frames, removed


def content_hash(df):
    joined = "\n".join(df["text"].astype(str) + "\t" + df["label"].astype(str))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def write_variant(frames, out_dir, name, drop_class=None):
    target = os.path.join(out_dir, name)
    os.makedirs(target, exist_ok=True)
    manifest = {"variant": name, "dropped_class": drop_class, "splits": {}}

    for split in SPLIT_PRIORITY:
        df = frames[split].copy()
        if drop_class:
            df = df[df["label"].str.lower() != drop_class].reset_index(drop=True)
        df = df[["text", "label"]]
        df.to_csv(os.path.join(target, f"{split}.csv"), index=False)

        counts = df["label"].value_counts().to_dict()
        manifest["splits"][split] = {
            "n": len(df),
            "sha256_16": content_hash(df),
            "class_counts": counts,
        }
        pct = {k: round(v / len(df) * 100, 2) for k, v in counts.items()}
        print(f"  {name}/{split}.csv  n={len(df):>6}  {pct}")

    with open(os.path.join(target, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    frames = load(args.in_dir)
    print("Loaded raw splits:",
          {k: len(v) for k, v in frames.items()})

    frames, removed = drop_cross_split_duplicates(frames)
    print("Removed cross-split duplicates:", removed)

    print("\nPRIMARY (4-class, matches Kermani et al. 2025):")
    write_variant(frames, args.out_dir, "4class", drop_class=DROP_FOR_4CLASS)

    print("\nSECONDARY (5-class, includes offmychest as non-clinical control):")
    write_variant(frames, args.out_dir, "5class", drop_class=None)

    print("\nDone. Use data/processed/4class/ as the primary experiment.")
    print("Every model condition must read from the same directory.")


if __name__ == "__main__":
    main()


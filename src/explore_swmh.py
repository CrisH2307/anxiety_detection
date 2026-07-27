"""
SWMH dataset validation and exploratory analysis.

Purpose: answer the open questions that block the Experimental Design section.
  Q1. What are the actual column names and label values?
  Q2. What is the class distribution? (Xue et al. 2026 show these Reddit
      corpora are severely imbalanced; we need our own numbers, not theirs.)
  Q3. Does the dataset ship with train/val/test splits, and are they clean?
  Q4. How long are posts? (Determines truncation for BERT's 512-token limit
      and cost for GPT-4o zero-shot passes.)
  Q5. Label leakage: do posts literally contain their class name?
      This matters because SWMH labels derive from subreddit membership.
      If "anxiety" appears verbatim in anxiety-class posts, a classifier can
      win by keyword matching rather than by detecting anxiety-related
      language. Zhu et al. (2025) handle this by building a keyword-removed
      variant (DAUR_PRE). We need to know the size of the problem first.

Usage:
    python src/explore_swmh.py --data-dir data/raw
"""

import argparse
import glob
import os
import re

import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 50)

# Class names we expect based on the SWMH dataset card and Kermani et al. (2025).
# Verify against actual values; do not assume.
EXPECTED_CLASS_HINTS = [
    "anxiety", "depression", "bipolar", "suicide", "suicidewatch", "offmychest",
]


def find_csvs(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not paths:
        raise SystemExit(
            f"No CSVs found under {data_dir!r}.\n"
            "Download SWMH from HuggingFace (AIMH/SWMH) after your access "
            "request is approved, and place the files in data/raw/."
        )
    return paths


def q1_schema(frames):
    print("\n" + "=" * 70)
    print("Q1. SCHEMA")
    print("=" * 70)
    for name, df in frames.items():
        print(f"\n--- {name} --- shape={df.shape}")
        print("columns:", list(df.columns))
        print(df.dtypes.to_string())
        print("\nfirst row:")
        with pd.option_context("display.max_colwidth", 200):
            print(df.head(1).T.to_string())


def guess_columns(df):
    """Heuristically identify the text and label columns."""
    text_col = label_col = None
    for c in df.columns:
        lc = c.lower()
        if text_col is None and lc in ("text", "body", "selftext", "post", "content"):
            text_col = c
        if label_col is None and lc in ("label", "class", "target", "subreddit", "y"):
            label_col = c
    # Fallback: longest average string column is probably the text
    if text_col is None:
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if obj_cols:
            text_col = max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())
    # Fallback: lowest-cardinality column is probably the label
    if label_col is None:
        cands = [c for c in df.columns if c != text_col and df[c].nunique() <= 20]
        if cands:
            label_col = min(cands, key=lambda c: df[c].nunique())
    return text_col, label_col


def q2_class_distribution(frames, text_col, label_col):
    print("\n" + "=" * 70)
    print(f"Q2. CLASS DISTRIBUTION  (label column: {label_col!r})")
    print("=" * 70)
    for name, df in frames.items():
        if label_col not in df.columns:
            print(f"\n--- {name} --- label column not present, skipping")
            continue
        counts = df[label_col].value_counts(dropna=False)
        pct = (counts / len(df) * 100).round(2)
        out = pd.DataFrame({"n": counts, "pct": pct})
        print(f"\n--- {name} --- total={len(df)}")
        print(out.to_string())
        imbalance = counts.max() / counts.min() if counts.min() > 0 else float("inf")
        print(f"imbalance ratio (max/min): {imbalance:.2f}x")
        if imbalance > 3:
            print("  NOTE: >3x imbalance. Report macro-F1, not accuracy.")
            print("  Cite Xue et al. (2026) on class imbalance in these corpora.")


def q3_splits(frames, text_col):
    print("\n" + "=" * 70)
    print("Q3. SPLITS AND LEAKAGE BETWEEN SPLITS")
    print("=" * 70)
    print("files found:", list(frames.keys()))
    if len(frames) < 2:
        print("Only one file. You will need to create your own split.")
        print("Use a fixed seed and save the split so every model sees the same data.")
        return
    # Check for duplicate posts appearing in more than one split
    seen = {}
    for name, df in frames.items():
        if text_col not in df.columns:
            continue
        seen[name] = set(df[text_col].astype(str).str.strip())
    names = list(seen.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = seen[names[i]] & seen[names[j]]
            flag = "  <-- LEAKAGE, investigate" if overlap else ""
            print(f"{names[i]} vs {names[j]}: {len(overlap)} shared posts{flag}")


def q4_lengths(frames, text_col):
    print("\n" + "=" * 70)
    print(f"Q4. POST LENGTH  (text column: {text_col!r})")
    print("=" * 70)
    for name, df in frames.items():
        if text_col not in df.columns:
            continue
        words = df[text_col].astype(str).str.split().str.len()
        # ~1.3 tokens per word is a rough English estimate
        print(f"\n--- {name} ---")
        print(words.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1).to_string())
        over512 = (words * 1.3 > 512).mean() * 100
        print(f"est. share exceeding 512 tokens (BERT limit): {over512:.1f}%")
        est_tokens = (words * 1.3).sum()
        print(f"est. total tokens if sent to a frontier API: {est_tokens:,.0f}")


def q5_label_leakage(frames, text_col, label_col):
    print("\n" + "=" * 70)
    print("Q5. LABEL-NAME LEAKAGE IN POST TEXT")
    print("=" * 70)
    print("If a post literally contains its own class name, a model can win by")
    print("keyword matching. Measure this now; it shapes your robustness check.\n")
    for name, df in frames.items():
        if text_col not in df.columns or label_col not in df.columns:
            continue
        print(f"--- {name} ---")
        for cls in sorted(df[label_col].dropna().unique()):
            sub = df[df[label_col] == cls]
            clean_cls = str(cls).lower().replace("self.", "").replace("_", " ")
            term = re.escape(clean_cls)
            hits = sub[text_col].astype(str).str.lower().str.contains(term, regex=True)
            print(f"  {str(cls):<20} {hits.mean() * 100:5.1f}% contain '{cls}'")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--text-col", default=None, help="override text column")
    ap.add_argument("--label-col", default=None, help="override label column")
    args = ap.parse_args()

    paths = find_csvs(args.data_dir)
    frames = {os.path.basename(p): pd.read_csv(p) for p in paths}

    q1_schema(frames)

    first = next(iter(frames.values()))
    text_col = args.text_col or guess_columns(first)[0]
    label_col = args.label_col or guess_columns(first)[1]
    print(f"\n>>> Using text_col={text_col!r}, label_col={label_col!r}")
    print(">>> If wrong, rerun with --text-col and --label-col.")

    q2_class_distribution(frames, text_col, label_col)
    q3_splits(frames, text_col)
    q4_lengths(frames, text_col)
    q5_label_leakage(frames, text_col, label_col)

    print("\n" + "=" * 70)
    print("REPORT THESE BACK: label values, class counts, split sizes,")
    print("length percentiles, and the leakage percentages.")
    print("=" * 70)


if __name__ == "__main__":
    main()

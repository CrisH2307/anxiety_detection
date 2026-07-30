
"""
Keyword leakage check.

Measures how often a post contains a term naming its own class. Uses the
SAME lexicon the training scripts strip with (labels.STRIP_TERMS), so the
reported percentage describes exactly what redaction removes. Previously
this file kept a private CLASS_TERMS copy, which would silently diverge the
moment the strip lexicon changed.

Usage:
    python src/check_leakage.py --data-dir data/processed/4class
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import STRIP_TERMS


def normalize(label):
    return str(label).lower().replace("self.", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed/4class")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No CSVs in {args.data_dir!r}")

    for path in paths:
        df = pd.read_csv(path)
        df["_cls"] = df[args.label_col].map(normalize)
        text = df[args.text_col].astype(str).str.lower()

        print("\n" + "=" * 72)
        print(f"{os.path.basename(path)}   n={len(df)}")
        print("=" * 72)
        print(f"{'class':<15}{'own-term %':>12}{'any-term %':>12}   top term")
        print("-" * 72)

        for cls in sorted(df["_cls"].unique()):
            mask = df["_cls"] == cls
            sub_text = text[mask]
            terms = STRIP_TERMS.get(cls, [cls])

            own = pd.Series(False, index=sub_text.index)
            per_term = {}
            for t in terms:
                pat = r"\b" + re.escape(t) + r"\w*"
                hit = sub_text.str.contains(pat, regex=True, na=False)
                per_term[t] = hit.mean() * 100
                own = own | hit

            any_hit = pd.Series(False, index=sub_text.index)
            for other_terms in STRIP_TERMS.values():
                for t in other_terms:
                    pat = r"\b" + re.escape(t) + r"\w*"
                    any_hit = any_hit | sub_text.str.contains(pat, regex=True, na=False)

            top = max(per_term, key=per_term.get)
            print(f"{cls:<15}{own.mean()*100:>11.1f}%{any_hit.mean()*100:>11.1f}%"
                  f"   {top} ({per_term[top]:.1f}%)")

    print("\nNote: percentages are computed WITHIN each class independently")
    print("and do not sum to 100. 'any-term' counts posts containing a term")
    print("belonging to any class, capturing cross-class comorbidity language.")


if __name__ == "__main__":
    main()
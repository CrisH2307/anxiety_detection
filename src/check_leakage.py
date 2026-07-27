"""
Keyword leakage check (corrected).

The earlier version searched for the raw label string ("self.Anxiety"), which
never matches. This strips the "self." prefix and also checks morphological
variants, since a post in r/Anxiety is likely to say "anxious" rather than
"anxiety".

Why this matters: SWMH labels come from subreddit membership. If posts contain
their own condition name, a classifier can score well by keyword matching
instead of by detecting the underlying language. Zhu et al. (2025) address this
by building a keyword-removed variant (DAUR_PRE). We need our own numbers
before deciding whether to do the same.

Usage:
    python src/check_leakage.py --data-dir data/raw
"""

import argparse
import glob
import os
import re

import pandas as pd

# Surface forms to test per class. Word-boundary matched, case-insensitive.
CLASS_TERMS = {
    "anxiety": ["anxiety", "anxious", "anxieties"],
    "depression": ["depression", "depressed", "depressive"],
    "bipolar": ["bipolar", "manic", "mania", "hypomania"],
    "suicidewatch": ["suicide", "suicidal", "kill myself", "end my life"],
    "offmychest": ["off my chest", "vent", "venting"],
}


def normalize(label):
    return str(label).lower().replace("self.", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
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
            terms = CLASS_TERMS.get(cls, [cls])

            # Does the post contain a term for its OWN class?
            own = pd.Series(False, index=sub_text.index)
            per_term = {}
            for t in terms:
                pat = r"\b" + re.escape(t) + r"\w*"
                hit = sub_text.str.contains(pat, regex=True, na=False)
                per_term[t] = hit.mean() * 100
                own = own | hit

            # Does the post contain a term for ANY class? (cross-class noise)
            any_hit = pd.Series(False, index=sub_text.index)
            for other_terms in CLASS_TERMS.values():
                for t in other_terms:
                    pat = r"\b" + re.escape(t) + r"\w*"
                    any_hit = any_hit | sub_text.str.contains(pat, regex=True, na=False)

            top = max(per_term, key=per_term.get)
            print(
                f"{cls:<15}{own.mean() * 100:>11.1f}%{any_hit.mean() * 100:>11.1f}%"
                f"   {top} ({per_term[top]:.1f}%)"
            )

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("own-term % under ~25%: leakage is mild. Note it, no stripped variant needed.")
    print("own-term % 25-50%:     borderline. Run the stripped variant as a")
    print("                       robustness check on the anxiety class at minimum.")
    print("own-term % over ~50%:  substantial. A keyword-stripped variant becomes")
    print("                       a required condition, not optional.")


if __name__ == "__main__":
    main()

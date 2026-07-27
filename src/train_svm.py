"""
Classical baseline: TF-IDF (word 1-2 grams) + LinearSVC, for the 4-class
SWMH task. Runs entirely on CPU, no GPU needed, this is your Mac's job.

Why this design:

- TF-IDF + n-grams is the same family of approach Kallstenius et al.
  (2025) used to beat LLMs on mental-health classification (their 95%
  headline number), and it's what Shen & Rudzicz (2017) built the
  original anxiety-detection feature-engineering line on. Using it here
  keeps your classical condition comparable to both.

- LinearSVC, not a kernel SVM: kernel SVMs don't scale past a few
  thousand examples, LinearSVC does and is the standard choice for
  high-dimensional sparse text features like TF-IDF.

- class_weight="balanced": SWMH is 2.4x imbalanced. Without this, the
  classifier can lean on the majority class (depression) more than it
  should. Same imbalance-awareness reasoning as the LLM script's use of
  Xue et al. (2026) on this exact issue.

- Uses the FULL 29,558-example train set, unlike the LLM script's 8,000
  subsample. SVM training here takes seconds, not hours, so there's no
  reason to subsample, and using more data is a fair, defensible choice
  since it's not compute-constrained the way fine-tuning is.

- Same STRIP_TERMS and full/stripped variant logic as
  train_llm_qlora.py and check_leakage.py, so all three scripts agree
  on what "stripped" means.

- Writes JSON in the exact same schema as train_llm_qlora.py's output,
  so the notebook plotting cells work unmodified across every condition.

Usage:
    python src/train_svm.py --variant full
    python src/train_svm.py --variant stripped
"""

import argparse
import json
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import LinearSVC

LABELS = ["depression", "SuicideWatch", "Anxiety", "bipolar"]
SEED = 42

# Must match check_leakage.py and train_llm_qlora.py's STRIP_TERMS.
STRIP_TERMS = {
    "anxiety": ["anxiety", "anxious", "anxieties"],
    "depression": ["depression", "depressed", "depressive"],
    "bipolar": ["bipolar", "manic", "mania", "hypomania"],
    "suicidewatch": ["suicide", "suicidal", "kill myself", "end my life"],
}


def strip_keywords(text):
    out = text
    for terms in STRIP_TERMS.values():
        for t in terms:
            out = re.sub(r"\b" + re.escape(t) + r"\w*", "[REDACTED]", out, flags=re.I)
    return out


def load_split(path, variant):
    df = pd.read_csv(path)
    if variant == "stripped":
        df["text"] = df["text"].astype(str).apply(strip_keywords)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "stripped"], required=True)
    ap.add_argument("--data-dir", default="data/processed/4class")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--max-features", type=int, default=50000)
    args = ap.parse_args()

    print(f"Variant: {args.variant}")

    train_df = load_split(os.path.join(args.data_dir, "train.csv"), args.variant)
    test_df = load_split(os.path.join(args.data_dir, "test.csv"), args.variant)
    print(f"train={len(train_df)}  test={len(test_df)}")
    print("train class balance:\n", train_df["label"].value_counts(normalize=True).round(3))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=args.max_features,
        min_df=2,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_df["text"].astype(str))
    X_test = vectorizer.transform(test_df["text"].astype(str))

    clf = LinearSVC(class_weight="balanced", random_state=SEED, max_iter=5000)
    clf.fit(X_train, train_df["label"])

    preds = clf.predict(X_test)
    true = test_df["label"].tolist()

    report = classification_report(true, preds, labels=LABELS, output_dict=True, zero_division=0)
    macro_f1 = f1_score(true, preds, labels=LABELS, average="macro", zero_division=0)

    print(classification_report(true, preds, labels=LABELS, zero_division=0))
    print(f"macro-F1: {macro_f1:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_svm_{args.variant}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": "TF-IDF (1-2gram) + LinearSVC",
            "variant": args.variant,
            "macro_f1": macro_f1,
            "anxiety_f1": report.get("Anxiety", {}).get("f1-score"),
            "unparseable_rate": 0.0,  # SVM always predicts a valid class
            "full_report": report,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

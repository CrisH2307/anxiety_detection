"""
Shared constants for every script in this project.

Why this file exists: LABELS and STRIP_TERMS were previously duplicated in
train_llm_qlora.py, train_encoder.py, train_svm.py and check_leakage.py.
Four copies drift. If the strip lexicon differs by even one term between
the leakage measurement and the training script, the reported leakage
percentage no longer describes the text the model actually saw.

Import from here everywhere:
    from labels import LABELS, STRIP_TERMS, strip_keywords
"""

import re

# Task labels. Order is fixed and used for report column ordering.
LABELS = ["depression", "SuicideWatch", "Anxiety", "bipolar"]

# Terms redacted in the "stripped" variant.
#
# SCOPE RULE, stated explicitly because it is a design decision, not an
# oversight: this lexicon targets CONDITION-NAMING vocabulary only, the
# words a person uses to name their diagnosis. It deliberately excludes
# SYMPTOM vocabulary ("worried", "restless", "can't sleep", "racing
# thoughts"), because symptom language is exactly what we want a model to
# use legitimately. Stripping symptoms too would not test lexical
# dependence, it would test whether the model can classify from nothing.
#
# Expanded from the original four-term-per-class version after reviewer
# feedback that the lexicon was too narrow.
STRIP_TERMS = {
    "anxiety": [
        "anxiety", "anxious", "anxieties", "anxiousness",
        "gad", "generalized anxiety", "generalised anxiety",
        "social anxiety", "panic disorder", "panic attack",
        "agoraphobia", "phobia",
    ],
    "depression": [
        "depression", "depressed", "depressive", "depressing",
        "mdd", "major depressive", "dysthymia", "dysthymic",
    ],
    "bipolar": [
        "bipolar", "manic", "mania", "hypomania", "hypomanic",
        "bipolar disorder", "bipolar 1", "bipolar 2",
        "bipolar i", "bipolar ii", "manic depression",
    ],
    "suicidewatch": [
        "suicide", "suicidal", "suicidality",
        "kill myself", "killing myself", "end my life",
        "ending my life", "take my own life", "kms",
    ],
}

# Longest-first so multi-word terms match before their single-word parts
# ("panic attack" before "panic", "manic depression" before "manic").
_ALL_TERMS = sorted(
    {t for terms in STRIP_TERMS.values() for t in terms},
    key=len,
    reverse=True,
)

_PATTERNS = [
    re.compile(r"\b" + re.escape(t) + r"\w*", flags=re.IGNORECASE)
    for t in _ALL_TERMS
]


def strip_keywords(text, mode="delete"):
    """Redact condition-naming terms from a post.

    mode controls what replaces the matched span. This matters more than it
    looks:

      "delete"  -> removed entirely, leaving no trace.
                   RECOMMENDED DEFAULT. The earlier "[REDACTED]" approach
                   was a methodological bug: the marker appeared once per
                   removed term, so its COUNT correlated with the class. A
                   model could learn "three markers means anxiety" without
                   ever seeing the word, which is precisely the shortcut the
                   stripped condition exists to eliminate.

      "neutral" -> replaced with a common, class-neutral filler word.
                   Preserves sentence length and rough grammaticality.
                   Still inserts a marker, but one that carries no
                   class-specific count signal beyond what deletion leaves.

      "mask"    -> replaced with "[REDACTED]". Reproduces the original
                   (flawed) behaviour. Retained ONLY so the earlier results
                   can be regenerated for comparison. Do not use for new
                   experiments.

    Running "delete" and "neutral" and comparing gives a cheap sensitivity
    check on whether the redaction method itself drives the result.
    """
    if mode not in ("delete", "neutral", "mask"):
        raise ValueError(f"unknown redaction mode: {mode!r}")

    replacement = {"delete": " ", "neutral": " something ", "mask": "[REDACTED]"}[mode]

    out = text
    for pat in _PATTERNS:
        out = pat.sub(replacement, out)

    # Deletion leaves double spaces; normalise so whitespace itself does not
    # become a weak signal of how many terms were removed.
    out = re.sub(r"\s+", " ", out).strip()
    return out


def count_stripped(text):
    """How many terms would be redacted. Diagnostic only, never a feature."""
    return sum(len(pat.findall(text)) for pat in _PATTERNS)
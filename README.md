# Anxiety Detection: Fine-Tuned Small LLMs vs. Frontier Zero-Shot

Course project (Project 1, Research Paper). Cris covers data + models;
teammate covers writing.

## Research question

On anxiety-labeled Reddit posts, does an instruction fine-tuned small
open-weight LLM achieve classification performance equal to or better than a
frontier-scale LLM used zero-shot, and how do both compare to classical and
encoder baselines?

**H1** (primary): fine-tuned small open-weight LLM achieves macro-F1 >= frontier
LLM zero-shot, under identical splits, prompts, and decoding settings.

**H2** (secondary, exploratory): models differ in which tokens drive predictions.

## Model conditions (5)

| # | Condition | Model | Adaptation |
|---|-----------|-------|------------|
| 1 | Classical baseline | SVM + n-gram/lexicon features | trained |
| 2 | Encoder | BERT-base | fine-tuned |
| 3 | Encoder (domain) | MentalBERT | fine-tuned |
| 4 | Frontier LLM | GPT-4o | zero-shot (few-shot secondary) |
| 5 | Small open-weight LLM | Llama-3-8B or Qwen2.5-7B | LoRA fine-tuned |

Held constant across all conditions: dataset, splits, preprocessing, label
set. For LLM conditions also: prompt template, decoding temperature, output
constraints. Only model family and adaptation strategy vary.

## Dataset

SWMH (Reddit SuicideWatch and Mental Health Collection), 54,412 posts.
Gated on HuggingFace: <https://huggingface.co/datasets/AIMH/SWMH>
Requires institutional email and a data use agreement.

Citation: Ji, S., Li, X., Huang, Z., & Cambria, E. (2021). Suicidal ideation
and mental disorder detection with attentive relation networks. *Neural
Computing and Applications*.

**Data use terms**: research only, no redistribution, no attempt to identify
users, store on password-protected machines. Keep `data/` out of git.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the SWMH CSVs in `data/raw/`.

## Step 1: validate the dataset (do this first)

```bash
python src/explore_swmh.py --data-dir data/raw
```

This answers five questions that block the Experimental Design section:

1. Actual column names and label values
2. Class distribution and imbalance ratio
3. Whether splits ship with the data, and whether posts leak across them
4. Post length distribution (BERT 512-token truncation, frontier API cost)
5. Label-name leakage: do anxiety posts literally contain the word "anxiety"

Question 5 matters most. SWMH labels come from subreddit membership, so if
posts contain their own class name, a classifier can win by keyword matching
rather than by detecting anxiety-related language. Zhu et al. (2025) address
this by building a keyword-removed variant (DAUR_PRE). If leakage is high, we
run every condition twice: full text, and keyword-stripped.

If the auto-detected columns are wrong:

```bash
python src/explore_swmh.py --data-dir data/raw --text-col TEXT --label-col LABEL
```

## Task formulation

Multi-class (4-class: anxiety / depression / bipolar / suicidal ideation),
reporting **per-class anxiety F1** plus macro-F1.

Reason: Kermani et al. (2025) report anxiety-class F1 of 0.86 (fine-tuned) and
0.74 (zero-shot) on SWMH with LLaMA-3-8B. Matching their formulation gives us
a published baseline to anchor against. Collapsing to binary would inflate
scores and break that comparison.

## Metrics

Macro-F1 (primary), per-class anxiety F1, precision, recall, AUC.
Report macro-F1 rather than accuracy given class imbalance (cf. Xue et al. 2026).

Also track, for the frontier condition: **refusal rate**. Lee et al. (2026)
found frontier models refusing on affectively sensitive content. Refusals must
be counted separately, not scored as misclassifications.

## Next steps

- [ ] Run `explore_swmh.py`, report results
- [ ] Decide keyword-stripped variant based on Q5
- [ ] Freeze splits with a fixed seed, save to `data/processed/`
- [ ] Confirm compute available for LoRA fine-tuning
- [ ] Build SVM baseline
- [ ] Fine-tune BERT / MentalBERT
- [ ] GPT-4o zero-shot pass
- [ ] LoRA fine-tune small open-weight model
- [ ] SHAP / LIME cross-model comparison
# anxiety_detection

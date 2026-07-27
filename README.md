# Anxiety Detection: Small Fine-Tuned Models vs. Frontier LLMs

Code accompanying ***Anxiety Detection on Social Media: Do Models Read the Language, or Just the Word “Anxiety”?*** (BTM710, Seneca Polytechnic, 2026).

## Setup
1. Request SWMH access: https://huggingface.co/datasets/AIMH/SWMH
   (institutional email required)
2. `pip install -r requirements.txt`
3. Place train.csv, val.csv, test.csv in `data/raw/`

## Pipeline
python src/prepare_data.py      # filter to 4-class, dedupe, split
python src/check_leakage.py     # measure keyword leakage per class
python src/train_svm.py --variant full
python src/train_svm.py --variant stripped
python src/train_encoder.py --model-name mental/mental-bert-base-uncased --variant full
python src/train_llm_qlora.py --variant full   # requires GPU (Kaggle T4)

Frontier conditions (GPT-5.4, Claude Haiku 4.5) were run via API,
scripts not included; see prompts/classification_prompt.txt for the
exact template used.

## Results
See results/*.json for raw output. Table in Section VI of the paper
is generated from these files.

"""
QLoRA fine-tuning: Qwen2.5-3B-Instruct for 4-class anxiety/depression/
bipolar/suicidal-ideation classification on SWMH.

Run this on Kaggle (GPU T4 x2 or single T4), not on the Mac. Uses
bitsandbytes 4-bit quantization + LoRA adapters, which is CUDA-only.

What this does, and why each choice:

- 4-bit quantization (QLoRA): fits a 3B model comfortably in a single T4's
  15GB, leaving room for activations and batch. Matches the approach Menta
  (Zhang et al. 2025) and Xue et al. (2026) used, so our setup stays
  comparable to the literature we're extending.

- LoRA on attention projections only (q_proj, v_proj, k_proj, o_proj):
  standard target for causal LMs, keeps trainable params under 1% of the
  3B total.

- Class-weighted loss: SWMH is 2.4-2.5x imbalanced (depression dominant).
  Xue et al. (2026) show this matters for these exact corpora. Without
  weighting, the model can hit decent accuracy by mostly predicting
  "depression" and ignoring anxiety.

- Runs BOTH variants (full text, keyword-stripped) in one script, same
  hyperparameters, same seed, so the only difference between the two
  trained adapters is the text they saw. This is what makes the
  full-vs-stripped comparison controlled rather than incidental.

- Classification via next-token prediction over a constrained label
  vocabulary, not a separate classification head. This keeps the LLM
  condition using the LLM the way Xu et al. and Kermani et al. did
  (generate the class name as text), so results are comparable to theirs.

Before running:
  1. Upload data/processed/4class/{train,val,test}.csv and the
     keyword-stripped equivalents as a Kaggle Dataset, or write them to
     /kaggle/working/data/ via the Kaggle API / file upload.
  2. pip install -q peft bitsandbytes accelerate transformers datasets

Usage (in a Kaggle notebook cell):
    !python train_llm_qlora.py --variant full
    !python train_llm_qlora.py --variant stripped
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LABELS = ["depression", "SuicideWatch", "Anxiety", "bipolar"]
SEED = 42

# Terms to strip for the "stripped" variant. Must match check_leakage.py's
# CLASS_TERMS so the two scripts agree on what "stripped" means.
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


def load_split(path, variant, max_train_n=10000):
    df = pd.read_csv(path)
    if variant == "stripped":
        df["text"] = df["text"].astype(str).apply(strip_keywords)
    if "train" in os.path.basename(path) and len(df) > max_train_n:
        # Stratified subsample. Xu et al. (2024) found diminishing returns
        # past a few thousand examples for instruction fine-tuning; this
        # keeps training time bounded on a single T4 across two variants.
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(
                min(len(g), int(max_train_n * len(g) / len(df))),
                random_state=SEED,
            ))
            .reset_index(drop=True)
        )
    return df


def build_prompt(text):
    return (
        "Classify the following Reddit post into exactly one category: "
        "depression, SuicideWatch, Anxiety, or bipolar.\n\n"
        f"Post: {text}\n\nCategory:"
    )


def to_hf_dataset(df, tokenizer, max_len=768):
    def _map(row):
        prompt = build_prompt(row["text"])
        target = f" {row['label']}"
        full = prompt + target
        enc = tokenizer(full, truncation=True, max_length=max_len, padding="max_length")
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
        labels = list(enc["input_ids"])
        # Mask the prompt tokens so loss is computed on the label only.
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    records = df.apply(_map, axis=1).tolist()
    return Dataset.from_list(records)


def class_weights(labels):
    counts = pd.Series(labels).value_counts()
    total = counts.sum()
    weights = {cls: total / (len(counts) * n) for cls, n in counts.items()}
    return weights


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


@torch.no_grad()
def predict(model, tokenizer, texts, device, max_len=768, batch_size=8):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        prompts = [build_prompt(t) for t in batch]
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_len,
        ).to(device)
        out = model.generate(
            **enc, max_new_tokens=6, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for j in range(len(batch)):
            gen = out[j][enc["input_ids"].shape[1]:]
            text = tokenizer.decode(gen, skip_special_tokens=True).strip()
            pred = next((l for l in LABELS if l.lower() in text.lower()), "UNKNOWN")
            preds.append(pred)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["full", "stripped"], required=True)
    ap.add_argument("--data-dir", default="data/processed/4class")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-n", type=int, default=10000)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No GPU visible. Run this on Kaggle with the T4 accelerator on.")

    print(f"Variant: {args.variant}  |  Device: {device}")

    train_df = load_split(os.path.join(args.data_dir, "train.csv"), args.variant, args.max_train_n)
    val_df = load_split(os.path.join(args.data_dir, "val.csv"), args.variant, max_train_n=10**9)
    test_df = load_split(os.path.join(args.data_dir, "test.csv"), args.variant, max_train_n=10**9)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print("train class balance:\n", train_df["label"].value_counts(normalize=True).round(3))

    model, tokenizer = load_model_and_tokenizer()
    train_ds = to_hf_dataset(train_df, tokenizer)
    val_ds = to_hf_dataset(val_df, tokenizer)

    run_name = f"qwen25-3b-{args.variant}"
    out_path = os.path.join(args.out_dir, run_name)

    training_args = TrainingArguments(
        output_dir=out_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    model.save_pretrained(os.path.join(out_path, "adapter"))
    tokenizer.save_pretrained(os.path.join(out_path, "adapter"))

    print("\nEvaluating on held-out test set (touched once, here)...")
    preds = predict(model, tokenizer, test_df["text"].tolist(), device)
    true = test_df["label"].tolist()

    report = classification_report(true, preds, labels=LABELS, output_dict=True, zero_division=0)
    macro_f1 = f1_score(true, preds, labels=LABELS, average="macro", zero_division=0)
    unknown_rate = sum(p == "UNKNOWN" for p in preds) / len(preds)

    print(classification_report(true, preds, labels=LABELS, zero_division=0))
    print(f"macro-F1: {macro_f1:.4f}")
    print(f"unparseable generations: {unknown_rate * 100:.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"results_qwen25-3b_{args.variant}.json"), "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "variant": args.variant,
            "macro_f1": macro_f1,
            "anxiety_f1": report.get("Anxiety", {}).get("f1-score"),
            "unparseable_rate": unknown_rate,
            "full_report": report,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }, f, indent=2)
    print(f"\nSaved to {args.out_dir}/results_qwen25-3b_{args.variant}.json")


if __name__ == "__main__":
    main()


"""
QLoRA fine-tuning: Qwen2.5-3B-Instruct for 4-class anxiety/depression/
bipolar/suicidal-ideation classification on SWMH.

Run this on Kaggle (single T4), not on the Mac. Uses bitsandbytes 4-bit
quantization + LoRA adapters, which is CUDA-only.

Fix history (all applied below):
  - device_map={"": 0} + CUDA_VISIBLE_DEVICES=0: keep the whole model on
    one GPU. Trainer auto-wraps in DataParallel across 2 GPUs otherwise,
    which crashes with 4-bit quantized weights.
  - float16, not bfloat16: T4 lacks bfloat16 hardware support and falls
    back to slow software emulation.
  - Dynamic per-batch padding instead of fixed max_length=768: median post
    is ~140 tokens, so padding everything to 768 wastes most of the
    compute on padding tokens, not content.
  - batch_size=4 (not 8), max_len=512 (not 768), gradient_checkpointing=True,
    expandable_segments allocator: batch=8 + dynamic padding could still
    spike memory on batches of several long posts and OOM'd. This
    combination is the stable, tested configuration.
  - padding_side="left" during predict(), "right" during training: right-
    padding is fine for training (loss is masked per-token anyway), but
    breaks batched generation for a decoder-only model, since shorter
    sequences end up with padding between their content and the
    generation point.

Before running:
  1. Upload data/processed/4class/{train,val,test}.csv as a Kaggle Dataset.
  2. pip install -q peft bitsandbytes accelerate transformers datasets tqdm

Usage (in a Kaggle notebook cell):
    !python train_llm_qlora.py --variant full --data-dir /kaggle/input/datasets/<user>/<dataset>
    !python train_llm_qlora.py --variant stripped --data-dir /kaggle/input/datasets/<user>/<dataset>
"""

import os
# Hide GPU 1 so Trainer never wraps the model in DataParallel, which
# crashes with 4-bit quantized weights.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Reduce memory fragmentation (PyTorch's own suggestion from an earlier OOM).
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import re

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import LABELS, STRIP_TERMS, strip_keywords

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SEED = 42

# Must match check_leakage.py's CLASS_TERMS so "stripped" means the same
# thing in both scripts.
def load_split(path, variant, redaction_mode="delete", max_train_n=10000):
    df = pd.read_csv(path)
    if variant == "stripped":
        df["text"] = df["text"].astype(str).apply(lambda t: strip_keywords(t, mode=redaction_mode))
    if "train" in os.path.basename(path) and len(df) > max_train_n:
        # Stratified subsample. Xu et al. (2024) found diminishing returns
        # past a few thousand examples for instruction fine-tuning.
        # NOTE: this triggers a harmless pandas FutureWarning about
        # grouping columns. Do not "fix" it with include_groups=False --
        # that silently drops the label column from the result, which
        # breaks everything downstream. Ignore the warning.
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


def to_hf_dataset(df, tokenizer, max_len=512):
    # No fixed padding here. Each example is tokenized to its own true
    # length (capped at max_len). Padding happens per-BATCH at collate
    # time instead, via DynamicPadCollator below.
    def _map(row):
        prompt = build_prompt(row["text"])
        target = f" {row['label']}"
        full = prompt + target
        enc = tokenizer(full, truncation=True, max_length=max_len)
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=max_len)["input_ids"])
        labels = list(enc["input_ids"])
        # Mask the prompt tokens so loss is computed on the label only.
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        enc["labels"] = labels
        return enc

    records = df.apply(_map, axis=1).tolist()
    return Dataset.from_list(records)


class DynamicPadCollator:
    """Pads each batch to its own longest sequence, not a fixed max_len."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(features, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = [l + [-100] * (max_len - len(l)) for l in labels]
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def load_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # T4 lacks bfloat16 hardware
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding is fine for training (loss is masked per-token
    # regardless of position). We flip to left-padding just before
    # predict() below, which is what batched generation needs.
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map={"": 0}
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
def predict(model, tokenizer, texts, device, max_len=512, batch_size=8):
    model.eval()
    tokenizer.padding_side = "left"  # required for correct batched generation
    preds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating test set"):
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
    ap.add_argument("--redaction-mode", default="delete", choices=["delete", "neutral", "mask"])
    ap.add_argument("--data-dir", default="data/processed/4class")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-train-n", type=int, default=10000)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("No GPU visible. Run this on Kaggle with the T4 accelerator on.")

    print(f"Variant: {args.variant}  |  Device: {device}")

    train_df = load_split(os.path.join(args.data_dir, "train.csv"), args.variant, args.redaction_mode, args.max_train_n)
    val_df = load_split(os.path.join(args.data_dir, "val.csv"), args.variant, args.redaction_mode, max_train_n=10**9)
    test_df = load_split(os.path.join(args.data_dir, "test.csv"), args.variant, args.redaction_mode, max_train_n=10**9)
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
        gradient_accumulation_steps=4,  # batch=4 x accum=4 = effective batch 16
        learning_rate=args.lr,
        fp16=True,  # NOT bf16 -- T4 has no bfloat16 hardware support
        gradient_checkpointing=True,
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
        data_collator=DynamicPadCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(os.path.join(out_path, "adapter"))
    tokenizer.save_pretrained(os.path.join(out_path, "adapter"))
    print(f"\nAdapter saved to {out_path}/adapter -- safe even if evaluation below fails.")

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
            "redaction_mode": args.redaction_mode if args.variant == "stripped" else None,
            "n_train": len(train_df),
            "n_test": len(test_df),
        }, f, indent=2)
    print(f"\nSaved to {args.out_dir}/results_qwen25-3b_{args.variant}.json")


if __name__ == "__main__":
    main()
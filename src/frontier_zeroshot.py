"""
Frontier LLM zero-shot evaluation: GPT-4o, Claude Opus 4.8, Gemini 3.1 Pro.

Runs on your Mac. No GPU needed, this is API calls only.

Why three providers, not one:
  Xu et al. (2024) and Beyond Scale (Jia et al., 2025) each tested only
  GPT-family frontier models. Testing across three providers is a
  stronger robustness claim for the "frontier zero-shot" condition than
  any single paper in this literature made, and guards against one
  provider's quirks (e.g. safety refusals) being mistaken for a general
  finding about frontier-scale models.

  IMPORTANT: use each provider's FRONTIER tier, not its fast/cheap tier.
  gpt-4o-mini, Claude Haiku, and Gemini Flash are NOT frontier-scale --
  using them would repeat Kallstenius et al.'s (2025) mistake, whose
  "LLM underperforms" finding used GPT-4o-mini for both LLM conditions.

Refusal tracking (separate from misclassification):
  Lee et al. (2026) found frontier models refusing to classify Reddit
  posts describing personal distress, which depressed their reported
  accuracy for reasons unrelated to classification capability. A refusal
  scored as "wrong" conflates two different failures. This script
  detects refusals via keyword heuristics on the raw response and
  reports refusal_rate as its own field, separate from unparseable_rate
  (genuinely malformed output) and from ordinary misclassification.

Cost control:
  Estimates total cost BEFORE sending any requests and requires
  confirmation. Test set is ~9,167 posts; at typical frontier pricing
  this is a few dollars per provider, per variant. Verify current
  pricing yourself before running, it changes.

Setup:
  pip install openai anthropic google-genai
  export OPENAI_API_KEY=...
  export ANTHROPIC_API_KEY=...
  export GOOGLE_API_KEY=...
  export DEEPSEEK_API_KEY=...

Usage:
    python src/frontier_zeroshot.py --provider openai --variant full
    python src/frontier_zeroshot.py --provider anthropic --variant full
    python src/frontier_zeroshot.py --provider google --variant full
    (repeat each with --variant stripped)
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

LABELS = ["depression", "SuicideWatch", "Anxiety", "bipolar"]
SEED = 42

STRIP_TERMS = {
    "anxiety": ["anxiety", "anxious", "anxieties"],
    "depression": ["depression", "depressed", "depressive"],
    "bipolar": ["bipolar", "manic", "mania", "hypomania"],
    "suicidewatch": ["suicide", "suicidal", "kill myself", "end my life"],
}

PROVIDER_CONFIG = {
    "openai": {
        # GPT-4o was OpenAI's flagship through mid-2025; the GPT-5 family
        # (launched August 2025) replaced it as frontier. gpt-4o still
        # works via the API but is legacy now, using it would repeat the
        # same "not actually frontier" mistake flagged for Gemini Flash
        # and GPT-4o-mini elsewhere in this project.
        # ACTION REQUIRED: replace this string with the current flagship
        # GPT-5-family model, the "Instant" variant specifically, not
        # "Thinking". Thinking does visible/extended reasoning before
        # answering, the same mechanism that made Claude Opus 4.8 far
        # more expensive than estimated (reasoning tokens bill as
        # output). A 4-way classification task doesn't need that.
        # Verify the exact current model string yourself at
        # platform.openai.com/docs/models before running -- naming has
        # moved fast (5.2, 5.3, 5.4, 5.5 all shipped within months of
        # each other) and this comment will go stale quickly.
        "model": "gpt-5.4", 
        "est_input_price_per_mtok": 2.50,   # VERIFY current pricing yourself
        "est_output_price_per_mtok": 10.00,
    },
    "anthropic": {
        "model": "claude-sonnet-5",
        # Sonnet 5 pricing: $3/M input, $15/M output ($2/$10 introductory
        # through 2026-08-31). Higher than Haiku 4.5 ($1/$5) -- re-check the
        # $3 budget note in Experimental Design / Limitations before running
        # the full test set, not just --limit.
        "est_input_price_per_mtok": 3.00,
        "est_output_price_per_mtok": 15.00,
    },
    "google": {
        "model": "gemini-3.1-pro",
        "est_input_price_per_mtok": 2.00,   # verify current pricing yourself
        "est_output_price_per_mtok": 12.00,
    },
    "deepseek": {
        "model": "deepseek-chat",  # DeepSeek-V3, their frontier tier -- verify current pricing yourself
        "est_input_price_per_mtok": 0.27,
        "est_output_price_per_mtok": 1.10,
    }
}

REFUSAL_PATTERNS = [
    r"i (can'?t|cannot|won'?t|am not able to)",
    r"i'?m (not able|unable) to",
    r"as an ai",
    r"i don'?t feel comfortable",
    r"i'?m sorry,? (but )?i",
    r"this (content|post) (contains|describes|involves) sensitive",
]


def strip_keywords(text):
    out = text
    for terms in STRIP_TERMS.values():
        for t in terms:
            out = re.sub(r"\b" + re.escape(t) + r"\w*", "[REDACTED]", out, flags=re.I)
    return out


def build_prompt(text):
    return (
        "Classify the following Reddit post into exactly one category: "
        "depression, SuicideWatch, Anxiety, or bipolar. "
        "Respond with only the category name, nothing else.\n\n"
        f"Post: {text}\n\nCategory:"
    )


def parse_response(raw):
    text = raw.strip()
    low = text.lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, low):
            return "REFUSED", raw
    # Use the LAST label mention, not the first. Short forced-format
    # responses (GPT-4o, Gemini) only ever contain one label anyway, so
    # this is a no-op for them. But longer reasoning-style responses
    # (e.g. Opus 4.8 with adaptive thinking) may mention a category while
    # reasoning toward a different final answer -- the last mention is
    # the model's actual conclusion, not an earlier one it considered
    # and moved past.
    found = [l for l in LABELS if l.lower() in low]
    if not found:
        return "UNKNOWN", raw
    last_positions = {l: low.rfind(l.lower()) for l in found}
    pred = max(last_positions, key=last_positions.get)
    return pred, raw


def call_openai(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=50,
        temperature=0,
    )
    return resp.choices[0].message.content


def call_anthropic(client, model, prompt):
    # claude-sonnet-5 runs adaptive thinking by default when `thinking` is
    # omitted -- disabled here since this is single-label classification,
    # not a reasoning task, and thinking blocks would otherwise shift or
    # eat into the classification response.
    resp = client.messages.create(
        model=model,
        max_tokens=50,
        thinking={"type": "disabled"},
        messages=[{"role": "user", "content": prompt}],
    )
    return next(b.text for b in resp.content if b.type == "text")


def call_google(client, model, prompt):
    resp = client.models.generate_content(model=model, contents=prompt)
    return resp.text


CALL_FN = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "deepseek": call_openai,  # DeepSeek's API is OpenAI-compatible
}


def get_client(provider):
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if provider == "google":
        from google import genai
        return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    if provider == "deepseek":
        from openai import OpenAI
        return OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    raise ValueError(provider)


def estimate_cost(texts, cfg):
    # Rough estimate: ~1.3 tokens/word for input, fixed small output budget.
    total_words = sum(len(t.split()) for t in texts)
    input_tokens = total_words * 1.3 + len(texts) * 40  # + prompt template overhead
    output_tokens = len(texts) * 8  # short category-name responses
    cost = (
        input_tokens / 1e6 * cfg["est_input_price_per_mtok"]
        + output_tokens / 1e6 * cfg["est_output_price_per_mtok"]
    )
    return cost, int(input_tokens), int(output_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDER_CONFIG), required=True)
    ap.add_argument("--variant", choices=["full", "stripped"], required=True)
    ap.add_argument("--data-dir", default="data/processed/4class")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--limit", type=int, default=None,
                     help="test on a subset first, e.g. --limit 200, before committing to the full run")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls, raise if rate-limited")
    ap.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    args = ap.parse_args()

    cfg = PROVIDER_CONFIG[args.provider]
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    if args.variant == "stripped":
        test_df["text"] = test_df["text"].astype(str).apply(strip_keywords)
    if args.limit:
        test_df = test_df.sample(args.limit, random_state=SEED).reset_index(drop=True)

    texts = test_df["text"].tolist()
    cost, in_tok, out_tok = estimate_cost(texts, cfg)
    print(f"Provider: {args.provider}  |  Model: {cfg['model']}  |  Variant: {args.variant}")
    print(f"n={len(texts)}  est. input tokens={in_tok:,}  est. output tokens={out_tok:,}")
    print(f"ESTIMATED COST: ${cost:.2f}  (verify current pricing yourself; this is approximate)")

    if not args.yes:
        confirm = input("Proceed? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return

    client = get_client(args.provider)
    call_fn = CALL_FN[args.provider]

    preds, raws = [], []
    running_calls = 0
    for text in tqdm(texts, desc=f"{args.provider} {args.variant}"):
        prompt = build_prompt(text)
        try:
            raw = call_fn(client, cfg["model"], prompt)
        except Exception as e:
            raw = f"[API_ERROR: {e}]"
        pred, _ = parse_response(raw)
        preds.append(pred)
        raws.append(raw)
        running_calls += 1
        if running_calls % 50 == 0:
            avg_resp_len = sum(len(r) for r in raws[-50:]) / 50
            if avg_resp_len > 400:
                print(
                    f"\nWARNING: average response length over the last 50 calls "
                    f"is {avg_resp_len:.0f} characters, much longer than a short "
                    f"category label. This provider may be spending more on "
                    f"reasoning/output tokens than expected. Check your API "
                    f"console's ACTUAL billed usage now, not just this script's "
                    f"upfront estimate, before letting this continue."
                )
        if args.sleep:
            time.sleep(args.sleep)

    true = test_df["label"].tolist()
    refusal_rate = sum(p == "REFUSED" for p in preds) / len(preds)
    unparseable_rate = sum(p == "UNKNOWN" for p in preds) / len(preds)

    # Refusals and unparseable both count as wrong for the F1 calculation
    # (they ARE wrong predictions), but are reported as separate rates so
    # the failure mode is visible, not hidden inside one aggregate number.
    report = classification_report(true, preds, labels=LABELS, output_dict=True, zero_division=0)
    macro_f1 = f1_score(true, preds, labels=LABELS, average="macro", zero_division=0)

    print(classification_report(true, preds, labels=LABELS, zero_division=0))
    print(f"macro-F1: {macro_f1:.4f}")
    print(f"refusal rate: {refusal_rate * 100:.1f}%")
    print(f"unparseable rate: {unparseable_rate * 100:.1f}%")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_{args.provider}-{cfg['model']}_{args.variant}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": cfg["model"],
            "provider": args.provider,
            "variant": args.variant,
            "macro_f1": macro_f1,
            "anxiety_f1": report.get("Anxiety", {}).get("f1-score"),
            "unparseable_rate": unparseable_rate,
            "refusal_rate": refusal_rate,
            "full_report": report,
            "n_test": len(test_df),
        }, f, indent=2)

    raw_path = os.path.join(args.out_dir, f"raw_{args.provider}_{args.variant}.csv")
    pd.DataFrame({"text": texts, "true": true, "pred": preds, "raw_response": raws}).to_csv(raw_path, index=False)

    print(f"\nSaved to {out_path}")
    print(f"Raw responses saved to {raw_path} (useful for inspecting refusals/unparseable cases later)")


if __name__ == "__main__":
    main()
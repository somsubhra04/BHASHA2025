#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

SUPPORTED_LANGS = ["hindi", "telugu", "bangla", "malayalam", "tamil"]

SYSTEM_PROMPT = """You are a Grammatical Error Correction (GEC) assistant for low-resource Indian languages.
Your job: correct only grammar, spelling, spacing, matras/diacritics, punctuation, and light word-form errors.
Do NOT translate. Preserve the meaning, script, and style of the input language.
Return ONLY the corrected sentence with no quotes, no labels, no extra text.
If the input is already correct, return it unchanged."""

def build_zero_shot_prompt():
    """Return a single system message only (zero-shot)."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]

def correct_sentence(client, base_messages, sent: str, model: str = "gpt-4.1-mini", temperature: float = 0.2) -> str:
    messages = list(base_messages) + [{"role": "user", "content": f"INPUT:\n{sent}\n\nOUTPUT:"}]
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        input=messages
    )
    try:
        text = resp.output_text.strip()
    except Exception:
        try:
            text = resp.output[0].content[0].text.strip()
        except Exception:
            text = sent
    return " ".join(text.splitlines()).strip()

def main():
    parser = argparse.ArgumentParser(description="Zero-shot GEC inference for Indian languages")
    parser.add_argument("--lang", type=str, required=True, help=f"one of: {', '.join(SUPPORTED_LANGS)}")
    parser.add_argument("--data_root", type=str, default="data", help="root folder containing language subfolders")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=-1, help="debug: limit number of rows")
    args = parser.parse_args()

    lang = args.lang.lower()
    if lang not in SUPPORTED_LANGS:
        print(f"[ERROR] Unsupported --lang '{lang}'. Choose from {SUPPORTED_LANGS}.", file=sys.stderr)
        sys.exit(1)

    lang_dir = os.path.join(args.data_root, lang)
    dev_csv = os.path.join(lang_dir, "dev.csv")
    out_csv = os.path.join(lang_dir, "predictions_zeroshot.csv")

    if not os.path.isfile(dev_csv):
        print(f"[ERROR] Missing file: {dev_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(dev_csv)
    if not {"Input sentence", "Output sentence"}.issubset(df.columns):
        print("[ERROR] dev.csv must have columns: 'Input sentence' and 'Output sentence'", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        df = df.head(args.limit)

    try:
        client = OpenAI()  # API key is read from env
    except Exception as e:
        print(f"[ERROR] Failed to init OpenAI client: {e}", file=sys.stderr)
        sys.exit(1)

    base_messages = build_zero_shot_prompt()

    inputs = df["Input sentence"].fillna("").astype(str).tolist()
    preds = []

    print(f"[INFO] Running ZERO-SHOT GEC inference on {len(inputs)} sentences for '{lang}'...")
    for sent in tqdm(inputs):
        try:
            corrected = correct_sentence(client, base_messages, sent, model=args.model, temperature=args.temperature)
        except Exception as e:
            corrected = sent
            print(f"[WARN] Error on sentence; returning original. Err={e}", file=sys.stderr)
        preds.append(corrected)

    out_df = pd.DataFrame({"Input sentence": inputs, "Output sentence": preds})
    out_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved zero-shot predictions to: {out_csv}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

SUPPORTED_LANGS = ["hindi", "telugu", "bangla", "malayalam", "tamil"]

SYSTEM_PROMPT = """You are a Grammatical Error Correction (GEC) assistant for low-resource Indian languages.
Your job: correct only grammar, spelling, spacing, matras/diacritics, punctuation, and light word-form errors.
Do NOT translate. Preserve the meaning, script, and style of the input language.
Return ONLY the corrected sentence with no quotes, no labels, no extra text.
If the input is already correct, return it unchanged."""


def build_zero_shot_messages(sentence: str):
    """return messages array for llama"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"INPUT:\n{sentence}\n\nOUTPUT:"}
    ]


def call_llama(messages, api_key):
    """Send a single non-streaming request to NVIDIA NIM Llama-4."""

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"

    payload = {
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.2,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "stream": False
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
        resp_json = resp.json()
        text = resp_json["choices"][0]["message"]["content"]
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Llama API failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Zero-shot GEC inference using NVIDIA NIM Llama-4")
    parser.add_argument("--lang", type=str, required=True, help=f"one of: {', '.join(SUPPORTED_LANGS)}")
    parser.add_argument("--data_root", type=str, default="test", help="root folder containing language test folders")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--rpm", type=int, default=35)
    args = parser.parse_args()

    lang = args.lang.lower()
    if lang not in SUPPORTED_LANGS:
        print(f"[ERROR] Unsupported --lang '{lang}'. Choose from {SUPPORTED_LANGS}.", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[ERROR] Please export NVIDIA_API_KEY first.", file=sys.stderr)
        print("export NVIDIA_API_KEY='your_key_here'")
        sys.exit(1)

    lang_csv = os.path.join(args.data_root, f"{lang}.csv")
    if not os.path.isfile(lang_csv):
        print(f"[ERROR] Missing test file: {lang_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(lang_csv)
    if "Input sentence" not in df.columns:
        print("[ERROR] Test CSV must have column: 'Input sentence'", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        df = df.head(args.limit)

    out_dir = os.path.join("infer_llama", lang)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "predictions.csv")

    inputs = df["Input sentence"].fillna("").astype(str).tolist()
    preds = []

    sleep_time = max(60.0 / args.rpm, 1.8)

    print(f"[INFO] Running ZERO-SHOT LLAMA inference on {len(inputs)} sentences for '{lang}'...")
    print(f"[INFO] Enforcing {args.rpm} RPM → sleep {sleep_time:.2f}s between requests")

    for sent in tqdm(inputs):
        msgs = build_zero_shot_messages(sent)
        output = call_llama(msgs, api_key)

        if output is None:
            output = sent  # fallback

        preds.append(output)
        time.sleep(sleep_time)

    out_df = pd.DataFrame({"Input sentence": inputs, "Output sentence": preds})
    out_df.to_csv(out_csv, index=False, encoding="utf-8")

    print(f"[OK] Saved LLAMA zero-shot predictions to: {out_csv}")


if __name__ == "__main__":
    main()

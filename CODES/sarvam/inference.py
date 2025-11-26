#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------
# Configuration
# --------------------------
MODEL_PATH = "outputs/sarvam_m_gec_lora/final_merged"
INPUT_FILE = "infer_data/dev.csv"
RESULTS_DIR = "results"
OUTPUT_FILE = os.path.join(RESULTS_DIR, "dev_inference_hi.csv")

PROMPT_TEMPLATE = (
    "### निर्देश:\n"
    "नीचे दिए गए वाक्य की व्याकरण-सुधारित रूप प्रदान करें।\n\n"
    "### इनपुट:\n{src}\n\n"
    "### उत्तर:\n"
)

# --------------------------
# Prepare model & tokenizer
# --------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

# Ensure SentencePiece model exists
if not os.path.exists(os.path.join(MODEL_PATH, "tokenizer.model")):
    print("⚠️ tokenizer.model missing. Copy it from base model if needed.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)
model.eval()

# --------------------------
# Load CSV data
# --------------------------
df = pd.read_csv(INPUT_FILE)
if "Input sentence" not in df.columns:
    raise ValueError("The CSV must contain a column named 'Input sentence'")

inputs = df["Input sentence"].astype(str).tolist()
refs = df["Output sentence"].astype(str).tolist() if "Output sentence" in df.columns else None

# --------------------------
# Generate predictions
# --------------------------
print(f"Generating predictions for {len(inputs)} sentences...")
outputs = []

for src in tqdm(inputs):
    prompt = PROMPT_TEMPLATE.format(src=src)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        gen = model.generate(
            **enc,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.0,
            top_p=0.9,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
    text = tok.decode(gen[0], skip_special_tokens=True)
    # Extract only the answer part
    if "### उत्तर:" in text:
        pred = text.split("### उत्तर:")[-1].strip()
    else:
        pred = text.strip()
    outputs.append(pred)

# --------------------------
# Save results
# --------------------------
out_df = pd.DataFrame({
    "Input sentence": inputs,
    "Reference": refs if refs else ["" for _ in range(len(inputs))],
    "Prediction": outputs,
})
out_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ Inference complete. Saved predictions to: {OUTPUT_FILE}")
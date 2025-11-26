# -*- coding: utf-8 -*-

import tiktoken
import google.generativeai as genai

genai.configure(api_key="")

import pandas as pd
import tiktoken
import google.generativeai as genai
import os

enc = tiktoken.get_encoding("o200k_base")


files = [
    "bn-test.csv",
    "tam-test.csv",
    "mal-test.csv",
    "hi-test.csv",
    "tel-test.csv"
]
target_col = "Input sentence" 

print(f"{'File':<15} | {'Words':<8} | {'GPT-4o Mini':<12}")
print("-" * 60)

results = []

for filename in files:
    if not os.path.exists(filename):
        print(f"{filename:<15} | FILE NOT FOUND")
        continue

    try:
        df = pd.read_csv(filename)

        if target_col not in df.columns:
            print(f"{filename:<15} | Column '{target_col}' not found")
            continue

        text_data = df[target_col].dropna().astype(str).tolist()

        total_words = 0
        gpt_tokens = 0
        gemini_tokens = 0

        total_words = sum(len(s.split()) for s in text_data)

        for s in text_data:
            gpt_tokens += len(enc.encode(s))

        gpt_fertility = round(gpt_tokens / total_words, 2) if total_words > 0 else 0


        print(f"{filename:<15} | {total_words:<8} | {gpt_fertility:<12}")

        results.append({
            "Language": filename.split('-')[0].upper(),
            "GPT-4o Fertility": gpt_fertility,
        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")

results


# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install transformers huggingface_hub

import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import login
import os

# --- AUTHENTICATION ---
login(token="")

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

print(f"Loading Tokenizer for {model_id}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
except OSError:
    print("Error: You must log in to Hugging Face and accept the Llama 4 license first.")
    exit()

files = [
    "bn-test.csv", "tam-test.csv", "mal-test.csv",
    "hi-test.csv", "tel-test.csv"
]
target_col = "Input sentence"

print(f"\n{'File':<15} | {'Words':<8} | {'Llama 4 Fertility':<18}")
print("-" * 50)

results = []

for filename in files:
    if not os.path.exists(filename):
        continue

    try:
        df = pd.read_csv(filename)
        text_data = df[target_col].dropna().astype(str).tolist()

        total_words = sum(len(s.split()) for s in text_data)

        encoded_batch = tokenizer(text_data, add_special_tokens=False)["input_ids"]
        total_tokens = sum(len(ids) for ids in encoded_batch)

        fertility = round(total_tokens / total_words, 2) if total_words > 0 else 0

        print(f"{filename:<15} | {total_words:<8} | {fertility:<18}")

        results.append({
            "Language": filename.split('-')[0].upper(),
            "Llama 4 Fertility": fertility
        })

    except Exception as e:
        print(f"Error processing {filename}: {e}")


import pandas as pd
import tiktoken
import google.generativeai as genai
from transformers import AutoTokenizer
from huggingface_hub import login


login(token="")

genai.configure(api_key="")


enc_gpt = tiktoken.get_encoding("o200k_base")

tok_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-4-Maverick-17B-128E-Instruct")

tok_gemini_proxy = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
real_gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# --- SAMPLES ---
samples = {
    "HI": "हम बस धीडे- धीडे चलते-चलते गलियों में धूम ही रहे थे कि हमने एक लड़की को बहुत सुंडर कपड़े बोचते हुए देखा।",
    "MAL": "ഇന്ന് സഹോദരന്മാർ തമ്മിൽ അടിക്കുന്ന ഈ കാലഘട്ടത്തിൽ നാം ലക്ഷ്മണനിൽ നിന്നും ഒരുപാട് കണ്ടുപിടിക്കേണ്ടതുണ്ട്.",
    "TAM": "திரும்ப ஹோட்டலுக்கு செள்ள டாக்சி கிடைக்காததால், அவர்கள் பெற்றோருடன் என் பெற்றோர்களை அனுப்பி வைத்தேன்.",
    "BN": "উত্তর হিমালয় ভারত পাহারা দিচ্ছে , এই জেনে নিশ্চিত হয়ে প্রায় তিন দিকের উপকূল রয়ে গেল চিরকাল অরক্ষিত ।",
    "TEL": "మను చెవి పరిపిత శబ్దం కన్నా ఎక్కువ తీవ్రతతో ఉన్న శబ్దాలు వినటం వల్ల వినికిడి శక్తి కొల్పోయే ప్రమాదం ఉంది"
}

def get_gpt_data(text):
    ids = enc_gpt.encode(text)
    tokens = [enc_gpt.decode_single_token_bytes(i).decode('utf-8', errors='replace') for i in ids]
    return len(ids), tokens

def get_llama_data(text):
    tokens = tok_llama.tokenize(text)
    return len(tokens), tokens

def get_gemini_data(text):
    # Get REAL count from API
    try:
        real_count = real_gemini_model.count_tokens(text).total_tokens
    except:
        real_count = "Err"

    # Get Visuals from Proxy (Gemma 2)
    visual_tokens = tok_gemini_proxy.tokenize(text)

    return real_count, visual_tokens

print(f"{'Lang':<5} | {'CHARS':<6} | {'Model':<12} | {'Count':<5} | {'Token Breakdown'}")
print("-" * 110)

for lang, text in samples.items():
    char_count = len(text)

    c_gpt, t_gpt = get_gpt_data(text)
    print(f"{lang:<5} | {char_count:<6} | {'GPT-4o':<12} | {c_gpt:<5} | {t_gpt}")

    c_gem, t_gem = get_gemini_data(text)
    print(f"{'':<5}| {'':<6} | {'Gemini 2.5':<12} | {c_gem:<5} | {t_gem}")

    c_llama, t_llama = get_llama_data(text)
    print(f"{'':<5}| {'':<6} | {'Llama 4':<12} | {c_llama:<5} | {t_llama}")

    print("-" * 110)
# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install bert-score indic-nlp-library pandas scikit-learn tqdm

import os
import zipfile
import glob
import pandas as pd
from collections import Counter
from bert_score import score as bert_score
from sklearn.metrics import fbeta_score
from tqdm import tqdm
import re
import json
from indicnlp.tokenize import indic_tokenize
from indicnlp import common

common.set_resources_path("/path/to/indic_nlp_resources")

def tokenize(text):
    return re.findall(r"\w+|\S", text.strip().lower())

def tokenize2(text):
    return indic_tokenize.trivial_tokenize(text)

# --- Compute Corpus GLEU (1–4 gram) (as per task scoring func) ---
def corpus_gleu_1to4(ref_tokens_list, hyp_tokens_list):
    """
    Calculates the corpus-level GLEU score (average of min(Precision, Recall)
    for n-grams from n=1 to 4).
    """
    scores = []
    for n in range(1, 5):
        ref_total = hyp_total = matches = 0
        for rt, ht in zip(ref_tokens_list, hyp_tokens_list):
            r = Counter(tuple(rt[i:i+n]) for i in range(max(0, len(rt)-n+1)))
            h = Counter(tuple(ht[i:i+n]) for i in range(max(0, len(ht)-n+1)))

            ref_total += sum(r.values())
            hyp_total += sum(h.values())

            if r and h:
                matches += sum((r & h).values())

        prec = (matches / hyp_total) if hyp_total > 0 else 0.0
        rec  = (matches / ref_total) if ref_total > 0 else 0.0
        scores.append(min(prec, rec))

    return (sum(scores) / len(scores)) * 100.0 if scores else 0.0


# --- Compute F0.5 ---
def compute_f05_score(references, predictions):
    """
    Compute F0.5 score for each sentence pair and return the average.

    Args:
        references (list of str): List of reference sentences.
        predictions (list of str): List of predicted sentences.

    Returns:
        float: Average F0.5 score at the sentence level.
    """
    f05_scores = []

    for ref, pred in zip(references, predictions):
        ref_tokens = set(tokenize2(ref))
        pred_tokens = set(tokenize2(pred))

        tp = len(ref_tokens & pred_tokens)
        fp = len(pred_tokens - ref_tokens)
        fn = len(ref_tokens - pred_tokens)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0

        f05_scores.append(f05)

    # Return average F0.5 across all sentences
    return sum(f05_scores) / len(f05_scores)

# --- Main evaluation loop ---
results = []

for zip_path in glob.glob("*-few-gem-predictions.zip"): #method, model inf to be evaluated
    lang = zip_path.split("-")[0]
    gold_path = f"{lang}-fin-test.csv"

    if not os.path.exists(gold_path):
        print(f"⚠️ Gold file not found for {lang}, skipping...")
        continue

    # Unzip predictions
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(f"./tmp_{lang}/")

    pred_file = os.path.join(f"./tmp_{lang}/", "predictions.csv")
    preds_df = pd.read_csv(pred_file)
    gold_df = pd.read_csv(gold_path)

    # Align columns
    references = gold_df["Output sentence"].astype(str).tolist()
    predictions = preds_df["Output sentence"].astype(str).tolist()

    # Tokenize for GLEU
    ref_tokens_list = [tokenize(r) for r in references]
    hyp_tokens_list = [tokenize(h) for h in predictions]

    # Compute metrics
    P, R, F1 = bert_score(predictions, references, lang="multilingual", rescale_with_baseline=False)
    bert_f1 = F1.mean().item()

    f05 = compute_f05_score(references, predictions)
    gleu = corpus_gleu_1to4(ref_tokens_list, hyp_tokens_list)

    # results
    results.append({
        "Language": lang,
        "BERTScore_F1": bert_f1,
        "F0.5": f05,
        "Corpus_GLEU_1to4": gleu
    })

results_df = pd.DataFrame(results)

results_df.to_csv("evaluation_results.csv", index=False)
results_df
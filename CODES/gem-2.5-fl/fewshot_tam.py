# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install -q google-generativeai
# !pip install -q pandas
# !pip install -q nltk

import pandas as pd
import google.generativeai as genai
import nltk
from nltk.translate.gleu_score import sentence_gleu

nltk.download('punkt')
nltk.download('punkt_tab')

TRAIN_FILE_PATH = '/content/tam-train.csv'
VALIDATION_FILE_PATH = '/content/tam-dev.csv'
TEST_FILE_PATH = '/content/tam-test.csv'

train_df = pd.read_csv(TRAIN_FILE_PATH)
val_df = pd.read_csv(VALIDATION_FILE_PATH)
test_df = pd.read_csv(TEST_FILE_PATH)


from google.colab import userdata
genai.configure(api_key=userdata.get('GEMINI_API_KEY'))

model = genai.GenerativeModel('gemini-2.5-flash')

train_df

val_df

test_df

import time
# Variable to track the time of the last request
last_request_time = None
REQUEST_LIMIT = 10  # requests per minute
REQUEST_INTERVAL = 60 / REQUEST_LIMIT  # Interval in seconds between requests

def create_few_shot_prompt(input_sentence, num_examples=10):
    """
    Creates a few-shot prompt for the Gemini model.
    """
    # Sample 10 random examples from the training data
    few_shot_examples = train_df.sample(n=num_examples)

    # Build the prompt string

    prompt_parts = [
    "You are a Tamil Grammatical Error Correction assistant, in low resource settings. Your task is to accurately identify and correct grammatical errors in the given Tamil sentence. Correct all types of grammatical errors:",
    "**Verb usage**: Correct conjugation, tense, aspect, and agreement with the subject.",
    "**Pronouns**: Usage of proper personal, possessive, and reflexive pronouns.",
    "**Prepositions**: Correct use of postpositions or prepositions in context.",
    "Fix spelling mistakes, diacritic marks (matras), and punctuation errors.",
    "Gender and number agreement**: Ensure adjectives, nouns, and verbs match in gender (masculine/feminine) and number (singular/plural).",
    "The output should be ONLY the CORRECTED sentence, without any extra text or explanation. If the input is already correct, return it unchanged. Please ensure the corrections follow the rules and preserve the intended meaning. Below are 10 random sentences for your reference."
    ]

    for _, row in few_shot_examples.iterrows():
        prompt_parts.append(f"Input: {row['Input sentence']}\nOutput: {row['Output sentence']}\n\n")

    prompt_parts.append(f"Input: {input_sentence}\nOutput:")

    return "".join(prompt_parts)

def get_gemini_prediction(prompt):
    """
    Sends the prompt to the Gemini API and returns the corrected sentence,
    with rate-limiting to avoid exceeding the 10 requests per minute limit.
    """
    global last_request_time

    # If this isn't the first request, calculate how long since the last one
    if last_request_time is not None:
        time_since_last_request = time.time() - last_request_time
        if time_since_last_request < REQUEST_INTERVAL:
            # Sleep for the remaining time to respect the rate limit
            time_to_wait = REQUEST_INTERVAL - time_since_last_request
            print(f"Rate limit exceeded. Waiting for {time_to_wait:.2f} seconds...")
            time.sleep(time_to_wait)

    try:
        # Update the last request time to the current time
        last_request_time = time.time()

        # Send the request to Gemini
        response = model.generate_content(prompt)

        # Return the corrected sentence, stripping any leading/trailing whitespace
        return response.text.strip()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


predictions = []
references = []

print("Starting inference on validation set...")

for index, row in val_df.iterrows():
    input_text = row['Input sentence']
    reference_text = row['Output sentence']

    # Create the few-shot prompt for the current sentence
    prompt = create_few_shot_prompt(input_text, num_examples=10)

    # Get the model's prediction
    predicted_text = get_gemini_prediction(prompt)

    if predicted_text:
        predictions.append(predicted_text)
        references.append(reference_text)
        print(f"Processed {index + 1}/{len(val_df)}: Input: '{input_text[:30]}...' -> Prediction: '{predicted_text[:30]}...'")

# Calculate the overall GLEU score
if predictions and references:
    tokenized_predictions = [nltk.word_tokenize(p) for p in predictions]
    tokenized_references = [[nltk.word_tokenize(r)] for r in references]

    # Calculate corpus-level GLEU score
    corpus_gleu_score = nltk.translate.gleu_score.corpus_gleu(tokenized_references, tokenized_predictions)

    print("\n" + "="*50)
    print(f"Corpus-level GLEU Score: {corpus_gleu_score:.4f}")
    print("="*50)

else:
    print("\nNo predictions!")

predictions

references

from google.colab import files

data = {
    'Input Sentence': val_df['Input sentence'].tolist(),
    'Output Sentence': references,
    'Generated Sentence': predictions
}

df_results = pd.DataFrame(data)

df_results.to_csv('gec_tam_results.csv', index=False, encoding='utf-8-sig')

print("Downloading the results CSV file...")
files.download('gec_tam_results.csv')

"""

---

"""

predictions = []

print("Starting inference on test set...")

for index, row in test_df.iterrows():
    input_text = row['Input sentence']

    prompt = create_few_shot_prompt(input_text, num_examples=10)

    predicted_text = get_gemini_prediction(prompt)

    if predicted_text:
        predictions.append(predicted_text)
        print(f"Processed {index + 1}/{len(test_df)}: Input: '{input_text[:30]}...' -> Prediction: '{predicted_text[:30]}...'")

predictions

len(predictions)

#final test set runs
from google.colab import files
import zipfile
import os

data = {
    'Input sentence': test_df['Input sentence'].tolist(),
    'Output sentence': predictions
}

df_results = pd.DataFrame(data)
csv_filename = 'predictions.csv'

df_results.to_csv(csv_filename, index=False, encoding='utf-8-sig')

zip_filename = 'tam-few-gem-predictions.zip'

with zipfile.ZipFile(zip_filename, 'w') as zipf:
    zipf.write(csv_filename, os.path.basename(csv_filename))

print("Downloading the ZIP file...")
files.download(zip_filename)
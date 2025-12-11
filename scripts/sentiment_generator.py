pip install transformers torch pandas

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

INPUT_CSV = "data/news_deduped.csv"
OUTPUT_CSV = "data/news_deduped_finbert.csv"
TEXT_COLUMN = "clean_headline"   # from your file
BATCH_SIZE = 32
MODEL_NAME = "ProsusAI/finbert"  # FinBERT for financial sentiment

df = pd.read_csv(INPUT_CSV)

df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

id2label = model.config.id2label 

def run_finbert(texts):
    """Return list of (label, score, prob_negative, prob_neutral, prob_positive)"""
    results = []

    # iterate by batches
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        probs_cpu = probs.cpu().numpy()

        for row_probs in probs_cpu:
            max_idx = row_probs.argmax()
            label = id2label[int(max_idx)]  # 'negative', 'neutral', 'positive'
            score = float(row_probs[max_idx])

            # assume order [negative, neutral, positive]; check model.config.id2label if needed
            prob_negative = float(row_probs[0])
            prob_neutral = float(row_probs[1]) if probs.shape[1] > 1 else None
            prob_positive = float(row_probs[2]) if probs.shape[1] > 2 else None

            results.append((label, score, prob_negative, prob_neutral, prob_positive))

    return results

texts = df[TEXT_COLUMN].tolist()
sentiment_results = run_finbert(texts)

(
    df["finbert_label"],
    df["finbert_score"],
    df["finbert_prob_negative"],
    df["finbert_prob_neutral"],
    df["finbert_prob_positive"],
) = zip(*sentiment_results)

label_to_numeric = {"negative": -1, "neutral": 0, "positive": 1}
df["finbert_numeric"] = df["finbert_label"].map(label_to_numeric)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved with FinBERT scores to {OUTPUT_CSV}")

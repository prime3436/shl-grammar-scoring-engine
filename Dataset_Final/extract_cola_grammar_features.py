import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

torch.set_num_threads(8)

DATA_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026\Dataset_Final"

print("="*60)
print("EXTRACTING ROBERTA-COLA GRAMMATICAL ACCEPTABILITY FEATURES")
print("="*60)

train_df = pd.read_csv(os.path.join(DATA_DIR, "train_features.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_features.csv"))

model_name = "textattack/roberta-base-CoLA"
print(f"Loading {model_name} from cache...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

def extract_cola_for_dataframe(df, name="dataset"):
    print(f"\nProcessing {name} ({len(df)} audios)...")
    t0 = time.time()
    
    # Flatten sentences and record mapping
    all_sents = []
    sent_map = [] # (row_index)
    
    for idx, row in df.iterrows():
        text = str(row['transcript']).strip()
        if not text or text.lower() == 'nan':
            sents = [""]
        else:
            sents = nltk.sent_tokenize(text)
            if not sents:
                sents = [text]
        for s in sents:
            all_sents.append(s)
            sent_map.append(idx)
            
    print(f"Total sentences to evaluate: {len(all_sents)}")
    
    # Batched inference
    BATCH_SIZE = 64
    all_probs = []
    
    for i in range(0, len(all_sents), BATCH_SIZE):
        batch_texts = all_sents[i:i+BATCH_SIZE]
        # Clean empty texts
        batch_cleaned = [t if t.strip() else "Empty." for t in batch_texts]
        
        inputs = tokenizer(batch_cleaned, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            all_probs.extend(probs)
            
        if (i // BATCH_SIZE) % 10 == 0 or i + BATCH_SIZE >= len(all_sents):
            print(f"  Processed {min(i + BATCH_SIZE, len(all_sents))}/{len(all_sents)} sentences...")
            
    all_probs = np.array(all_probs)
    
    # Aggregate back to per-audio metrics
    records = []
    sent_map = np.array(sent_map)
    for idx in range(len(df)):
        idxs = np.where(sent_map == idx)[0]
        if len(idxs) == 0:
            records.append({
                'cola_mean_prob': 0.5,
                'cola_min_prob': 0.5,
                'cola_max_prob': 0.5,
                'cola_std_prob': 0.0,
                'cola_error_count': 0,
                'cola_error_rate': 0.0,
                'cola_clean_rate': 0.0,
                'cola_severe_errors': 0
            })
        else:
            doc_p = all_probs[idxs]
            n_sents = len(doc_p)
            mean_p = float(np.mean(doc_p))
            min_p = float(np.min(doc_p))
            max_p = float(np.max(doc_p))
            std_p = float(np.std(doc_p)) if n_sents > 1 else 0.0
            err_cnt = int(np.sum(doc_p < 0.5))
            err_rate = float(err_cnt / max(n_sents, 1))
            clean_rate = float(np.sum(doc_p > 0.85) / max(n_sents, 1))
            severe_cnt = int(np.sum(doc_p < 0.25))
            
            records.append({
                'cola_mean_prob': mean_p,
                'cola_min_prob': min_p,
                'cola_max_prob': max_p,
                'cola_std_prob': std_p,
                'cola_error_count': err_cnt,
                'cola_error_rate': err_rate,
                'cola_clean_rate': clean_rate,
                'cola_severe_errors': severe_cnt
            })
            
    res_df = pd.DataFrame(records)
    res_df['filename'] = df['filename']
    print(f"Completed {name} in {time.time() - t0:.2f}s")
    return res_df

train_cola = extract_cola_for_dataframe(train_df, "Train")
test_cola = extract_cola_for_dataframe(test_df, "Test")

# Compute correlation with train labels (speech samples)
speech_mask = train_df['label'] > 0
y_speech = train_df.loc[speech_mask, 'label'].values
for col in ['cola_mean_prob', 'cola_min_prob', 'cola_max_prob', 'cola_error_count', 'cola_error_rate', 'cola_clean_rate']:
    corr = np.corrcoef(train_cola.loc[speech_mask, col].values, y_speech)[0, 1]
    print(f"Feature '{col:20s}' correlation with Grammar Label: {corr:+.4f}")

# Save
tr_out = os.path.join(DATA_DIR, "train_cola_features.csv")
te_out = os.path.join(DATA_DIR, "test_cola_features.csv")
train_cola.to_csv(tr_out, index=False)
test_cola.to_csv(te_out, index=False)
print(f"\nSaved CoLA features to:")
print(f"  -> {tr_out}")
print(f"  -> {te_out}")

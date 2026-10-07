import os
import time
import soundfile as sf
import numpy as np
import pandas as pd
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

torch.set_num_threads(8)

DATA_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026\Dataset_Final"
train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_out = os.path.join(DATA_DIR, "train_wav2vec2_features.npy")
test_out = os.path.join(DATA_DIR, "test_wav2vec2_features.npy")

print("Loading Wav2Vec2 model and processor...")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
model.eval()
print("Model loaded successfully.")

def extract_features_for_df(df, audio_dir, out_file, desc):
    if os.path.exists(out_file):
        arr = np.load(out_file)
        if len(arr) == len(df):
            print(f"{desc} already fully extracted ({len(arr)} samples).")
            return arr

    features = []
    t0 = time.time()
    total = len(df)
    
    for idx, row in df.iterrows():
        fname = row["filename"]
        audio_path = os.path.join(audio_dir, fname)
        
        try:
            y, sr = sf.read(audio_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            
            # Extract central 8 seconds of speech (from 2s to 10s or available window)
            dur_samples = len(y)
            start = int(min(sr * 2.0, max(0, dur_samples * 0.1)))
            end = int(min(dur_samples, start + sr * 8.0))
            
            chunk = y[start:end]
            if len(chunk) < sr * 1.0:
                chunk = y  # fallback to entire audio
                
            inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        except Exception as e:
            print(f"Error {fname}: {e}")
            emb = np.zeros(768, dtype=np.float32)
            
        features.append(emb)
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 0.1)
            eta = (total - (idx + 1)) / max(rate, 0.01)
            print(f"[{desc}] {idx + 1}/{total} done ({elapsed:.1f}s, {rate:.2f} it/s, ETA: {eta:.0f}s)")
            np.save(out_file, np.array(features, dtype=np.float32))
            
    res = np.array(features, dtype=np.float32)
    np.save(out_file, res)
    print(f"[{desc}] Complete! Saved {res.shape} to {out_file}")
    return res

print("\n--- Extracting Test Wav2Vec2 Features (216 samples) ---")
extract_features_for_df(test_df, os.path.join(DATA_DIR, "test"), test_out, "TEST")

print("\n--- Extracting Train Wav2Vec2 Features (769 samples) ---")
extract_features_for_df(train_df, os.path.join(DATA_DIR, "train"), train_out, "TRAIN")

print("All Wav2Vec2 features extracted and saved!")

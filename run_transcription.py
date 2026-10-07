import os
import sys
import time
import pandas as pd
import whisper

BASE_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026"
DATA_DIR = os.path.join(BASE_DIR, "Dataset_Final")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_csv_path = os.path.join(DATA_DIR, "test.csv")
train_csv_path = os.path.join(DATA_DIR, "train.csv")

test_df = pd.read_csv(test_csv_path)
train_df = pd.read_csv(train_csv_path)

test_audio_dir = os.path.join(DATA_DIR, "test")
train_audio_dir = os.path.join(DATA_DIR, "train")

test_out_path = os.path.join(OUTPUT_DIR, "test_transcripts.csv")
train_out_path = os.path.join(OUTPUT_DIR, "train_transcripts.csv")

print("Loading Whisper tiny.en model...")
model = whisper.load_model("tiny.en")
print("Whisper model loaded.")

def transcribe_list(df, audio_dir, out_path, desc):
    records = []
    done_filenames = set()
    if os.path.exists(out_path):
        existing_df = pd.read_csv(out_path)
        records = existing_df.to_dict("records")
        done_filenames = set(existing_df["filename"].tolist())
        print(f"Resuming {desc}: {len(records)} already done.")
    
    total = len(df)
    t0 = time.time()
    for idx, row in df.iterrows():
        fn = row["filename"]
        if fn in done_filenames:
            continue
        audio_path = os.path.join(audio_dir, fn)
        try:
            res = model.transcribe(
                audio_path,
                language="en",
                beam_size=1,
                best_of=1,
                condition_on_previous_text=False
            )
            text = res.get("text", "").strip()
        except Exception as e:
            print(f"Error {fn}: {e}")
            text = ""
        records.append({"filename": fn, "transcript": text})
        
        if len(records) % 15 == 0 or len(records) == total:
            pd.DataFrame(records).to_csv(out_path, index=False)
            elapsed = time.time() - t0
            print(f"{desc} progress: {len(records)}/{total} ({round(elapsed, 1)}s)")
    
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"{desc} completed! Saved to {out_path}")

print("Starting test transcription (216 files)...")
transcribe_list(test_df, test_audio_dir, test_out_path, "TEST")

print("Starting train transcription (769 files)...")
transcribe_list(train_df, train_audio_dir, train_out_path, "TRAIN")

print("All transcriptions finished successfully!")

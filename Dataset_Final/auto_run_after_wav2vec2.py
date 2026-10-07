import os
import time
import subprocess

DATA_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026\Dataset_Final"
train_w2v = os.path.join(DATA_DIR, "train_wav2vec2_features.npy")

print("Waiting for Wav2Vec2 train extraction to complete (769 samples)...")
while True:
    if os.path.exists(train_w2v):
        size = os.path.getsize(train_w2v)
        samples = (size - 128) // (768 * 4)
        print(f"Current progress: {samples}/769 ({samples/769*100:.1f}%)")
        if samples >= 769:
            print("Wav2Vec2 extraction finished! Starting master pipeline training with full phonetic representations...")
            break
    time.sleep(20)

subprocess.run(["python", "train_master_pipeline.py"], cwd=DATA_DIR)
print("Pipeline run complete!")

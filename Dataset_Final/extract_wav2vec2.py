import pandas as pd
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import os
from tqdm import tqdm

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

def extract(csv_file, audio_dir, out_file):
    df = pd.read_csv(csv_file)
    features = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fname = row['filename']
        audio_path = os.path.join(audio_dir, fname)
        
        speech, sr = librosa.load(audio_path, sr=16000)
        
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        last_hidden_states = outputs.last_hidden_state
        avg_hidden = last_hidden_states.mean(dim=1).squeeze().numpy()
        features.append(avg_hidden)
        
    np.save(out_file, np.array(features))

extract('train.csv', 'train', 'train_wav2vec2_features.npy')
extract('test.csv', 'test', 'test_wav2vec2_features.npy')
print("Saved all wav2vec2 features")

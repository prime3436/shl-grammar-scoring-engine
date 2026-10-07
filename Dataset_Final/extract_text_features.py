import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-mpnet-base-v2')

train_df = pd.read_csv('train_transcripts.csv')
test_df = pd.read_csv('test_transcripts.csv')

# Handle missing transcripts
train_texts = train_df['transcript'].fillna("").tolist()
test_texts = test_df['transcript'].fillna("").tolist()

train_emb = model.encode(train_texts, show_progress_bar=True)
test_emb = model.encode(test_texts, show_progress_bar=True)

np.save('train_text_features.npy', train_emb)
np.save('test_text_features.npy', test_emb)
print("Saved all text features")

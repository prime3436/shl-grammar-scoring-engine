import pandas as pd
import numpy as np
import librosa
import os
from sentence_transformers import SentenceTransformer
import textstat
import nltk
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # basic
        duration = librosa.get_duration(y=y, sr=sr)
        
        # mfcc
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # spectral
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # rms
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_mean = np.mean(mel, axis=1)
        
        # Pause detection
        non_mute_intervals = librosa.effects.split(y, top_db=20)
        total_speech_duration = np.sum([intv[1] - intv[0] for intv in non_mute_intervals]) / sr
        speech_ratio = total_speech_duration / duration if duration > 0 else 0
        pause_count = len(non_mute_intervals) - 1
        
        features = {
            'duration': duration,
            'cent_mean': cent_mean,
            'cent_std': cent_std,
            'rolloff_mean': rolloff_mean,
            'rolloff_std': rolloff_std,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'speech_ratio': speech_ratio,
            'pause_count': pause_count
        }
        for i in range(20):
            features[f'mfcc_{i}_mean'] = mfcc_mean[i]
            features[f'mfcc_{i}_std'] = mfcc_std[i]
            
        for i in range(128):
            features[f'mel_{i}_mean'] = mel_mean[i]
            
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_text_features(df, text_col='transcript'):
    # Sentence Transformer embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df[text_col].fillna('').tolist(), show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
    
    # Textstat features
    df['flesch_reading_ease'] = df[text_col].apply(lambda x: textstat.flesch_reading_ease(str(x)))
    df['smog_index'] = df[text_col].apply(lambda x: textstat.smog_index(str(x)))
    df['flesch_kincaid_grade'] = df[text_col].apply(lambda x: textstat.flesch_kincaid_grade(str(x)))
    df['coleman_liau_index'] = df[text_col].apply(lambda x: textstat.coleman_liau_index(str(x)))
    df['automated_readability_index'] = df[text_col].apply(lambda x: textstat.automated_readability_index(str(x)))
    df['dale_chall_readability_score'] = df[text_col].apply(lambda x: textstat.dale_chall_readability_score(str(x)))
    df['difficult_words'] = df[text_col].apply(lambda x: textstat.difficult_words(str(x)))
    df['linsear_write_formula'] = df[text_col].apply(lambda x: textstat.linsear_write_formula(str(x)))
    df['gunning_fog'] = df[text_col].apply(lambda x: textstat.gunning_fog(str(x)))
    df['text_standard'] = df[text_col].apply(lambda x: textstat.text_standard(str(x), float_output=True))
    
    # Basic features
    df['char_count'] = df[text_col].apply(lambda x: len(str(x)))
    df['word_count'] = df[text_col].apply(lambda x: len(str(x).split()))
    df['sentence_count'] = df[text_col].apply(lambda x: textstat.sentence_count(str(x)))
    df['avg_word_length'] = df['char_count'] / (df['word_count'] + 1e-5)
    df['avg_sentence_length'] = df['word_count'] / (df['sentence_count'] + 1e-5)
    
    # POS tagging
    def get_pos_counts(text):
        tokens = nltk.word_tokenize(str(text))
        pos_tags = nltk.pos_tag(tokens)
        counts = Counter(tag for word, tag in pos_tags)
        return counts

    pos_counts_list = []
    for text in tqdm(df[text_col], desc="POS tagging"):
        pos_counts_list.append(get_pos_counts(text))
        
    pos_df = pd.DataFrame(pos_counts_list).fillna(0)
    pos_df.columns = [f'pos_{col}' for col in pos_df.columns]
    
    # Combine
    result_df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True), pos_df.reset_index(drop=True)], axis=1)
    return result_df

# Read data
base_dir = 'C:/Users/HP/Downloads/shl-hiring-assessment-2026/Dataset_Final'
train_labels = pd.read_csv(os.path.join(base_dir, 'train.csv'))
train_transcripts = pd.read_csv(os.path.join(base_dir, 'train_transcripts.csv'))
test_labels = pd.read_csv(os.path.join(base_dir, 'test.csv'))
test_transcripts = pd.read_csv(os.path.join(base_dir, 'test_transcripts.csv'))

train = pd.merge(train_labels, train_transcripts, on='filename')
test = pd.merge(test_labels, test_transcripts, on='filename', how='left')

print("Extracting text features for train...")
train = extract_text_features(train)
print("Extracting text features for test...")
test = extract_text_features(test)

# Audio features
print("Extracting audio features for train...")
train_audio_feats = []
for file in tqdm(train['filename']):
    feat = extract_audio_features(os.path.join(base_dir, 'train', file))
    train_audio_feats.append(feat)
train_audio_df = pd.DataFrame(train_audio_feats)
train = pd.concat([train.reset_index(drop=True), train_audio_df.reset_index(drop=True)], axis=1)

print("Extracting audio features for test...")
test_audio_feats = []
for file in tqdm(test['filename']):
    feat = extract_audio_features(os.path.join(base_dir, 'test', file))
    test_audio_feats.append(feat)
test_audio_df = pd.DataFrame(test_audio_feats)
test = pd.concat([test.reset_index(drop=True), test_audio_df.reset_index(drop=True)], axis=1)

# Adding rate of speech
train['speaking_rate'] = train['word_count'] / (train['duration'] + 1e-5)
test['speaking_rate'] = test['word_count'] / (test['duration'] + 1e-5)

train.to_csv(os.path.join(base_dir, 'train_features.csv'), index=False)
test.to_csv(os.path.join(base_dir, 'test_features.csv'), index=False)
print("Saved features!")

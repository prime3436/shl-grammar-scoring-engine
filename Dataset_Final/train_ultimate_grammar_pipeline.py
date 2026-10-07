import os
import re
import sys
import time
import numpy as np
import pandas as pd
import textstat
import nltk
from nltk import pos_tag, word_tokenize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.stats import pearsonr
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

BASE_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026"
DATA_DIR = os.path.join(BASE_DIR, "Dataset_Final")

print("="*65)
print("   SHL 2026 GRAMMAR SCORING - ULTIMATE CoLA + POS + MULTI-MODAL PIPELINE")
print("="*65)

# 1. Load Data
train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_features.csv'))
test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_features.csv'))
train_txt = np.load(os.path.join(DATA_DIR, 'train_text_features.npy'))
test_txt = np.load(os.path.join(DATA_DIR, 'test_text_features.npy'))

train_cola = pd.read_csv(os.path.join(DATA_DIR, 'train_cola_features.csv'))
test_cola = pd.read_csv(os.path.join(DATA_DIR, 'test_cola_features.csv'))

# Filter normal samples for speech scoring
mask = train_df['label'] > 0
train_df_nz = train_df[mask].reset_index(drop=True)
train_txt_nz = train_txt[mask]
train_cola_nz = train_cola[mask].reset_index(drop=True)
y_nz = train_df_nz['label'].values

print(f"Total training speech samples: {len(y_nz)}")
print(f"Target stats -> Mean: {y_nz.mean():.4f}, Std: {y_nz.std():.4f}, Min: {y_nz.min():.1f}, Max: {y_nz.max():.1f}")

# 2. Rich Linguistic, Syntactic & POS Feature Extraction
SUBORD_CONJ = {'although', 'because', 'since', 'unless', 'whereas', 'while', 'if', 'as', 'though', 'even though', 'provided that'}
COORD_CONJ = {'and', 'but', 'or', 'so', 'yet', 'for', 'nor'}
MODALS = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
FILLERS = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'actually', 'basically', 'so', 'well', 'okay', 'right'}
PREPOSITIONS = {'in', 'on', 'at', 'to', 'for', 'with', 'from', 'by', 'about', 'as', 'into', 'like', 'through', 'after', 'over', 'between', 'out', 'against', 'during', 'without', 'before', 'under', 'around', 'among'}
FORMAL_TRANSITIONS = {'furthermore', 'moreover', 'nevertheless', 'consequently', 'subsequently', 'therefore', 'however', 'in addition', 'on the other hand', 'in contrast', 'as a result', 'for instance', 'for example'}

def extract_advanced_linguistics(df):
    records = []
    for idx, row in df.iterrows():
        raw_text = str(row['transcript'])
        text = raw_text.lower()
        words = re.findall(r'\b[a-z]+\b', text)
        N = len(words)
        V = len(set(words))
        
        # Word counts & hapax legomena
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        hapax = sum(1 for w, c in word_counts.items() if c == 1)
        
        # Lexical diversity metrics
        root_ttr = V / np.sqrt(max(N, 1))
        log_ttr = np.log(max(V, 1)) / np.log(max(N, 2))
        maas = (np.log(max(N, 2)) - np.log(max(V, 1))) / (np.log(max(N, 2)) ** 2 + 1e-5)
        honore = 100 * np.log(max(N, 2)) / (1 - (hapax / max(V, 1)) + 1e-4)
        
        # Word length distribution
        word_lens = [len(w) for w in words] if words else [0]
        mean_word_len = np.mean(word_lens)
        std_word_len = np.std(word_lens) if len(word_lens) > 1 else 0
        long_words = sum(1 for w in words if len(w) >= 7)
        very_long_words = sum(1 for w in words if len(w) >= 9)
        
        # Syntactic complexity
        subord_count = sum(1 for w in words if w in SUBORD_CONJ)
        coord_count = sum(1 for w in words if w in COORD_CONJ)
        modal_count = sum(1 for w in words if w in MODALS)
        prep_count = sum(1 for w in words if w in PREPOSITIONS)
        
        # Formal transitions
        formal_trans_count = sum(1 for t in FORMAL_TRANSITIONS if t in text)
        
        # Fluency & hesitations
        filler_count = sum(1 for w in words if w in FILLERS)
        rep_count = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
        
        # Sentence structure
        sents = [s.strip() for s in raw_text.split('.') if s.strip()]
        sent_lens = [len(s.split()) for s in sents] if sents else [0]
        mean_sent_len = np.mean(sent_lens)
        std_sent_len = np.std(sent_lens) if len(sent_lens) > 1 else 0
        
        # Part-of-Speech analysis via NLTK
        tokens = word_tokenize(raw_text) if raw_text.strip() else []
        tagged = pos_tag(tokens) if tokens else []
        total_tags = max(len(tagged), 1)
        
        nouns = sum(1 for _, tag in tagged if tag.startswith('NN'))
        verbs = sum(1 for _, tag in tagged if tag.startswith('VB'))
        adjs = sum(1 for _, tag in tagged if tag.startswith('JJ'))
        advs = sum(1 for _, tag in tagged if tag.startswith('RB'))
        vbn = sum(1 for _, tag in tagged if tag == 'VBN') # Past participles ("have spoken", "was created")
        vbg = sum(1 for _, tag in tagged if tag == 'VBG') # Gerunds/present participles ("speaking", "creating")
        
        # Timing & syllables
        dur = max(row['duration'], 0.5)
        speech_dur = max(dur * row.get('speech_ratio', 0.5), 0.1)
        syllables = textstat.syllable_count(text)
        
        records.append({
            'root_ttr': root_ttr,
            'log_ttr': log_ttr,
            'maas': maas,
            'honore': honore,
            'mean_word_len': mean_word_len,
            'std_word_len': std_word_len,
            'long_words': long_words,
            'long_word_ratio': long_words / max(N, 1),
            'very_long_words': very_long_words,
            'very_long_word_ratio': very_long_words / max(N, 1),
            'subord_count': subord_count,
            'subord_ratio': subord_count / max(N, 1),
            'coord_count': coord_count,
            'coord_ratio': coord_count / max(N, 1),
            'modal_count': modal_count,
            'modal_ratio': modal_count / max(N, 1),
            'prep_count': prep_count,
            'prep_ratio': prep_count / max(N, 1),
            'formal_trans_count': formal_trans_count,
            'filler_count': filler_count,
            'filler_ratio': filler_count / max(N, 1),
            'rep_count': rep_count,
            'mean_sent_len': mean_sent_len,
            'std_sent_len': std_sent_len,
            'noun_ratio': nouns / total_tags,
            'verb_ratio': verbs / total_tags,
            'adj_ratio': adjs / total_tags,
            'adv_ratio': advs / total_tags,
            'vbn_ratio': vbn / max(verbs, 1),
            'vbg_ratio': vbg / max(verbs, 1),
            'noun_verb_ratio': nouns / max(verbs, 1),
            'syllable_rate': syllables / dur,
            'articulation_rate': syllables / speech_dur,
            'syllables_per_word': syllables / max(N, 1),
            'vocal_dynamic_1': row.get('mfcc_1_std', 0) / (abs(row.get('mfcc_0_std', 0)) + 1e-4),
            'vocal_dynamic_2': row.get('mfcc_2_std', 0) / (abs(row.get('mfcc_1_std', 0)) + 1e-4),
            'spectral_contrast': row.get('rolloff_std', 0) / (row.get('rolloff_mean', 1) + 1e-4)
        })
    return pd.DataFrame(records)

print("Extracting advanced linguistic, syntactic & POS features...")
train_ling = extract_advanced_linguistics(train_df_nz)
test_ling = extract_advanced_linguistics(test_df)

# Base audio features from tabular
base_feats = [c for c in train_df_nz.columns if c in test_df.columns and c not in ['filename', 'transcript', 'label'] and not c.startswith('emb_')]
X_base_tr = train_df_nz[base_feats].fillna(0).values
X_base_te = test_df[base_feats].fillna(0).values

# RoBERTa-CoLA Features
cola_cols = [c for c in train_cola.columns if c != 'filename']
X_cola_tr = train_cola_nz[cola_cols].fillna(0).values
X_cola_te = test_cola[cola_cols].fillna(0).values
print(f"Loaded {len(cola_cols)} CoLA grammatical acceptability features.")

# Word & Char TFIDF + SVD
train_texts = train_df_nz['transcript'].fillna('').tolist()
test_texts = test_df['transcript'].fillna('').tolist()

print("Extracting multi-level TF-IDF representations...")
word_tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=3000)
X_w_tr = word_tfidf.fit_transform(train_texts)
X_w_te = word_tfidf.transform(test_texts)
svd_w = TruncatedSVD(n_components=24, random_state=42)
W_tr = svd_w.fit_transform(X_w_tr)
W_te = svd_w.transform(X_w_te)

char_tfidf = TfidfVectorizer(ngram_range=(3, 5), analyzer='char', min_df=3, max_features=3000)
X_c_tr = char_tfidf.fit_transform(train_texts)
X_c_te = char_tfidf.transform(test_texts)
svd_c = TruncatedSVD(n_components=16, random_state=42)
C_tr = svd_c.fit_transform(X_c_tr)
C_te = svd_c.transform(X_c_te)

# Sentence Transformers SVD
emb_cols = [c for c in train_df_nz.columns if c.startswith('emb_')]
X_minilm_tr = train_df_nz[emb_cols].fillna(0).values
X_minilm_te = test_df[emb_cols].fillna(0).values

X_all_emb_tr = np.hstack([X_minilm_tr, train_txt_nz])
X_all_emb_te = np.hstack([X_minilm_te, test_txt])

svd_emb = TruncatedSVD(n_components=48, random_state=42)
E_tr = svd_emb.fit_transform(X_all_emb_tr)
E_te = svd_emb.transform(X_all_emb_te)

# Wav2Vec2 phonetic embeddings
w2v_tr_path = os.path.join(DATA_DIR, "train_wav2vec2_features.npy")
w2v_te_path = os.path.join(DATA_DIR, "test_wav2vec2_features.npy")

feature_blocks_tr = [X_base_tr, train_ling.values, X_cola_tr, W_tr, C_tr, E_tr]
feature_blocks_te = [X_base_te, test_ling.values, X_cola_te, W_te, C_te, E_te]

if os.path.exists(w2v_tr_path) and os.path.exists(w2v_te_path):
    print("Found Wav2Vec2 phonetic embeddings! Loading...")
    w2v_tr = np.load(w2v_tr_path)
    w2v_te = np.load(w2v_te_path)
    w2v_tr_nz = w2v_tr[mask]
    svd_w2v = TruncatedSVD(n_components=32, random_state=42)
    W2V_tr = svd_w2v.fit_transform(w2v_tr_nz)
    W2V_te = svd_w2v.transform(w2v_te)
    feature_blocks_tr.append(W2V_tr)
    feature_blocks_te.append(W2V_te)
    print(f"Added Wav2Vec2 SVD components (32 dimensions, explained variance: {svd_w2v.explained_variance_ratio_.sum():.3f})")

X_tr = np.hstack(feature_blocks_tr)
X_te = np.hstack(feature_blocks_te)
print(f"Final feature space: {X_tr.shape[1]} features across {len(y_nz)} samples.")

# 3. Model Training & Cross-Validation
FOLDS = 5
y_bins = pd.qcut(y_nz, q=FOLDS, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

models = {
    'Ridge': Ridge(alpha=30.0),
    'BayesianRidge': BayesianRidge(),
    'ElasticNet': ElasticNet(alpha=0.02, l1_ratio=0.5, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=8, min_samples_split=6, random_state=42, n_jobs=-1),
    'LGBM': lgb.LGBMRegressor(n_estimators=600, learning_rate=0.018, num_leaves=16, colsample_bytree=0.35, subsample=0.75, reg_alpha=1.5, reg_lambda=3.5, random_state=42, n_jobs=-1, verbose=-1),
    'CatBoost': CatBoostRegressor(iterations=700, learning_rate=0.018, depth=5, l2_leaf_reg=4.5, random_state=42, verbose=0),
    'XGBoost': xgb.XGBRegressor(n_estimators=600, learning_rate=0.018, max_depth=4, subsample=0.75, colsample_bytree=0.35, reg_alpha=1.5, reg_lambda=3.5, random_state=42, n_jobs=-1, verbosity=0)
}

oofs = {}
preds = {}

print("\n--- Training 7-Model Cross-Validated Regressors ---")
for name, m in models.items():
    oof = np.zeros(len(y_nz))
    pred = np.zeros(len(X_te))
    mat_tr = X_tr_s if name in ['Ridge', 'BayesianRidge', 'ElasticNet'] else X_tr
    mat_te = X_te_s if name in ['Ridge', 'BayesianRidge', 'ElasticNet'] else X_te
    
    for tr, va in skf.split(mat_tr, y_bins):
        m.fit(mat_tr[tr], y_nz[tr])
        oof[va] = m.predict(mat_tr[va])
        pred += m.predict(mat_te) / FOLDS
        
    oofs[name] = oof
    preds[name] = pred
    rmse = root_mean_squared_error(y_nz, oof)
    pr = pearsonr(y_nz, oof)[0]
    print(f"  {name:15s} -> CV RMSE: {rmse:.4f} | Pearson: {pr:.4f}")

# 4. Optimal SLSQP Weight Optimization
def obj(w):
    p = sum(w[i] * oofs[name] for i, name in enumerate(models.keys()))
    return root_mean_squared_error(y_nz, p)

w0 = [1.0/len(models)] * len(models)
res = minimize(obj, w0, bounds=[(0, 1)]*len(models))
w_opt = res.x / np.sum(res.x)

ens_oof = sum(w_opt[i] * oofs[name] for i, name in enumerate(models.keys()))
ens_test = sum(w_opt[i] * preds[name] for i, name in enumerate(models.keys()))

print("\n--- Optimal Ensemble Configuration ---")
for name, w in zip(models.keys(), w_opt):
    print(f"  {name:15s} weight: {w:.4f}")

raw_rmse = root_mean_squared_error(y_nz, ens_oof)
raw_pr = pearsonr(y_nz, ens_oof)[0]
print(f"New Ensemble Raw OOF RMSE: {raw_rmse:.4f} | Pearson: {raw_pr:.4f}")

# 5. Optimal Linear Calibration & Dynamic Range Alignment
a_opt = raw_pr * (np.std(y_nz) / np.std(ens_oof))
b_opt = np.mean(y_nz) - a_opt * np.mean(ens_oof)

cal_oof = a_opt * ens_oof + b_opt
cal_test = a_opt * ens_test + b_opt
cal_test = np.clip(cal_test, 1.0, 5.0)

cal_rmse = root_mean_squared_error(y_nz, cal_oof)
print(f"New Ensemble Calibrated OOF RMSE: {cal_rmse:.4f} (Slope: {a_opt:.3f}, Intercept: {b_opt:.3f})")
print(f"New Predictions -> Mean: {cal_test.mean():.4f}, Std: {cal_test.std():.4f}, Min: {cal_test.min():.4f}, Max: {cal_test.max():.4f}")

# 6. Orthogonal Ensembling with Previous Strong Submissions
# Load previous submissions for safe multi-system blending
prev_sub_path = os.path.join(DATA_DIR, "submission_master_ensemble.csv") # got ~0.40 on LB
old_sub_path = os.path.join(BASE_DIR, "final_submission.csv")           # got 0.5073 on LB

prev_preds = None
old_preds = None
if os.path.exists(prev_sub_path):
    prev_preds = pd.read_csv(prev_sub_path)['label'].values
    print(f"\nLoaded previous ~0.40 submission. Correlation with new: {np.corrcoef(cal_test, prev_preds)[0,1]:.4f}")

if os.path.exists(old_sub_path):
    old_preds = pd.read_csv(old_sub_path)['label'].values
    print(f"Loaded old 0.5073 submission. Correlation with new: {np.corrcoef(cal_test, old_preds)[0,1]:.4f}")

# Multi-Architecture Blended Submissions:
# Variant A: Pure New CoLA-enhanced master ensemble
sub_cola_only = cal_test

# Variant B: 75% New CoLA + 25% Previous ~0.40 Master (Orthogonal Error Cancellation)
if prev_preds is not None:
    sub_blend_prev = 0.75 * cal_test + 0.25 * prev_preds
else:
    sub_blend_prev = cal_test

# Variant C: 70% New CoLA + 20% Previous ~0.40 Master + 10% Old Baseline (Full Trinity Ensemble)
if prev_preds is not None and old_preds is not None:
    sub_trinity = 0.70 * cal_test + 0.20 * prev_preds + 0.10 * old_preds
else:
    sub_trinity = sub_blend_prev

# Save all candidate submissions
sub_df = pd.DataFrame({'filename': test_df['filename'], 'label': sub_blend_prev})
sub_df.to_csv(os.path.join(DATA_DIR, 'submission_cola_blend.csv'), index=False)
sub_df.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
sub_df.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)

pd.DataFrame({'filename': test_df['filename'], 'label': sub_cola_only}).to_csv(os.path.join(DATA_DIR, 'submission_cola_pure.csv'), index=False)
pd.DataFrame({'filename': test_df['filename'], 'label': sub_trinity}).to_csv(os.path.join(DATA_DIR, 'submission_trinity.csv'), index=False)

print("\n" + "="*65)
print("SUCCESSFULLY CREATED CANDIDATE SUBMISSIONS:")
print(f"  1. [MAIN / ACTIVE] submission.csv (75% CoLA Master + 25% Previous Master)")
print(f"  2. submission_cola_pure.csv (100% CoLA Master)")
print(f"  3. submission_trinity.csv (70% CoLA + 20% Prev + 10% Old)")
print("="*65)
print(sub_df.head(15))

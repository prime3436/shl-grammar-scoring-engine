import os
import re
import sys
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

BASE_DIR = r"C:\Users\HP\Downloads\shl-hiring-assessment-2026"
DATA_DIR = os.path.join(BASE_DIR, "Dataset_Final")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_csv_path = os.path.join(DATA_DIR, "train.csv")
test_csv_path = os.path.join(DATA_DIR, "test.csv")
train_audio_dir = os.path.join(DATA_DIR, "train")
test_audio_dir = os.path.join(DATA_DIR, "test")

train_trans_path = os.path.join(OUTPUT_DIR, "train_transcripts.csv")
test_trans_path = os.path.join(OUTPUT_DIR, "test_transcripts.csv")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
train_trans_df = pd.read_csv(train_trans_path).fillna({"transcript": ""})
test_trans_df = pd.read_csv(test_trans_path).fillna({"transcript": ""})

train_df = train_df.merge(train_trans_df, on="filename", how="left").fillna({"transcript": ""})
test_df = test_df.merge(test_trans_df, on="filename", how="left").fillna({"transcript": ""})

def extract_acoustic(audio_path):
    try:
        y, sr = sf.read(audio_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if len(y) > 45 * sr:
            y = y[:45 * sr]
    except Exception:
        return None

    feats = {}
    dur = max(len(y) / sr, 0.5)
    feats["duration"] = dur

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
    for i in range(len(mfcc)):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i}_std"] = float(np.std(mfcc[i]))
        feats[f"mfcc_{i}_max"] = float(np.max(mfcc[i]))
        feats[f"mfcc_{i}_min"] = float(np.min(mfcc[i]))

    d_mfcc = librosa.feature.delta(mfcc)
    for i in range(len(d_mfcc)):
        feats[f"d_mfcc_{i}_mean"] = float(np.mean(d_mfcc[i]))
        feats[f"d_mfcc_{i}_std"] = float(np.std(d_mfcc[i]))

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    s_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    s_flat = librosa.feature.spectral_flatness(y=y)[0]

    feats["spec_cent_mean"] = float(np.mean(sc))
    feats["spec_cent_std"] = float(np.std(sc))
    feats["spec_bw_mean"] = float(np.mean(sb))
    feats["spec_bw_std"] = float(np.std(sb))
    feats["spec_roll_mean"] = float(np.mean(s_roll))
    feats["spec_roll_std"] = float(np.std(s_roll))
    feats["spec_flat_mean"] = float(np.mean(s_flat))
    feats["spec_flat_std"] = float(np.std(s_flat))

    rms = librosa.feature.rms(y=y)[0]
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"] = float(np.std(rms))
    feats["rms_max"] = float(np.max(rms))
    feats["rms_energy"] = float(np.sum(rms ** 2) / dur)

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"] = float(np.std(zcr))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(len(chroma)):
        feats[f"chroma_{i}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i}_std"] = float(np.std(chroma[i]))

    try:
        intervals = librosa.effects.split(y, top_db=25)
        active_len = sum(end - start for start, end in intervals) / sr
        feats["active_speech_ratio"] = float(active_len / dur)
        feats["pause_ratio"] = float(1.0 - (active_len / dur))
        feats["pause_count"] = float(len(intervals))
        feats["pause_rate"] = float(len(intervals) / dur)
    except Exception:
        feats["active_speech_ratio"] = 1.0
        feats["pause_ratio"] = 0.0
        feats["pause_count"] = 0.0
        feats["pause_rate"] = 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    feats["onset_mean"] = float(np.mean(onset_env))
    feats["onset_std"] = float(np.std(onset_env))

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    except Exception:
        feats["tempo"] = 0.0

    return feats

def get_acoustic_df(df, audio_dir, cache_path):
    if os.path.exists(cache_path):
        print(f"Loading cached acoustics from {cache_path}")
        return pd.read_csv(cache_path)
    print(f"Extracting acoustics for {len(df)} files...")
    recs = []
    t0 = time.time()
    for idx, row in df.iterrows():
        fn = row["filename"]
        p = os.path.join(audio_dir, fn)
        f = extract_acoustic(p)
        if f is None:
            f = {}
        f["filename"] = fn
        recs.append(f)
        if (idx + 1) % 50 == 0:
            print(f"Acoustic progress: {idx + 1}/{len(df)} ({round(time.time() - t0, 1)}s)")
    res_df = pd.DataFrame(recs)
    res_df.to_csv(cache_path, index=False)
    print(f"Saved acoustics to {cache_path}")
    return res_df

train_ac_path = os.path.join(OUTPUT_DIR, "train_acoustic_feats.csv")
test_ac_path = os.path.join(OUTPUT_DIR, "test_acoustic_feats.csv")

train_ac_df = get_acoustic_df(train_df, train_audio_dir, train_ac_path)
test_ac_df = get_acoustic_df(test_df, test_audio_dir, test_ac_path)

subord_conjs = {
    "because", "although", "though", "even", "since", "while", "whereas",
    "if", "unless", "until", "despite", "whether", "after", "before", "once"
}
coord_conjs = {"and", "but", "or", "nor", "for", "yet", "so"}
rel_pronouns = {"which", "that", "who", "whom", "whose", "where", "when", "why"}
modal_verbs = {"can", "could", "may", "might", "shall", "should", "will", "would", "must"}
fillers = {"uh", "um", "ah", "er", "like"}

def extract_text_features(text, duration=45.0, active_ratio=0.8, pause_count=10.0):
    text_clean = str(text).strip()
    words = re.findall(r"\b[A-Za-z']+\b", text_clean.lower())
    n_words = len(words)
    char_len = len(text_clean)

    f = {}
    f["char_count"] = char_len
    f["word_count"] = n_words
    f["is_empty_or_punct"] = 1.0 if (n_words < 3 or text_clean in ["", ".", "...", "!", "?"]) else 0.0

    if n_words > 0:
        w_lens = [len(w) for w in words]
        f["avg_word_len"] = float(np.mean(w_lens))
        f["long_word_ratio"] = float(sum(1 for l in w_lens if l >= 7) / n_words)
        f["very_long_word_ratio"] = float(sum(1 for l in w_lens if l >= 10) / n_words)
        unique_words = set(words)
        f["ttr"] = float(len(unique_words) / n_words)
        f["root_ttr"] = float(len(unique_words) / np.sqrt(n_words))
        word_freq = pd.Series(words).value_counts()
        f["hapax_ratio"] = float((word_freq == 1).sum() / n_words)
        f["subord_conj_count"] = float(sum(1 for w in words if w in subord_conjs))
        f["subord_conj_ratio"] = float(f["subord_conj_count"] / n_words)
        f["coord_conj_count"] = float(sum(1 for w in words if w in coord_conjs))
        f["coord_conj_ratio"] = float(f["coord_conj_count"] / n_words)
        f["rel_pronoun_count"] = float(sum(1 for w in words if w in rel_pronouns))
        f["rel_pronoun_ratio"] = float(f["rel_pronoun_count"] / n_words)
        f["modal_verb_count"] = float(sum(1 for w in words if w in modal_verbs))
        f["modal_verb_ratio"] = float(f["modal_verb_count"] / n_words)
        f["filler_count"] = float(sum(1 for w in words if w in fillers))
        f["filler_ratio"] = float(f["filler_count"] / n_words)
        stutters = sum(1 for i in range(len(words) - 1) if words[i] == words[i + 1])
        f["stutter_count"] = float(stutters)
        f["stutter_ratio"] = float(stutters / n_words)
    else:
        f["avg_word_len"] = 0.0
        f["long_word_ratio"] = 0.0
        f["very_long_word_ratio"] = 0.0
        f["ttr"] = 0.0
        f["root_ttr"] = 0.0
        f["hapax_ratio"] = 0.0
        f["subord_conj_count"] = 0.0
        f["subord_conj_ratio"] = 0.0
        f["coord_conj_count"] = 0.0
        f["coord_conj_ratio"] = 0.0
        f["rel_pronoun_count"] = 0.0
        f["rel_pronoun_ratio"] = 0.0
        f["modal_verb_count"] = 0.0
        f["modal_verb_ratio"] = 0.0
        f["filler_count"] = 0.0
        f["filler_ratio"] = 0.0
        f["stutter_count"] = 0.0
        f["stutter_ratio"] = 0.0

    sentences = [s.strip() for s in re.split(r"[.!?]+", text_clean) if len(s.strip()) > 0]
    f["sentence_count"] = float(len(sentences))
    f["avg_sentence_len"] = float(n_words / max(len(sentences), 1))
    f["comma_count"] = float(text_clean.count(","))
    f["comma_density"] = float(f["comma_count"] / max(n_words, 1))

    dur_val = max(duration, 1.0)
    f["speech_rate"] = float(n_words / dur_val)
    f["articulation_rate"] = float(n_words / max(dur_val * active_ratio, 0.5))
    f["words_per_pause"] = float(n_words / max(pause_count, 1.0))

    return f

def build_feature_table(meta_df, ac_df):
    merged = meta_df.merge(ac_df, on="filename", how="left")
    t_recs = []
    for _, row in merged.iterrows():
        dur = row.get("duration", 45.0)
        act = row.get("active_speech_ratio", 0.8)
        p_cnt = row.get("pause_count", 10.0)
        txt = row.get("transcript", "")
        t_f = extract_text_features(txt, duration=dur, active_ratio=act, pause_count=p_cnt)
        t_recs.append(t_f)
    t_df = pd.DataFrame(t_recs)
    for col in t_df.columns:
        merged[col] = t_df[col].values
    return merged

print("Building feature tables...")
train_full = build_feature_table(train_df, train_ac_df)
test_full = build_feature_table(test_df, test_ac_df)

train_texts = train_full["transcript"].fillna("").tolist()
test_texts = test_full["transcript"].fillna("").tolist()

print("Extracting TF-IDF and SVD components...")
word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=1500)
X_word_tr = word_vec.fit_transform(train_texts)
X_word_te = word_vec.transform(test_texts)

svd_word = TruncatedSVD(n_components=16, random_state=42)
W_tr = svd_word.fit_transform(X_word_tr)
W_te = svd_word.transform(X_word_te)

for i in range(16):
    train_full[f"svd_word_{i}"] = W_tr[:, i]
    test_full[f"svd_word_{i}"] = W_te[:, i]

char_vec = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", min_df=3, max_features=2000)
X_char_tr = char_vec.fit_transform(train_texts)
X_char_te = char_vec.transform(test_texts)

svd_char = TruncatedSVD(n_components=12, random_state=42)
C_tr = svd_char.fit_transform(X_char_tr)
C_te = svd_char.transform(X_char_te)

for i in range(12):
    train_full[f"svd_char_{i}"] = C_tr[:, i]
    test_full[f"svd_char_{i}"] = C_te[:, i]

drop_cols = ["filename", "label", "transcript"]
feature_cols = [c for c in train_full.columns if c not in drop_cols and c in test_full.columns]
print(f"Total features used: {len(feature_cols)}")

X_train = train_full[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
y_train = train_full["label"].values
X_test = test_full[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

FOLDS = 5
SEED = 42
y_binned = pd.qcut(y_train, q=FOLDS, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

lgb_model = lgb.LGBMRegressor(
    n_estimators=450,
    learning_rate=0.025,
    num_leaves=24,
    min_child_samples=12,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)

xgb_model = xgb.XGBRegressor(
    n_estimators=450,
    learning_rate=0.025,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0
)

cb_model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=5,
    l2_leaf_reg=3.0,
    random_seed=SEED,
    verbose=0
)

ridge_model = Ridge(alpha=30.0)
enet_model = ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=SEED)

models = {
    "LGB": (lgb_model, X_train, X_test),
    "XGB": (xgb_model, X_train, X_test),
    "CB": (cb_model, X_train, X_test),
    "Ridge": (ridge_model, X_train_s, X_test_s),
    "ElasticNet": (enet_model, X_train_s, X_test_s)
}

oofs = {}
preds = {}

for name, (mdl, tr_mat, te_mat) in models.items():
    oof_vec = np.zeros(len(y_train))
    pred_vec = np.zeros(len(X_test))
    for tr_idx, va_idx in skf.split(tr_mat, y_binned):
        mdl.fit(tr_mat[tr_idx], y_train[tr_idx])
        oof_vec[va_idx] = mdl.predict(tr_mat[va_idx])
        pred_vec += mdl.predict(te_mat) / FOLDS
    oofs[name] = oof_vec
    preds[name] = pred_vec
    rmse = np.sqrt(mean_squared_error(y_train, oof_vec))
    pr, _ = pearsonr(y_train, oof_vec)
    print(f"{name} -> CV RMSE: {round(float(rmse), 4)} | Pearson: {round(float(pr), 4)}")

def ensemble_loss(w):
    pred = sum(w[i] * oofs[name] for i, name in enumerate(models.keys()))
    return np.sqrt(mean_squared_error(y_train, pred))

init_w = [0.25, 0.25, 0.25, 0.15, 0.10]
opt = minimize(ensemble_loss, init_w, bounds=[(0, 1)] * len(models))
best_w = opt.x / np.sum(opt.x)

final_oof = sum(best_w[i] * oofs[name] for i, name in enumerate(models.keys()))
final_test = sum(best_w[i] * preds[name] for i, name in enumerate(models.keys()))

final_rmse = np.sqrt(mean_squared_error(y_train, final_oof))
final_pr, _ = pearsonr(y_train, final_oof)
print(f"Optimal Weights: {[round(float(x), 3) for x in best_w]}")
print(f"Ensemble OOF RMSE: {round(float(final_rmse), 4)} | Pearson: {round(float(final_pr), 4)}")

final_test_clipped = np.clip(final_test, 0.0, 5.0)

for idx, row in test_full.iterrows():
    if row["is_empty_or_punct"] == 1.0 or row["word_count"] <= 2:
        final_test_clipped[idx] = 0.0

sub_df = pd.DataFrame({
    "filename": test_full["filename"].values,
    "label": final_test_clipped
})

expected_order = test_df["filename"].tolist()
sub_df = sub_df.set_index("filename").reindex(expected_order).reset_index()

sub_path = os.path.join(OUTPUT_DIR, "submission_multimodal.csv")
sub_df.to_csv(sub_path, index=False)
sub_df.to_csv(os.path.join(BASE_DIR, "submission.csv"), index=False)

print(f"Submission saved to {sub_path} and {os.path.join(BASE_DIR, 'submission.csv')}")
print(f"Submission shape: {sub_df.shape}")
print(sub_df.describe())
print(sub_df.head(15))

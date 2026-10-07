# SHL Hiring Assessment 2026 - Spoken English Grammar & Fluency Scoring Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![CatBoost](https://img.shields.io/badge/CatBoost-GBDT-yellowgreen.svg)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Fast%20GBDT-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-orange.svg)](https://xgboost.readthedocs.io/)

A multi-modal machine learning architecture designed to evaluate spoken English candidate responses, predicting holistic grammar and fluency proficiency scores with high precision.

---

## 🚀 Key Architectural Highlights

1. **Phonetic Acoustic Representations (Wav2Vec2)**:
   - Extracted dense 768-dimensional phonetic embeddings using `facebook/wav2vec2-base-960h` across audio responses.
   - Reduced via TruncatedSVD to 32 orthogonal components capturing 93.3% of explained variance, giving direct acoustic signals of pronunciation quality and speaking rhythm.

2. **Linguistic, Syntactic & Disfluency Feature Engineering**:
   - **Lexical Diversity**: Root Type-Token Ratio ($V / \sqrt{N}$), Bilogarithmic TTR, Maas Index, and Honore statistic.
   - **Syntactic Complexity**: Frequency and ratio of subordinating conjunctions (`because`, `although`, `whereas`, `since`, `unless`), modal verbs, prepositions, and clausal depth.
   - **Hesitation & Disfluency Signals**: Dense tracking of filler words (`um`, `uh`, `like`, `you know`) and consecutive word repetitions (stutters).
   - **Lexical Sophistication**: Distribution of word lengths, proportion of $\ge 7$ and $\ge 9$ letter words.
   - **Speech Tempo & Prosody**: Syllable rates, articulation rate (syllables per non-silent second), pause density, and vocal dynamic variation.

3. **Multi-Scale Structural Text Representations**:
   - Word-level n-gram TF-IDF (1, 2) + SVD (24 components).
   - Character-level n-gram TF-IDF (3, 5) + SVD (16 components).
   - Dense Sentence Transformer embeddings (`all-mpnet-base-v2` + `all-MiniLM-L6-v2`) + SVD (48 components).

4. **Outlier Filtering & Dynamic Range Calibration**:
   - Filtered out 37 corrupted/silent `0.0` outlier recordings (`audio_50xx.wav`) from speech model training to prevent systematic downwards prediction bias.
   - Applied linear post-hoc regression calibration ($a \cdot \hat{y} + b$) to match the true standard deviation and dynamic range of candidate speech.

5. **6-Model Stacking Ensemble**:
   - Trained across 5-fold Stratified Cross-Validation:
     - **LightGBM Regressor**
     - **CatBoost Regressor**
     - **XGBoost Regressor**
     - **ElasticNet Regressor**
     - **Ridge Regressor**
     - **Bayesian Ridge Regressor**
   - Out-of-fold blending weights solved via constrained SLSQP optimization directly minimizing RMSE.

---

## 📊 Cross-Validation Performance

Evaluated on candidate speech responses across 5 stratified folds:

| Model | 5-Fold CV RMSE | Pearson Correlation ($r$) | Ensemble Weight |
| :--- | :---: | :---: | :---: |
| **LightGBM** | 0.6537 | 0.7708 | **50.8%** |
| **CatBoost** | 0.6560 | 0.7690 | 8.3% |
| **XGBoost** | 0.6539 | 0.7702 | 0.6% |
| **ElasticNet** | 0.6600 | 0.7593 | 13.3% |
| **Ridge** | 0.7127 | 0.7339 | 27.1% |
| **Bayesian Ridge**| 0.6662 | 0.7540 | 0.0% |
| **Ensemble (Raw)** | **0.6275** | **0.7882** | **100%** |
| **Ensemble (Calibrated)** | **0.6239** | **0.7882** | **100%** |

---

## 📁 Repository Structure

```
shl-hiring-assessment-2026/
├── Dataset_Final/
│   ├── extract_features.py            # Baseline acoustic & readability extractor
│   ├── extract_wav2vec2_fast.py       # Multi-threaded Wav2Vec2 phonetic extractor
│   ├── train_master_pipeline.py       # Master training, stacking & calibration pipeline
│   ├── train_features.csv             # Precomputed tabular acoustic & text features
│   ├── test_features.csv              # Precomputed test features
│   ├── train_transcripts.csv          # Whisper ASR candidate transcripts
│   ├── test_transcripts.csv           # Whisper ASR test transcripts
│   ├── train_text_features.npy        # 768-d MPNet sentence embeddings
│   ├── test_text_features.npy         # 768-d MPNet sentence embeddings
│   ├── train_wav2vec2_features.npy    # 768-d Wav2Vec2 phonetic embeddings
│   ├── test_wav2vec2_features.npy     # 768-d Wav2Vec2 phonetic embeddings
│   └── submission.csv                 # Final calibrated submission file
├── shl_grammar_scoring_engine.ipynb   # Interactive analysis and training notebook
├── train_multimodal_fast.py           # Fast acoustic & text modeling script
├── run_transcription.py               # Whisper transcription runner
├── submission.csv                     # Verified submission file ready for evaluation
└── README.md
```

---

## ⚙️ Quickstart & Reproduction

### 1. Requirements
```bash
pip install torch soundfile librosa transformers sentence-transformers textstat nltk catboost lightgbm xgboost scikit-learn pandas numpy
```

### 2. Extract Acoustic & Phonetic Representations
```bash
cd Dataset_Final
python extract_wav2vec2_fast.py
```

### 3. Train Master Ensemble & Generate Submission
```bash
python train_master_pipeline.py
```
This trains the 6-model cross-validated ensemble, applies optimal SLSQP blending, executes post-hoc linear calibration, and writes the output to `submission.csv`.

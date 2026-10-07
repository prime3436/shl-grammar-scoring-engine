import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Load original features
train_df = pd.read_csv('train_features.csv')
test_df = pd.read_csv('test_features.csv')

features = [c for c in train_df.columns if c not in ['filename', 'transcript', 'label']]
X_train_orig = train_df[features].fillna(0).values
y_train = train_df['label'].values
X_test_orig = test_df[features].fillna(0).values

# Load wav2vec2 features
train_w2v = np.load('train_wav2vec2_features.npy')
test_w2v = np.load('test_wav2vec2_features.npy')

X_train = np.hstack([X_train_orig, train_w2v])
X_test = np.hstack([X_test_orig, test_w2v])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def train_ridge(X, y, X_test):
    test_preds = np.zeros(len(X_test))
    val_preds = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_test)
        
        model = Ridge(alpha=10.0)
        model.fit(X_tr, y_tr)
        
        val_preds[va_idx] = model.predict(X_va)
        test_preds += model.predict(X_te) / kf.n_splits
    return val_preds, test_preds

def train_lgb(X, y, X_test):
    test_preds = np.zeros(len(X_test))
    val_preds = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        val_preds[va_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / kf.n_splits
    return val_preds, test_preds

val_ridge, test_ridge = train_ridge(X_train, y_train, X_test)
val_lgb, test_lgb = train_lgb(X_train, y_train, X_test)

print(f"Ridge RMSE: {root_mean_squared_error(y_train, val_ridge)}")
print(f"LGBM RMSE: {root_mean_squared_error(y_train, val_lgb)}")

# Combine
val_ens = val_ridge * 0.5 + val_lgb * 0.5
test_ens = test_ridge * 0.5 + test_lgb * 0.5
print(f"Ensemble RMSE: {root_mean_squared_error(y_train, val_ens)}")

sub = pd.DataFrame({'filename': test_df['filename'], 'label': test_ens})
sub.to_csv('submission_w2v_ens.csv', index=False)
print("Saved submission_w2v_ens.csv")

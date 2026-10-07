import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Load original features
train_df = pd.read_csv('train_features.csv')
test_df = pd.read_csv('test_features.csv')

features = [c for c in train_df.columns if c in test_df.columns and c not in ['filename', 'transcript', 'label']]
X_train_orig = train_df[features].fillna(0).values
y_train = train_df['label'].values
X_test_orig = test_df[features].fillna(0).values

# Load text features
train_txt = np.load('train_text_features.npy')
test_txt = np.load('test_text_features.npy')

# Load w2v features
train_w2v = np.load('train_w2v_features.npy')
test_w2v = np.load('test_w2v_features.npy')

X_train = np.hstack([X_train_orig, train_txt, train_w2v])
X_test = np.hstack([X_test_orig, test_txt, test_w2v])

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
        
        model = Ridge(alpha=100.0)
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
        
        model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=31, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        val_preds[va_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / kf.n_splits
    return val_preds, test_preds

def train_cb(X, y, X_test):
    test_preds = np.zeros(len(X_test))
    val_preds = np.zeros(len(X))
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        model = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6, random_state=42, verbose=0, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va))
        
        val_preds[va_idx] = model.predict(X_va)
        test_preds += model.predict(X_test) / kf.n_splits
    return val_preds, test_preds

val_ridge, test_ridge = train_ridge(X_train, y_train, X_test)
print(f"Ridge RMSE: {root_mean_squared_error(y_train, val_ridge)}")

val_lgb, test_lgb = train_lgb(X_train, y_train, X_test)
print(f"LGBM RMSE: {root_mean_squared_error(y_train, val_lgb)}")

val_cb, test_cb = train_cb(X_train, y_train, X_test)
print(f"CatBoost RMSE: {root_mean_squared_error(y_train, val_cb)}")

# Combine
val_ens = val_ridge * 0.2 + val_lgb * 0.4 + val_cb * 0.4
test_ens = test_ridge * 0.2 + test_lgb * 0.4 + test_cb * 0.4
print(f"Ensemble RMSE: {root_mean_squared_error(y_train, val_ens)}")

sub = pd.DataFrame({'filename': test_df['filename'], 'label': test_ens})
sub.to_csv('submission_all_ens.csv', index=False)
print("Saved submission_all_ens.csv")

val_df = pd.DataFrame({'filename': train_df['filename'], 'label': y_train, 'pred': val_ens})
val_df.to_csv('val_preds_all_ens.csv', index=False)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

base_dir = 'C:/Users/HP/Downloads/shl-hiring-assessment-2026/Dataset_Final'
train = pd.read_csv(os.path.join(base_dir, 'train_features.csv'))
test = pd.read_csv(os.path.join(base_dir, 'test_features.csv'))

target_col = 'label'

# Prepare data
drop_cols = ['filename', 'transcript', target_col]
# Check if there are any infinite or nan values
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Select only numeric columns
X = train.drop(columns=[c for c in drop_cols if c in train.columns]).select_dtypes(include=[np.number])
y = train[target_col]
X_test = test.drop(columns=[c for c in drop_cols if c in test.columns]).select_dtypes(include=[np.number])

X.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Ensure columns match
common_cols = list(set(X.columns) & set(X_test.columns))
X = X[common_cols]
X_test = X_test[common_cols]

import re
def make_unique(cols):
    new_cols = []
    for col in cols:
        new_col = re.sub(r'[^\w\s]', '_', str(col))
        original_new_col = new_col
        counter = 1
        while new_col in new_cols:
            new_col = f"{original_new_col}_{counter}"
            counter += 1
        new_cols.append(new_col)
    return new_cols

X.columns = make_unique(X.columns)
X_test.columns = make_unique(X_test.columns)

# We will use 5-fold cross validation and blending
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
oof_ridge = np.zeros(len(X))

preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))
preds_ridge = np.zeros(len(X_test))

# Normalize for Ridge
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # LightGBM
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    callbacks = [lgb.early_stopping(100, verbose=False)]
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    preds_lgb += model_lgb.predict(X_test) / n_splits
    
    # XGBoost
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=5,
        random_state=42,
        early_stopping_rounds=100
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    preds_xgb += model_xgb.predict(X_test) / n_splits
    
    # CatBoost
    model_cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        random_state=42,
        verbose=False
    )
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100
    )
    oof_cat[val_idx] = model_cat.predict(X_val)
    preds_cat += model_cat.predict(X_test) / n_splits
    
    # Ridge
    X_train_s, y_train_s = X_scaled[train_idx], y.iloc[train_idx]
    X_val_s = X_scaled[val_idx]
    
    model_ridge = Ridge(alpha=10.0, random_state=42)
    model_ridge.fit(X_train_s, y_train_s)
    oof_ridge[val_idx] = model_ridge.predict(X_val_s)
    preds_ridge += model_ridge.predict(X_test_scaled) / n_splits
    
print(f"LGBM RMSE: {root_mean_squared_error(y, oof_lgb):.4f}")
print(f"XGB RMSE: {root_mean_squared_error(y, oof_xgb):.4f}")
print(f"CatBoost RMSE: {root_mean_squared_error(y, oof_cat):.4f}")
print(f"Ridge RMSE: {root_mean_squared_error(y, oof_ridge):.4f}")

# Blend using a simple meta model (Linear Regression)
oof_df = pd.DataFrame({
    'lgb': oof_lgb,
    'xgb': oof_xgb,
    'cat': oof_cat,
    'ridge': oof_ridge
})
test_df = pd.DataFrame({
    'lgb': preds_lgb,
    'xgb': preds_xgb,
    'cat': preds_cat,
    'ridge': preds_ridge
})

from sklearn.linear_model import LinearRegression
meta_model = LinearRegression()
meta_model.fit(oof_df, y)
oof_blend = meta_model.predict(oof_df)
preds_blend = meta_model.predict(test_df)

print(f"Blend RMSE: {root_mean_squared_error(y, oof_blend):.4f}")

submission = test[['filename']].copy()
submission['label'] = preds_blend
submission.to_csv(os.path.join(base_dir, 'submission.csv'), index=False)
print("Saved submission.csv")

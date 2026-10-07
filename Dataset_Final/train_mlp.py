import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

df = pd.read_csv('train_features.csv')
features = [c for c in df.columns if c not in ['filename', 'transcript', 'label']]
X = df[features].fillna(0).values
y = df['label'].values

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmses = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    
    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    val_ds = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(y_va))
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model = MLP(X.shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    
    for epoch in range(150):
        model.train()
        for bx, by in train_dl:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, by in val_dl:
                val_preds.extend(model(bx).numpy())
        
        val_rmse = root_mean_squared_error(y_va, val_preds)
        scheduler.step(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            
    print(f"Fold {fold} Best RMSE: {best_val_rmse}")
    rmses.append(best_val_rmse)

print(f"Average CV RMSE: {np.mean(rmses)}")

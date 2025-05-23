import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ────────────────────────────────────────────────────────────────────────────
# 1. Model Components (two-hand input)
# ────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, seq_len, d_model]
        return x + self.pe[:, : x.size(1), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        return self.norm2(src + src2)

class SignEncoder(nn.Module):
    def __init__(
        self, feat_dim, d_model=64, n_layers=4, nhead=4, num_classes=10, dropout=0.1
    ):
        super().__init__()
        # We assume feat_dim = 21*3*2 = 126
        self.seq_len = feat_dim // 3      # 42 tokens (21 landmarks × 2 hands)
        self.input_dim = 3                # x,y,z per landmark
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, feat_dim]
        B = x.size(0)
        # reshape into tokens: [B, seq_len, input_dim]
        x = x.view(B, self.seq_len, self.input_dim)
        x = self.embedding(x)             # [B, seq_len, d_model]
        x = self.pos_encoding(x)          # + positional
        for layer in self.layers:
            x = layer(x)                  # [B, seq_len, d_model]
        x = x.transpose(1, 2)             # [B, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [B, d_model]
        return self.fc_out(x)             # [B, num_classes]

# ────────────────────────────────────────────────────────────────────────────
# 2. Training Script for Two-Hand Data
# ────────────────────────────────────────────────────────────────────────────

def train():
    csv_path = r"video_landmarks_dataset_filled.csv"
    df = pd.read_csv(csv_path)

    # Drop metadata, keep label + 126 features
    df = df.drop(columns=['video_id'])

    X = df.drop(columns=['label']).values.astype(np.float32)  # [N,126]
    y = df['label'].values

    # Preprocessing
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    joblib.dump(scaler, 'scaler2.pkl')
    joblib.dump(le, 'label_encoder2.pkl')
    num_classes = len(le.classes_)

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_scaled, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # DataLoaders
    class FrameDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y).long()
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            return self.X[i], self.y[i]

    train_loader = DataLoader(FrameDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(FrameDataset(X_val,   y_val),   batch_size=32)
    test_loader  = DataLoader(FrameDataset(X_test,  y_test),  batch_size=32)

    # Model + training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignEncoder(feat_dim=X.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0.0
    no_imp = 0
    for epoch in range(1, 70):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item()
                preds = logits.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch}  Train Loss: {train_loss:.4f}  Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict().copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 5:
                print("Early stopping.")
                break

    # Save best
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), 'best_sign_encoder2.pth')
    print(f"Training complete. Best Val Acc: {best_acc:.4f}")

    # Test
    model.eval()
    corr, tot = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            corr += (pred == yb).sum().item()
            tot += yb.size(0)
    print(f"Test Accuracy: {corr/tot:.4f}")

if __name__ == '__main__':
    train()

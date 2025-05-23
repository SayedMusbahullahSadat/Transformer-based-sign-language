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

#─────────────────────────────────────────────────────────────────────────────
# 1. Model Components
#─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, src):
        # Self-attention sublayer
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + src2)
        # Feed-forward sublayer
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = self.norm2(src + src2)
        return src

class SignEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=4, nhead=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.seq_len = 42       # 21 landmarks × 2 hands
        self.input_dim = 3      # (x, y, z)
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch_size, 126]
        B = x.size(0)
        x = x.view(B, self.seq_len, self.input_dim)     # [B, 42, 3]
        x = self.embedding(x)                           # [B, 42, d_model]
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)                           # [B, d_model, 42]
        x = self.global_pool(x).squeeze(-1)              # [B, d_model]
        logits = self.fc_out(x)                         # [B, num_classes]
        return logits

    def predict_video(self, frames: np.ndarray, scaler: StandardScaler, device: torch.device) -> int:
        # frames: [num_frames, 126]
        frames_scaled = scaler.transform(frames)         # [F, 126]
        inputs = torch.from_numpy(frames_scaled).float().to(device)
        self.eval()
        with torch.no_grad():
            logits = self(inputs)                       # [F, num_classes]
            probs = torch.softmax(logits, dim=1)        # [F, num_classes]
            avg_probs = probs.mean(dim=0)               # [num_classes]
            return torch.argmax(avg_probs).item()

#─────────────────────────────────────────────────────────────────────────────
# 2. Training Script
#─────────────────────────────────────────────────────────────────────────────
def train():
    # Update this path to your cleaned CSV
    csv_path = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\cleaned_vedio_dataset_version2.csv"
    df = pd.read_csv(csv_path)

    # Features & labels
    X = df.drop(columns=['label']).values            # [N, 126]
    y = df['label'].values                           # [N]

    # Preprocessing
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    num_classes = len(le.classes_)

    # Save preprocessing objects
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(le, "label_encoder.pkl")

    # Train/Val/Test split
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_enc, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Datasets & Loaders
    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).long()
        def __len__(self): return len(self.X)
        def __getitem__(self, i): return self.X[i], self.y[i]

    train_loader = DataLoader(DS(X_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(DS(X_val,   y_val),   batch_size=32)
    test_loader  = DataLoader(DS(X_test,  y_test),  batch_size=32)

    # Model init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignEncoder(d_model=64, n_layers=4, nhead=4, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Training loop with early stopping
    best_acc, no_imp = 0.0, 0
    EPOCHS, patience = 90, 5
    for epoch in range(1, EPOCHS+1):
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

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_loss += criterion(out, yb).item()
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(val_loader)
        val_acc = correct / total
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{EPOCHS}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict().copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break

    # Save best model
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), "best_sign_encoder.pth")
    print(f"Finished training. Best Val Acc: {best_acc:.4f}")

    # Optional test evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Test Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    train()

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# ────────────────────────────────────────────────────────────────────────────
# 1. Dataset: whole-video sequences for isolated gloss classification (two-hand)
# ────────────────────────────────────────────────────────────────────────────
class IsolatedSignVideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, scaler: StandardScaler, encoder: LabelEncoder):
        # df must contain video_id, label, and 126 two-hand features
        grouped = df.groupby('video_id')
        self.video_ids = []
        self.video_to_frames = {}
        self.video_to_label = {}
        for vid, sub in grouped:
            feats = sub.drop(columns=['video_id', 'label']).values.astype(np.float32)
            self.video_ids.append(vid)
            self.video_to_frames[vid] = feats
            self.video_to_label[vid] = sub['label'].iat[0]
        self.scaler = scaler
        self.encoder = encoder

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        seq = self.video_to_frames[vid]
        # scale using train-only scaler
        seq_scaled = self.scaler.transform(seq)
        frames = torch.from_numpy(seq_scaled).float()  # [T, feat_dim]
        lbl = self.encoder.transform([self.video_to_label[vid]])[0]
        return frames, lbl

# Collate to pad variable-length sequences in a batch
def sequence_collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)
    feat_dim = sequences[0].size(1)
    padded = []
    for seq in sequences:
        pad = torch.zeros(max_len - seq.size(0), feat_dim)
        padded.append(torch.cat([seq, pad], dim=0))
    stacked = torch.stack(padded, dim=0)  # [B, max_len, feat_dim]
    labels = torch.tensor(labels, dtype=torch.long)
    return stacked, labels, lengths

# ────────────────────────────────────────────────────────────────────────────
# 2. Model: sequence encoder with pooling (with higher dropout)
# ────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.3):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()
    def forward(self, src):
        attn, _ = self.self_attn(src, src, src)
        src = self.norm1(src + attn)
        ff = self.linear2(self.dropout(self.act(self.linear1(src))))
        return self.norm2(src + ff)

class SequenceSignEncoder(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_layers=4, nhead=4,
                 num_classes=10, dropout=0.3, max_len=500):
        super().__init__()
        self.embedding = nn.Linear(feat_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)
        self.max_len = max_len
    def forward(self, x):  # x: [B, T, feat_dim]
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.fc_out(x)
    def predict_sequence(self, frames, scaler, encoder, device):
        T, D = frames.shape
        if T > self.max_len:
            seq = frames[:self.max_len]
        else:
            pad = np.zeros((self.max_len - T, D), dtype=np.float32)
            seq = np.vstack([frames, pad])
        X = scaler.transform(seq)
        X_tensor = torch.from_numpy(X).unsqueeze(0).float().to(device)
        self.eval()
        with torch.no_grad():
            logits = self(X_tensor)
            idx = logits.argmax(dim=1).item()
        return encoder.inverse_transform([idx])[0]

# ────────────────────────────────────────────────────────────────────────────
# 3. Training Script for Two-Hand Sequence Model (with proper scaling on train)
# ────────────────────────────────────────────────────────────────────────────
def train_sequence_twohand(csv_path,
                            batch_size=4,
                            epochs=60,
                            lr=1e-4,
                            weight_decay=1e-5,
                            patience=5):
    # Load full DataFrame
    df = pd.read_csv(csv_path)
    # Split by video_id with stratification
    vids = df['video_id'].unique()
    labels_map = {vid: grp['label'].iat[0] for vid, grp in df.groupby('video_id')}
    y_vids = [labels_map[v] for v in vids]
    train_vids, val_vids = train_test_split(
        vids, test_size=0.2, random_state=42, stratify=y_vids)
    # Build train/val DataFrames
    train_df = df[df['video_id'].isin(train_vids)]
    val_df   = df[df['video_id'].isin(val_vids)]
    # Fit scaler/encoder on train only
    feat_cols = [c for c in df.columns if c not in ('video_id','label')]
    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df['label'].values
    scaler = StandardScaler().fit(X_train)
    encoder = LabelEncoder().fit(y_train)
    joblib.dump(scaler, 'scaler_sequence_twohand.pkl')
    joblib.dump(encoder, 'encoder_sequence_twohand.pkl')
    num_classes = len(encoder.classes_)
    # Create Dataset & Loader
    train_ds = IsolatedSignVideoDataset(train_df, scaler, encoder)
    val_ds   = IsolatedSignVideoDataset(val_df,   scaler, encoder)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=sequence_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              collate_fn=sequence_collate_fn)
    # Model, loss, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat_dim = len(feat_cols)
    model = SequenceSignEncoder(feat_dim=feat_dim,
                                 num_classes=num_classes,
                                 dropout=0.3,
                                 max_len=500).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=2)
    # Training loop
    best_acc, no_imp = 0.0, 0
    for epoch in range(1, epochs+1):
        model.train()
        t_loss, t_corr, t_tot = 0.0, 0, 0
        for xb, yb, lengths in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            preds = out.argmax(dim=1)
            t_corr += (preds == yb).sum().item()
            t_tot += yb.size(0)
        train_loss = t_loss / len(train_loader)
        train_acc  = t_corr / t_tot
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb, lengths in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                v_loss += criterion(out, yb).item()
                preds = out.argmax(dim=1)
                v_corr += (preds == yb).sum().item()
                v_tot += yb.size(0)
        val_loss = v_loss / len(val_loader)
        val_acc  = v_corr / v_tot
        scheduler.step(val_acc)
        print(f"Epoch {epoch}/{epochs}  Train Acc: {train_acc:.4f}  """
              f"Val Acc: {val_acc:.4f}  Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = model.state_dict().copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break
    # Save best
    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), 'sequence_twohand_encoder.pth')
    print(f"Finished. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    csv_path = r"video_landmarks_dataset_filled.csv"
    train_sequence_twohand(csv_path)

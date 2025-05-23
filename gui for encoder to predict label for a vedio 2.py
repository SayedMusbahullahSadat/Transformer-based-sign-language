import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import math

# ─── Configuration ───────────────────────────────────────────
SCALER_PATH        = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_CHECKPOINT   = "best_sign_encoder_from_filled.pth"

# ─── Model Components ────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1):
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
        self.self_attn = nn.MultiheadAttention(d_model, nhead,
                                               dropout=dropout, batch_first=True)
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
    def __init__(self, feat_dim, d_model=64,
                 n_layers=4, nhead=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.embedding   = nn.Linear(feat_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1)
        self.layers   = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward=2*d_model,
                                    dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out      = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, feat_dim]
        x = x.unsqueeze(1)              # [B, 1, feat_dim]
        x = self.embedding(x)           # [B, 1, d_model]
        x = self.pos_encoding(x)        # [B, 1, d_model]
        for layer in self.layers:
            x = layer(x)                # [B, 1, d_model]
        x = x.transpose(1, 2)           # [B, d_model, 1]
        x = self.global_pool(x).squeeze(-1)  # [B, d_model]
        return self.fc_out(x)           # [B, num_classes]

    def predict(self, frames, scaler, label_encoder, device):
        frames_scaled = scaler.transform(frames)
        inp = torch.from_numpy(frames_scaled).float().to(device)
        self.eval()
        with torch.no_grad():
            logits = self(inp)                    # [F, num_classes]
            probs  = torch.softmax(logits, dim=1)  # [F, num_classes]
            avg    = probs.mean(dim=0)             # [num_classes]
            idx    = torch.argmax(avg).item()
        return label_encoder.inverse_transform([idx])[0]

# ─── Load scaler, label‐encoder, and model ─────────────────────
scaler    = joblib.load(SCALER_PATH)
le        = joblib.load(LABEL_ENCODER_PATH)
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEAT_DIM  = 21 * 3 * 2  # 21 landmarks × (x,y,z) × 2 hands = 126
model     = SignEncoder(feat_dim=FEAT_DIM,
                        d_model=64,
                        n_layers=4,
                        nhead=4,
                        num_classes=len(le.classes_),
                        dropout=0.1).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))

# ─── MediaPipe Hands setup ───────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def extract_landmarks_from_video(path):
    cap = cv2.VideoCapture(path)
    all_rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        if res.multi_hand_landmarks:
            row = [0.0] * FEAT_DIM
            for hi, hand in enumerate(res.multi_hand_landmarks[:2]):
                base = hi * 21 * 3
                for i, lm in enumerate(hand.landmark):
                    row[base + 3*i    ] = lm.x
                    row[base + 3*i + 1] = lm.y
                    row[base + 3*i + 2] = lm.z
            all_rows.append(row)
    cap.release()
    return np.array(all_rows, dtype=np.float32)

def predict_sign(video_path):
    frames = extract_landmarks_from_video(video_path)
    if frames.size == 0:
        messagebox.showerror("Error","No hand landmarks detected.")
        return
    label = model.predict(frames, scaler, le, device)
    messagebox.showinfo("Result", f"Predicted Sign: {label}")

# ─── Build GUI ───────────────────────────────────────────────
root = tk.Tk()
root.title("Sign Language Predictor")
root.geometry("450x250")

tk.Label(root, text="Sign Language Predictor",
         font=(None, 18)).pack(pady=20)

tk.Button(root, text="Select Video File",
          font=(None, 14), width=20, height=2,
          command=lambda: predict_sign(
              filedialog.askopenfilename(
                filetypes=[("Videos","*.mp4;*.avi;*.mov")]
              )
          )).pack(pady=10)

root.mainloop()

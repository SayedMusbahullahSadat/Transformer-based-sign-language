import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import joblib
import math

# ─── Model Components ─────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):  # x: [F, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

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
        src2, _ = self.self_attn(src, src, src)
        src = self.norm1(src + src2)
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        return self.norm2(src + src2)

class SignEncoder(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_layers=4, nhead=4, num_classes=10, dropout=0.1):
        super().__init__()
        self.seq_len = feat_dim // 3  # tokens = (#landmarks*hands)
        self.input_dim = 3
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [F, feat_dim]
        B = x.size(0)
        x = x.view(B, self.seq_len, self.input_dim)  # [F, tokens, 3]
        x = self.embedding(x)                        # [F, tokens, d_model]
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)                        # [F, d_model, tokens]
        x = self.global_pool(x).squeeze(-1)           # [F, d_model]
        return self.fc_out(x)                        # [F, num_classes]

# ─── Load artifacts ──────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load('scaler2.pkl')
label_encoder = joblib.load('label_encoder2.pkl')
feat_dim = scaler.mean_.shape[0]  # should be 21*3*2 = 126
num_classes = len(label_encoder.classes_)
model = SignEncoder(feat_dim=feat_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_sign_encoder2.pth', map_location=device))
model.eval()

# ─── MediaPipe Hands ─────────────────────────────────────────────────────────
mp_processor = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5
)

# ─── Video landmark extraction ─────────────────────────────────────────────────
def extract_two_hand_landmarks(video_path):
    cap = cv2.VideoCapture(video_path)
    feats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_processor.process(rgb)
        coords = [0.0] * feat_dim
        if res.multi_hand_landmarks:
            for hi, hand in enumerate(res.multi_hand_landmarks[:2]):
                base = hi * 21 * 3
                for i, lm in enumerate(hand.landmark):
                    coords[base + 3*i    ] = lm.x
                    coords[base + 3*i + 1] = lm.y
                    coords[base + 3*i + 2] = lm.z
        feats.append(coords)
    cap.release()
    return np.array(feats, dtype=np.float32)

# ─── Prediction ──────────────────────────────────────────────────────────────
def predict_from_file(path):
    frames = extract_two_hand_landmarks(path)
    if frames.size == 0:
        messagebox.showerror('Error', 'No hand landmarks detected.')
        return
    X = scaler.transform(frames)
    with torch.no_grad():
        inp = torch.from_numpy(X).float().to(device)
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        avg = probs.mean(dim=0)
        pred = torch.argmax(avg).item()
        label = label_encoder.inverse_transform([pred])[0]
    result_var.set(f'Predicted Sign: {label}')
    messagebox.showinfo('Result', f'Predicted Sign: {label}')

# ─── Build GUI ────────────────────────────────────────────────────────────────
root = tk.Tk()
root.title('Video Sign Recognition')
root.geometry('400x200')

tk.Label(root, text='Video Sign Recognition', font=('Arial', 16)).pack(pady=10)

button = tk.Button(
    root,
    text='Select Video File',
    font=('Arial', 12),
    command=lambda: predict_from_file(
        filedialog.askopenfilename(filetypes=[('Video Files','*.mp4;*.avi;*.mov')])
    )
)
button.pack(pady=10)

result_var = tk.StringVar(value='')
result_label = tk.Label(root, textvariable=result_var, font=('Arial', 14))
result_label.pack(pady=10)

root.mainloop()

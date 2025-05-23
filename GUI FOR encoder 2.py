import os
import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import torch
import joblib
import mediapipe as mp
from PIL import Image, ImageTk
import math
import torch.nn as nn

# Drawing utilities
mp_hands = mp.solutions.hands
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

# ------- Model definitions (two-hand) -------
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
        return x + self.pe[:, :x.size(1), :]

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
    def __init__(self, feat_dim, d_model=64, n_layers=4, nhead=4,
                 num_classes=10, dropout=0.1):
        super().__init__()
        # feat_dim should be 21*3*2 = 126
        self.seq_len = feat_dim // 3      # 42 tokens
        self.input_dim = 3
        self.embedding = nn.Linear(self.input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward=2*d_model,
                                    dropout=dropout)
            for _ in range(n_layers)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [F, feat_dim]
        # reshape to tokens
        B = x.size(0)
        x = x.view(B, self.seq_len, self.input_dim)  # [F,42,3]
        x = self.embedding(x)                        # [F,42,d]
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2)                        # [F,d,42]
        x = self.global_pool(x).squeeze(-1)           # [F,d]
        return self.fc_out(x)                        # [F,num_classes]

# ------- Load artifacts -------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = joblib.load('scaler2.pkl')
label_encoder = joblib.load('label_encoder2.pkl')
feat_dim = scaler.mean_.shape[0]  # should be 126
num_classes = len(label_encoder.classes_)
model = SignEncoder(feat_dim=feat_dim, num_classes=num_classes).to(device)
model.load_state_dict(torch.load('best_sign_encoder2.pth', map_location=device))
model.eval()

# ------- MediaPipe hands -------
mp_processor = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=2,
                               min_detection_confidence=0.5)

# ------- GUI state -------
recording = False
frames = []

# ------- GUI functions -------
def update_frame():
    ret, frame = cap.read()
    if ret:
        disp = frame.copy()
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        res = mp_processor.process(rgb)
        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                drawing_utils.draw_landmarks(
                    disp, hl, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmarks_style(),
                    drawing_styles.get_default_hand_connections_style())
        else:
            cv2.putText(disp, 'No hand detected',
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)
        img = ImageTk.PhotoImage(Image.fromarray(
            cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        video_label.imgtk = img
        video_label.config(image=img)
        if recording:
            frames.append(frame.copy())
    root.after(10, update_frame)


def start_recording():
    global recording, frames
    recording = True
    frames = []
    start_btn.config(state=tk.DISABLED)
    stop_btn.config(state=tk.NORMAL)
    result_var.set('Recording...')


def stop_recording():
    global recording
    recording = False
    start_btn.config(state=tk.NORMAL)
    stop_btn.config(state=tk.DISABLED)
    result_var.set('Processing...')
    root.update_idletasks()
    predict_recording(frames)


def predict_recording(frames_list):
    feats = []
    for frm in frames_list:
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = mp_processor.process(rgb)
        # prepare 2-hand coords
        coords = [0.0] * feat_dim
        if res.multi_hand_landmarks:
            for hi, hand in enumerate(res.multi_hand_landmarks[:2]):
                base = hi * 21 * 3
                for i, lm in enumerate(hand.landmark):
                    coords[base + 3*i    ] = lm.x
                    coords[base + 3*i + 1] = lm.y
                    coords[base + 3*i + 2] = lm.z
        feats.append(np.array(coords, dtype=np.float32))
    if not feats:
        result_var.set('No frames captured')
        return
    X = scaler.transform(np.stack(feats, axis=0))
    with torch.no_grad():
        inp = torch.from_numpy(X).float().to(device)
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        avg = probs.mean(dim=0)
        pred = torch.argmax(avg).item()
        label = label_encoder.inverse_transform([pred])[0]
    result = f'Predicted Sign: {label}'
    result_var.set(result)
    messagebox.showinfo('Prediction', result)

# ------- GUI setup -------
root = tk.Tk()
root.title('Sign Recognition')
root.resizable(False, False)

video_label = tk.Label(root)
video_label.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)
start_btn = tk.Button(btn_frame, text='Start Recording', width=15, command=start_recording)
stop_btn = tk.Button(btn_frame, text='Stop Recording', width=15, state=tk.DISABLED, command=stop_recording)
start_btn.pack(side=tk.LEFT, padx=5)
stop_btn.pack(side=tk.LEFT, padx=5)

result_var = tk.StringVar(value='')
result_label = tk.Label(root, textvariable=result_var, font=('Arial',14))
result_label.pack(pady=10)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    messagebox.showerror('Error','Cannot open webcam')
    root.destroy()
else:
    update_frame()

root.protocol('WM_DELETE_WINDOW', lambda: (cap.release(), root.destroy()))
root.mainloop()

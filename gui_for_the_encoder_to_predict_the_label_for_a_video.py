import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import torch
import joblib

# ─── Configuration (adjust paths if needed) ─────────────────────────────
SCALER_PATH        = "scaler.pkl"               # path to scaler.pkl
LABEL_ENCODER_PATH = "label_encoder.pkl"        # path to label_encoder.pkl
MODEL_CHECKPOINT   = "best_sign_encoder.pth"    # path to your trained model checkpoint

# ─── Import the encoder class from the training script ─────────────────
from encoder_with_prediction_of_a_label import SignEncoder

# ─── Load preprocessing objects & model ─────────────────────────────────
scaler = joblib.load(SCALER_PATH)
le     = joblib.load(LABEL_ENCODER_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SignEncoder(num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device))
model.eval()

# ─── MediaPipe Hands setup ─────────────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ─── Helper: extract landmarks from video ────────────────────────────────
def extract_landmarks_from_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(rgb)
        if results.multi_hand_landmarks:
            row = [0] * (21 * 3 * 2)
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                base = hand_idx * 21 * 3
                for i, lm in enumerate(hand_landmarks.landmark):
                    row[base + 3*i]     = lm.x
                    row[base + 3*i + 1] = lm.y
                    row[base + 3*i + 2] = lm.z
            frames.append(row)
    cap.release()
    return np.array(frames)

# ─── Prediction logic ───────────────────────────────────────────────────
def predict_sign(video_path):
    frames = extract_landmarks_from_video(video_path)
    if frames.size == 0:
        messagebox.showerror("Error", "No hand landmarks detected.")
        return
    idx = model.predict_video(frames, scaler, device)
    sign = le.inverse_transform([idx])[0]
    messagebox.showinfo("Prediction", f"Predicted Sign: {sign}")

# ─── Build and run Tkinter GUI ──────────────────────────────────────────
root = tk.Tk()
root.title("Sign Language Predictor")
root.geometry("400x200")

btn = tk.Button(
    root,
    text="Select Video...",
    command=lambda: predict_sign(
        filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
    ),
    font=(None, 14),
    width=20,
    height=2
)
btn.pack(expand=True)

root.mainloop()

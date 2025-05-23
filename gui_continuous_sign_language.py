import tkinter as tk
from tkinter import filedialog, messagebox, font
from tkinter.scrolledtext import ScrolledText
import numpy as np
import torch
import cv2
import mediapipe as mp

from continuous_sign_dataset import ContinuousSignDataset
from continuous_sign_model_and_train import ContinuousSignEncoder
from encoder_with_prediction_of_a_label import PositionalEncoding, TransformerEncoderLayer

# ─── Configuration ────────────────────────────────────────────────────────
csv_path   = r"video_landmarks_dataset_filled.csv"  # adjust if needed
model_path = r"best_continuous_sign_encoder_ctc.pth"  # adjust if needed

# ─── Load dataset for encoder and blank index ──────────────────────────────
try:
    ds = ContinuousSignDataset(csv_path, N=1, blank_token='<blank>')
except Exception as e:
    messagebox.showerror("Initialization Error", f"Failed to load dataset:\n{e}")
    raise
blank_idx = ds.blank_idx
encoder   = ds.encoder
feat_dim  = next(iter(ds.video_to_frames.values())).shape[1]

# ─── Load CTC model ────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContinuousSignEncoder(
    feat_dim=feat_dim,
    d_model=64,
    n_layers=4,
    nhead=4,
    num_classes=len(encoder.classes_),
    dropout=0.1,
    max_len=1000
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ─── MediaPipe setup ──────────────────────────────────────────────────────
mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

# ─── Helper: CTC greedy decode ──────────────────────────────────────────────
def ctc_greedy_decode(log_probs: torch.Tensor):
    """
    Greedy CTC decode collapsing repeats and blanks into gloss sequence.
    """
    # preds as Python ints
    preds = log_probs.argmax(dim=1).cpu().numpy().tolist()  # list of ints
    decoded = []
    prev = None
    for p in preds:
        if p != blank_idx and p != prev:
            decoded.append(p)
        prev = p
    return encoder.inverse_transform(decoded)

# ─── Helper: extract landmarks from video ──────────────────────────────────
def extract_landmarks(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = mp_hands.process(rgb)
        if res.multi_hand_landmarks:
            row = [0.0] * (21*3*2)
            for hi, hand_landmarks in enumerate(res.multi_hand_landmarks[:2]):
                base = hi * 21 * 3
                for i, lm in enumerate(hand_landmarks.landmark):
                    row[base + 3*i]     = lm.x
                    row[base + 3*i + 1] = lm.y
                    row[base + 3*i + 2] = lm.z
            frames.append(row)
    cap.release()
    return np.array(frames, dtype=np.float32)

# ─── GUI Setup ────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Continuous Sign Language Translator")
root.geometry("640x480")
root.configure(padx=20, pady=20)

title_font = font.Font(family="Helvetica", size=18, weight="bold")
btn_font   = font.Font(family="Helvetica", size=14)

lbl_title = tk.Label(root, text="Continuous Sign Language Translator", font=title_font)
lbl_title.pack(pady=(0,10))

btn_select = tk.Button(
    root,
    text="Select Video",
    font=btn_font,
    width=20,
    command=lambda: on_select_video()
)
btn_select.pack(pady=(0,15))

output_box = ScrolledText(root, wrap=tk.WORD, height=10, font=("Helvetica", 12))
output_box.pack(fill=tk.BOTH, expand=True)
output_box.insert(tk.END, "Please select a video to begin.\n")
output_box.configure(state=tk.DISABLED)

# ─── Callback ─────────────────────────────────────────────────────────────
def on_select_video():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
    )
    if not file_path:
        return
    output_box.configure(state=tk.NORMAL)
    output_box.delete(1.0, tk.END)
    output_box.insert(tk.END, "Processing video...\n")
    output_box.update()

    # 1) Extract
    frames = extract_landmarks(file_path)
    if frames.size == 0:
        output_box.insert(tk.END, "No hand landmarks detected!\n")
        output_box.configure(state=tk.DISABLED)
        return

    # 2) Model inference
    inputs = torch.from_numpy(frames).unsqueeze(0)  # [1, T, feat].to(device)  # [T,1,feat]
    with torch.no_grad():
        logits   = model(inputs)                                # [T,1,C]
        log_probs = torch.log_softmax(logits, dim=2).squeeze(1) # [T,C]

    # 3) Decode
    gloss_seq = ctc_greedy_decode(log_probs)

    # 4) Display
    output_box.insert(tk.END, "Predicted sequence:\n")
    for g in gloss_seq:
        output_box.insert(tk.END, f"• {g}\n")
    output_box.configure(state=tk.DISABLED)

# ─── Run GUI ──────────────────────────────────────────────────────────────
root.mainloop()

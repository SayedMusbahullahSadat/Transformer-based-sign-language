import os
import sys
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────
csv_input = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\video_landmarks_dataset_with_ids.csv"
csv_output = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\video_landmarks_dataset_filled.csv"

# ─── Check input file exists ───────────────────────────────────────────
if not os.path.isfile(csv_input):
    print(f"Error: CSV file not found at {csv_input}")
    sys.exit(1)

# ─── Load CSV with error handling ───────────────────────────────────────
try:
    df = pd.read_csv(csv_input)
except pd.errors.EmptyDataError:
    print(f"Error: CSV at {csv_input} is empty or has no columns.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# ─── Check DataFrame content ────────────────────────────────────────────
if df.empty:
    print(f"Error: Loaded DataFrame from {csv_input} is empty.")
    sys.exit(1)

# ─── Identify feature columns (exclude video_id and label) ─────────────
reserved = ("video_id", "label")
feature_cols = [c for c in df.columns if c not in reserved]
if not feature_cols:
    print("Error: No feature columns found in CSV.")
    sys.exit(1)

# ─── Fill missing values in features with 0 ────────────────────────────
df[feature_cols] = df[feature_cols].fillna(0)

# ─── Save filled DataFrame ─────────────────────────────────────────────
try:
    df.to_csv(csv_output, index=False)
    print(f"Successfully filled missing landmarks and saved to {csv_output}")
except Exception as e:
    print(f"Error saving filled CSV: {e}")
    sys.exit(1)

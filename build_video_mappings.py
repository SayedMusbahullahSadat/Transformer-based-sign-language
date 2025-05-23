import pandas as pd

# ─── Configuration ────────────────────────────────────────────────────
csv_path = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\video_landmarks_dataset_filled.csv"

# ─── Load the filled CSV ───────────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"Total frames loaded: {len(df)}")

# ─── Group by video_id ─────────────────────────────────────────────────
video_to_frames = {}
video_to_label  = {}
for vid, sub in df.groupby("video_id"):
    # Extract feature matrix [num_frames, 126]
    feats = sub.drop(columns=["video_id","label"]).values
    video_to_frames[vid] = feats
    # All rows share the same label—take the first
    video_to_label[vid] = sub["label"].iat[0]

# ─── Sanity checks ─────────────────────────────────────────────────────
print(f"Total distinct videos: {len(video_to_frames)}")
# Print a few examples
for i, vid in enumerate(list(video_to_frames)[:5]):
    print(f"  Video {vid}: {video_to_frames[vid].shape[0]} frames → label = {video_to_label[vid]}")

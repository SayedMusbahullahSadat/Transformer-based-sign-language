import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Paths to video folder and output CSV
video_folder = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\Datasets\WLASL2\videos"
mapping_csv  = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\PRACTICE DATA\video_label_mapping.csv"
output_csv   = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\PRACTICE DATA\video_landmarks_dataset_with_ids.csv"

# Load the video-to-label mapping and pad video_id to 5 digits
try:
    mapping_df = pd.read_csv(mapping_csv)
    # Zero-pad IDs to length 5 (e.g., '23' -> '00023')
    mapping_df["video_id"] = mapping_df["video_id"].astype(str).str.zfill(5)
    video_to_label = dict(
        zip(mapping_df["video_id"], mapping_df["label"])
    )
except Exception as e:
    print(f"Error loading mapping CSV: {e}")
    video_to_label = {}

# Define columns for the output CSV, now including video_id
feature_cols = (
    [f"hand_1_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]] +
    [f"hand_2_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
)
columns = ["video_id", "label"] + feature_cols

# Create the CSV file with headers
try:
    pd.DataFrame([], columns=columns).to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")
except Exception as e:
    print(f"Error creating output CSV: {e}")

# Function to process each video and include video_id on each row
def process_video(video_path, video_id, label):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            video_data = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    # collect landmarks for up to two hands
                    lm_row = [None] * len(feature_cols)
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                        base = hand_idx * 21 * 3
                        for i, lm in enumerate(hand_landmarks.landmark):
                            lm_row[base + 3*i]     = lm.x
                            lm_row[base + 3*i + 1] = lm.y
                            lm_row[base + 3*i + 2] = lm.z
                    # prepend video_id and label
                    row = [video_id, label] + lm_row
                    video_data.append(row)
        cap.release()

        if video_data:
            try:
                video_df = pd.DataFrame(video_data, columns=columns)
                video_df.to_csv(
                    output_csv,
                    mode="a",
                    index=False,
                    header=False
                )
                print(f"Added {len(video_data)} frames for video: {video_id}")
            except Exception as e:
                print(f"Error writing data for {video_id}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {video_id}: {e}")

# Process all videos in folder, passing zero-padded video_id and label
for fname in tqdm(os.listdir(video_folder)):
    if not fname.lower().endswith(".mp4"):
        continue
    raw_id = os.path.splitext(fname)[0]
    video_id = raw_id.zfill(5)  # ensure 5-digit ID
    label = video_to_label.get(video_id, "unknown")
    video_path = os.path.join(video_folder, fname)
    process_video(video_path, video_id, label)

print(f"Finished saving landmarks (with IDs) to {output_csv}")

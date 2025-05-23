import os
import cv2
import mediapipe as mp
import pandas as pd
from tqdm import tqdm

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Paths to video folder and output CSV
video_folder = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\Datasets\WLASL\videos"
mapping_csv = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\video_label_mapping.csv"
output_csv = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\video_landmarks_dataset.csv"

# Load the video-to-label mapping
try:
    mapping_df = pd.read_csv(mapping_csv)
    video_to_label = dict(zip(mapping_df["video_id"].astype(str), mapping_df["label"]))
except Exception as e:
    print(f"Error loading mapping CSV: {e}")
    video_to_label = {}

# Define columns for the output CSV
columns = (
    [f"hand_1_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]] +
    [f"hand_2_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
)
columns.insert(0, "label")  # Add label column at the start

# Create the CSV file with headers
try:
    pd.DataFrame([], columns=columns).to_csv(output_csv, index=False)
    print(f"CSV file created: {output_csv}")
except Exception as e:
    print(f"Error creating output CSV: {e}")

# Function to process each video
def process_video(video_path, label):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        # Initialize MediaPipe Hands
        with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
            video_data = []  # Store landmarks for this video
            while cap.isOpened():
                try:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Convert frame to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)

                    if results.multi_hand_landmarks:
                        row = [None] * (21 * 3 * 2)  # Placeholder for two hands
                        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                            offset = hand_idx * 21 * 3  # Offset for hand_2
                            for i, lm in enumerate(hand_landmarks.landmark):
                                row[offset + i * 3] = lm.x
                                row[offset + i * 3 + 1] = lm.y
                                row[offset + i * 3 + 2] = lm.z
                        row.insert(0, label)  # Insert label at the start
                        video_data.append(row)

                except Exception as e:
                    print(f"Error processing frame in video {video_path}: {e}")
                    continue  # Skip the current frame and continue with the next one

        cap.release()

        # Append data for this video to the CSV
        if video_data:
            try:
                video_df = pd.DataFrame(video_data, columns=columns)
                video_df.to_csv(output_csv, mode="a", index=False, header=False)
                print(f"Added landmarks for video: {os.path.basename(video_path)}")
            except Exception as e:
                print(f"Error writing data for video {video_path}: {e}")

    except Exception as e:
        print(f"Unexpected error processing video {video_path}: {e}")

# Process all videos in the folder
for video_name in tqdm(os.listdir(video_folder)):
    try:
        if video_name.endswith(".mp4"):
            video_id = os.path.splitext(video_name)[0]  # Extract video ID
            label = video_to_label.get(video_id, "unknown")  # Get label from mapping
            video_path = os.path.join(video_folder, video_name)
            process_video(video_path, label)
    except Exception as e:
        print(f"Error processing file {video_name}: {e}")
        continue  # Skip the current file and continue with the next one

print(f"Landmarks for all videos have been saved to {output_csv}")

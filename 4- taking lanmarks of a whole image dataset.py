import mediapipe as mp
import cv2
import os
import csv

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Import FACEMESH connections

# File to save extracted landmarks
landmarks_file_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\landmarks_dataset.csv"

# Directory containing subdirectories with images
dataset_dir = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\Datasets\American Sign Language Dataset for Image Classifcation\asl_dataset"

# Check if the CSV file exists, and create it if not
if not os.path.exists(landmarks_file_path):
    with open(landmarks_file_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header row (e.g., features and label)
        header = [f"feature_{i}" for i in range(42 * 3)]  # normally 543 landmarks (x, y, z) but I just want to take hands so 42
        header.append("label")  # Add label column
        writer.writerow(header)

# Initiate holistic model
with mp_holistic.Holistic(
        static_image_mode=True,              # <--- we Add this line because of static images
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    # Iterate through each subdirectory (each representing a different sign)
    for sign_folder in os.listdir(dataset_dir):
        sign_folder_path = os.path.join(dataset_dir, sign_folder)

        # Check if it's a directory
        if os.path.isdir(sign_folder_path):
            print(f"Processing images in folder: {sign_folder}")

            # Iterate through each image in the subdirectory
            for filename in os.listdir(sign_folder_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    # Construct full path to the image
                    image_path = os.path.join(sign_folder_path, filename)

                    # Load an image from file
                    frame = cv2.imread(image_path)
                    if frame is None:
                        print(f"Failed to read image: {image_path}")
                        continue

                    # Recolor Feed for MediaPipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process the image and make detections
                    results = holistic.process(image)

                    # Recolor image back to BGR for rendering
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Extract landmarks and flatten into a single list
                    landmarks = []

                    """if results.face_landmarks:
                        for landmark in results.face_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])"""

                    if results.right_hand_landmarks:
                        for landmark in results.right_hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])

                    if results.left_hand_landmarks:
                        for landmark in results.left_hand_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])

                    """if results.pose_landmarks:
                        for landmark in results.pose_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z])"""

                    # Pad landmarks to ensure fixed length (543 * 3 = 1629) but because I want just hands it is gonna be 42 * 3
                    while len(landmarks) < 42 * 3:
                        landmarks.append(0.0)

                    # Add label for the sign (e.g., sign_folder)
                    label = sign_folder  # Use folder name as label
                    landmarks.append(label)

                    # Save landmarks to CSV
                    with open(landmarks_file_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(landmarks)

                    print(f"Landmarks saved for: {filename}")

print(f"All images processed and landmarks saved to CSV: {landmarks_file_path}")

import os
import csv
import cv2
import mediapipe as mp

# Number of landmarks for ONE hand: 21
# Each landmark has 3 coordinates (x, y, z).
# 21 * 3 = 63
TOTAL_HAND_FEATURES = 63

def extract_single_hand_features(image_path, hands_model):
    """
    Reads an image and uses MediaPipe Hands to detect a single hand.
    Returns a list of 63 floats for the first hand found.
    If no hand is detected, returns a zero-filled list of length 63.
    """
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Could not read image: {image_path}")
        # Return zeros for consistency
        return [0.0] * TOTAL_HAND_FEATURES

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Detect hands
    results = hands_model.process(image_rgb)

    # If at least one hand is detected, extract the FIRST hand
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        first_hand = results.multi_hand_landmarks[0]  # only the first hand
        features = []
        for lm in first_hand.landmark:
            features.extend([lm.x, lm.y, lm.z])  # (x, y, z)
        return features

    # If no hand detected, return zeros
    return [0.0] * TOTAL_HAND_FEATURES


def main():
    # 1) Path to your top-level dataset directory (each subfolder is a label)
    dataset_dir = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\Datasets\American Sign Language Dataset for Image Classifcation\asl_dataset"

    # 2) Output CSV file path
    output_csv = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\landmarks_dataset.csv"

    # 3) Create the CSV header if the file does not exist
    if not os.path.exists(output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # feature_0..feature_62 + label
            header = [f"feature_{i}" for i in range(TOTAL_HAND_FEATURES)]
            header.append("label")
            writer.writerow(header)

    # 4) Initialize MediaPipe Hands for static images
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,            # We only expect 1 hand
        min_detection_confidence=0.1
    ) as hands_model:

        # 5) Iterate over each subfolder in dataset_dir
        for subfolder_name in os.listdir(dataset_dir):
            subfolder_path = os.path.join(dataset_dir, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue  # Skip non-folder items

            print(f"Processing label (subfolder): {subfolder_name}")

            # Collect image files
            valid_extensions = ('.jpeg', '.jpg', '.png')
            image_files = [
                os.path.join(subfolder_path, fname)
                for fname in os.listdir(subfolder_path)
                if fname.lower().endswith(valid_extensions)
            ]

            # 6) Process each image
            for image_file in image_files:
                # Get 63 features (21 landmarks × 3 coords) for the first hand
                features_63 = extract_single_hand_features(image_file, hands_model)

                # Add label at the end
                row = features_63 + [subfolder_name]

                # Append row to CSV
                with open(output_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"Saved row for: {image_file}")

    print(f"\nAll done! Landmark features saved in CSV:\n{output_csv}")


if __name__ == "__main__":
    main()

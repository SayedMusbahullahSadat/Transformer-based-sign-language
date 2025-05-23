import mediapipe as mp
import cv2
import os

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Import FACEMESH connections

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    # Load an image from file
    image_path = r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\depositphotos_194974120-stock-photo-casual-man-full-body-in.jpg"
    frame = cv2.imread(image_path)  # Replace 'path_to_image.jpg' with the actual path

    # Recolor Feed for MediaPipe
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Process the image and make detections
    results = holistic.process(image)

    # Recolor image back to BGR for rendering
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Save detected results to a file
    results_file_path = os.path.join(os.path.dirname(image_path), "detection_results.txt")
    with open(results_file_path, "w") as f:
        if results.face_landmarks:
            f.write("Face Landmarks:\n")
            for idx, landmark in enumerate(results.face_landmarks.landmark):
                f.write(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}\n")

        if results.right_hand_landmarks:
            f.write("Right Hand Landmarks:\n")
            for idx, landmark in enumerate(results.right_hand_landmarks.landmark):
                f.write(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}\n")

        if results.left_hand_landmarks:
            f.write("Left Hand Landmarks:\n")
            for idx, landmark in enumerate(results.left_hand_landmarks.landmark):
                f.write(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}\n")

        if results.pose_landmarks:
            f.write("Pose Landmarks:\n")
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                f.write(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}\n")

    print(f"Detection results saved at: {results_file_path}")

    # 1. Draw face landmarks (using FACEMESH_TESSELATION)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,  # Face mesh tesselation
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), # Dots or circles
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1) # Connection lines
        )

    # 2. Draw right hand landmarks
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(225, 222, 33), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=1)
        )

    # 3. Draw left hand landmarks
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=1)
        )

    # 4. Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
        )

    # Save the processed image in the same directory as the original image
    save_path = os.path.join(os.path.dirname(image_path), "processed_image.jpg")
    cv2.imwrite(save_path, image)
    print(f"Processed image saved at: {save_path}")

    # Display the image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

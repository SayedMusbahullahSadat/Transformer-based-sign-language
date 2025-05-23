import mediapipe as mp
import cv2

# Initialize MediaPipe utilities
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh  # Import FACEMESH connections

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to retrieve frame from the webcam.")
            break

        # Recolor Feed for MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and make detections
        results = holistic.process(image)

        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks (using FACEMESH_TESSELATION)
        if results.face_landmarks: # you can remove the if statment and start from [mp_drawing.draw_landmarks(... ] and put it in the same indentation instead of if statement
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # Face mesh tesselation
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1), # first one of this [mp_drawing.DrawingSpec...] is for dots or circles
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1) # second one of this [mp_drawing.DrawingSpec...] is for connection lines between dots or joints
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

        # Display the image
        cv2.imshow('Raw Webcam Feed', image)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

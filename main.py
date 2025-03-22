import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# EAR threshold & frame counter
EAR_THRESHOLD = 0.25
CLOSED_EYE_FRAMES = 20  # Number of frames before triggering an alert
frame_counter = 0

# Start webcam
cam = cv2.VideoCapture(0)

def calculate_EAR(landmarks, eye_points, iw, ih):
    """Compute the Eye Aspect Ratio (EAR) for drowsiness detection."""
    p1 = np.array([landmarks[eye_points[0]].x * iw, landmarks[eye_points[0]].y * ih])  # Outer eye corner
    p4 = np.array([landmarks[eye_points[3]].x * iw, landmarks[eye_points[3]].y * ih])  # Inner eye corner

    p2 = np.array([landmarks[eye_points[1]].x * iw, landmarks[eye_points[1]].y * ih])  # Top inner
    p3 = np.array([landmarks[eye_points[2]].x * iw, landmarks[eye_points[2]].y * ih])  # Top outer
    p5 = np.array([landmarks[eye_points[4]].x * iw, landmarks[eye_points[4]].y * ih])  # Bottom outer
    p6 = np.array([landmarks[eye_points[5]].x * iw, landmarks[eye_points[5]].y * ih])  # Bottom inner

    # Calculate EAR
    vertical_dist = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal_dist = np.linalg.norm(p1 - p4)
    EAR = vertical_dist / (2.0 * horizontal_dist)
    
    return EAR

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face mesh detection
    results = face_mesh.process(rgb_frame)

    ih, iw, _ = frame.shape  # Get frame dimensions

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define eye landmarks
            right_eye = [33, 160, 158, 133, 153, 144]  
            left_eye = [362, 385, 387, 263, 373, 380]

            # Calculate EAR for both eyes
            right_EAR = calculate_EAR(face_landmarks.landmark, right_eye, iw, ih)
            left_EAR = calculate_EAR(face_landmarks.landmark, left_eye, iw, ih)
            avg_EAR = (right_EAR + left_EAR) / 2.0  # Average EAR for both eyes

            # Draw eye outlines
            def draw_eye_lines(eye_points):
                for i in range(len(eye_points) - 1):
                    x1, y1 = int(face_landmarks.landmark[eye_points[i]].x * iw), int(face_landmarks.landmark[eye_points[i]].y * ih)
                    x2, y2 = int(face_landmarks.landmark[eye_points[i + 1]].x * iw), int(face_landmarks.landmark[eye_points[i + 1]].y * ih)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            draw_eye_lines(right_eye)
            draw_eye_lines(left_eye)

            # Check for drowsiness
            if avg_EAR < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CLOSED_EYE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                frame_counter = 0  # Reset counter if eyes are open

            # Display EAR on screen
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Drowsiness Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera
cam.release()
cv2.destroyAllWindows()

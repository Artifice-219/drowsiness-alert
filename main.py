import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection and FaceMesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Open the camera
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection for bounding box
    face_results = face_detection.process(rgb_frame)

    # Face mesh detection for eye landmarks
    mesh_results = face_mesh.process(rgb_frame)

    ih, iw, _ = frame.shape  # Image dimensions

    # Draw bounding box around the face
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Draw eye outlines if face landmarks are detected
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Right Eye (Indices: Outer + Inner)
            right_eye = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]  
            # Left Eye (Indices: Outer + Inner)
            left_eye = [263, 387, 385, 362, 373, 380, 33, 160, 158, 133, 153, 144]

            # Function to draw lines connecting the landmarks
            def draw_eye_lines(eye_points):
                for i in range(len(eye_points) - 1):
                    x1, y1 = int(face_landmarks.landmark[eye_points[i]].x * iw), int(face_landmarks.landmark[eye_points[i]].y * ih)
                    x2, y2 = int(face_landmarks.landmark[eye_points[i + 1]].x * iw), int(face_landmarks.landmark[eye_points[i + 1]].y * ih)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            draw_eye_lines(right_eye)  # Draw lines for right eye
            draw_eye_lines(left_eye)   # Draw lines for left eye

    # Show the frame
    cv2.imshow('Face + Eyes Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera resources
cam.release()
cv2.destroyAllWindows()

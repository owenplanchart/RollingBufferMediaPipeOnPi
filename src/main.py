import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect only one face
    refine_landmarks=True,  # Use refined landmarks for better accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Define indices for the left and right eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Right eye landmarks

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect blinks and draw eye landmarks
def detect_blinks_and_draw_landmarks(face_roi):
    # Convert the face region to RGB
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    # Process the face region with MediaPipe Face Mesh
    results = face_mesh.process(face_roi_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye landmarks
            left_eye = np.array([
                (face_landmarks.landmark[idx].x * face_roi.shape[1],
                 face_landmarks.landmark[idx].y * face_roi.shape[0])
                for idx in LEFT_EYE_INDICES
            ])
            right_eye = np.array([
                (face_landmarks.landmark[idx].x * face_roi.shape[1],
                 face_landmarks.landmark[idx].y * face_roi.shape[0])
                for idx in RIGHT_EYE_INDICES
            ])

            # Draw eye landmarks on the face region
            for (x, y) in left_eye:
                cv2.circle(face_roi, (int(x), int(y)), 2, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(face_roi, (int(x), int(y)), 2, (255, 0, 0), -1)

            # Calculate the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # If EAR is below threshold, consider it a blink
            if ear < 0.2:
                return True

    return False

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Get the dimensions of the webcam feed
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load the movie file
    movie_path = "LookingUpAtTheSun.mp4"
    movie_cap = cv2.VideoCapture(movie_path)

    # Check if the movie file was opened successfully
    if not movie_cap.isOpened():
        print(f"Error: Could not open movie file {movie_path}.")
        return

    # Get the total number of frames in the movie
    total_frames = int(movie_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Flag to track if a blink is currently happening
    blink_in_progress = False

    # Variable to store the current movie frame
    current_movie_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Always show the bounding box + eye landmarks from the camera feed
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Draw bounding box on the live camera feed
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the face ROI from the camera feed
                face_roi = frame[y:y+h, x:x+w]

                # Detect blinks in the camera feed
                blink_detected = detect_blinks_and_draw_landmarks(face_roi)

                if blink_detected and not blink_in_progress:
                    # On blink, pick a random frame from the movie
                    random_frame_number = random.randint(0, total_frames - 1)
                    movie_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
                    ret_movie, current_movie_frame = movie_cap.read()
                    if ret_movie:
                        # Resize to match webcam dimensions
                        current_movie_frame = cv2.resize(current_movie_frame, (webcam_width, webcam_height))
                    else:
                        print("Error: Could not read random frame from the movie.")
                    blink_in_progress = True
                elif not blink_detected:
                    blink_in_progress = False

        # If we have a valid movie frame from the last blink, show it in a SECOND window
        # so the main window always shows the live camera feed with overlays.
        if current_movie_frame is not None:
            cv2.imshow("Random Movie Frame", current_movie_frame)
        else:
            # If no blink has happened or reading failed, close the movie window if open
            cv2.destroyWindow("Random Movie Frame")

        # Show the camera feed with bounding box & landmarks
        cv2.imshow("Camera Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    movie_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

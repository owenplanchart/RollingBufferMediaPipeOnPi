import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Detect only one face
    refine_landmarks=True,  # Use refined landmarks for better accuracy
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define indices for the left and right eye landmarks
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Right eye landmarks

# Function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    # Compute the Euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Compute the Euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to detect blinks and draw eye landmarks
def detect_blinks(frame):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye landmarks
            left_eye = np.array([(face_landmarks.landmark[idx].x * frame.shape[1], 
                                  face_landmarks.landmark[idx].y * frame.shape[0]) 
                                 for idx in LEFT_EYE_INDICES])
            right_eye = np.array([(face_landmarks.landmark[idx].x * frame.shape[1], 
                                   face_landmarks.landmark[idx].y * frame.shape[0]) 
                                  for idx in RIGHT_EYE_INDICES])

            # Calculate the eye aspect ratio for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            # Average the EAR for both eyes
            ear = (left_ear + right_ear) / 2.0

            # Check if the EAR is below the blink threshold
            if ear < 0.2:  # Adjust this threshold as needed
                cv2.putText(frame, "BLINK!", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Draw the eye landmarks on the frame
            for (x, y) in left_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)  # Green dots for left eye
            for (x, y) in right_eye:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Blue dots for right eye

    return frame

# Main function to capture video and detect blinks
def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect blinks and draw eye landmarks in the frame
        frame = detect_blinks(frame)

        # Display the frame
        cv2.imshow("Blink Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import cv2
import mediapipe as mp
import numpy as np
import random

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
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

# Function to detect blinks
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
                return True  # Blink detected

    return False  # No blink detected

# Main function
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect blinks in the frame
        if detect_blinks(frame):
            if not blink_in_progress:
                # Blink just started: select a random frame from the movie
                random_frame_number = random.randint(0, total_frames - 1)
                movie_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
                ret, movie_frame = movie_cap.read()
                if not ret:
                    print("Error: Could not read random frame from the movie.")
                    continue

                # Resize the movie frame to match the webcam feed dimensions
                movie_frame = cv2.resize(movie_frame, (webcam_width, webcam_height))
                blink_in_progress = True
        else:
            # Blink ended: reset the flag
            blink_in_progress = False

        # Display the movie frame if a blink is in progress
        if blink_in_progress:
            frame = movie_frame  # Replace the webcam frame with the resized movie frame

        # Display the frame
        cv2.imshow("Blink Triggered Movie Frame", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and movie capture objects
    cap.release()
    movie_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
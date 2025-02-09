import os
import cv2
import mediapipe as mp
import numpy as np
import random

# Suppress MediaPipe and TensorFlow Lite logs
os.environ['GLOG_minloglevel'] = '2'  # Only show WARNING and ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

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
def detect_blinks_and_draw_landmarks(frame, face_roi, x, y):
    # Check if the face region is valid
    if face_roi is None or face_roi.size == 0:
        return False

    # Convert the face region to RGB
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

    # Process the face region with MediaPipe Face Mesh
    results = face_mesh.process(face_roi_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract left and right eye landmarks
            left_eye = np.array([(face_landmarks.landmark[idx].x * face_roi.shape[1], 
                                  face_landmarks.landmark[idx].y * face_roi.shape[0]) 
                                 for idx in LEFT_EYE_INDICES])
            right_eye = np.array([(face_landmarks.landmark[idx].x * face_roi.shape[1], 
                                   face_landmarks.landmark[idx].y * face_roi.shape[0]) 
                                  for idx in RIGHT_EYE_INDICES])

            # Draw eye landmarks on the full frame (not just the face region)
            for (ex, ey) in left_eye:
                cv2.circle(frame, (int(ex) + x, int(ey) + y), 2, (0, 255, 0), -1)  # Green dots for left eye
            for (ex, ey) in right_eye:
                cv2.circle(frame, (int(ex) + x, int(ey) + y), 2, (255, 0, 0), -1)  # Blue dots for right eye

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

    # Variable to store the current movie frame
    current_movie_frame = None

    # Variables to store the face bounding box coordinates
    face_x, face_y, face_w, face_h = 0, 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the frame
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                # Get the bounding box of the face
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                face_x = int(bboxC.xmin * iw)
                face_y = int(bboxC.ymin * ih)
                face_w = int(bboxC.width * iw)
                face_h = int(bboxC.height * ih)

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

                # Crop the face region
                face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]

                # Detect blinks and draw eye landmarks on the full frame
                blink_detected = detect_blinks_and_draw_landmarks(frame, face_roi, face_x, face_y)

                if blink_detected:
                    if not blink_in_progress:
                        # Blink just started: select a random frame from the movie
                        random_frame_number = random.randint(0, total_frames - 1)
                        movie_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
                        ret, current_movie_frame = movie_cap.read()
                        if not ret:
                            print("Error: Could not read random frame from the movie.")
                            continue

                        # Resize the movie frame to match the webcam feed dimensions
                        current_movie_frame = cv2.resize(current_movie_frame, (webcam_width, webcam_height))
                        blink_in_progress = True
                else:
                    # Blink ended: reset the flag
                    blink_in_progress = False

        # Display the movie frame if a blink is in progress
        if blink_in_progress:
            # Create a copy of the movie frame to avoid modifying the original
            display_frame = current_movie_frame.copy()
           
            
            

            # Redraw the face rectangle and eye landmarks on the movie frame
            if results.detections:
                # Draw the bounding box on the movie frame
                cv2.rectangle(display_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

                # Redraw eye landmarks on the movie frame
                detect_blinks_and_draw_landmarks(display_frame, face_roi, face_x, face_y)

                 # Crop the face region from the movie frame
                #face_roi = display_frame[face_y:face_y+face_h, face_x:face_x+face_w]

            # Display the modified movie frame
            cv2.imshow("Blink Triggered Movie Frame", display_frame)
        else:
            # Display the webcam feed with face rectangle and eye landmarks
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
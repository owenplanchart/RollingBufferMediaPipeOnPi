import os
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Suppress MediaPipe and TensorFlow Lite logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def detect_blinks_and_draw_landmarks(frame, face_roi, x, y):
    if face_roi is None or face_roi.size == 0:
        return False
    face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(face_roi_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = np.array([(face_landmarks.landmark[idx].x * face_roi.shape[1], 
                                  face_landmarks.landmark[idx].y * face_roi.shape[0]) 
                                 for idx in LEFT_EYE_INDICES])
            right_eye = np.array([(face_landmarks.landmark[idx].x * face_roi.shape[1], 
                                   face_landmarks.landmark[idx].y * face_roi.shape[0]) 
                                  for idx in RIGHT_EYE_INDICES])
            for (ex, ey) in left_eye:
                cv2.circle(frame, (int(ex) + x, int(ey) + y), 2, (0, 255, 0), -1)
            for (ex, ey) in right_eye:
                cv2.circle(frame, (int(ex) + x, int(ey) + y), 2, (255, 0, 0), -1)
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            if ear < 0.2:
                return True
    return False

def main():
    cap = cv2.VideoCapture(0)
    webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    movie_path = "LookingUpAtTheSun.mp4"
    movie_cap = cv2.VideoCapture(movie_path)
    if not movie_cap.isOpened():
        print(f"Error: Could not open movie file {movie_path}.")
        return
    total_frames = int(movie_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    blink_in_progress = False
    current_movie_frame = None
    face_x, face_y, face_w, face_h = 0, 0, 0, 0

    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                face_x = int(bboxC.xmin * iw)
                face_y = int(bboxC.ymin * ih)
                face_w = int(bboxC.width * iw)
                face_h = int(bboxC.height * ih)
                cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
                face_roi = frame[face_y:face_y+face_h, face_x:face_x+face_w]
                blink_detected = detect_blinks_and_draw_landmarks(frame, face_roi, face_x, face_y)
                if blink_detected:
                    if not blink_in_progress:
                        random_frame_number = random.randint(0, total_frames - 1)
                        movie_cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
                        ret, current_movie_frame = movie_cap.read()
                        if not ret:
                            print("Error: Could not read random frame from the movie.")
                            continue
                        current_movie_frame = cv2.resize(current_movie_frame, (webcam_width, webcam_height))
                        blink_in_progress = True
                else:
                    blink_in_progress = False
        elapsed_time = time.time() - start_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if blink_in_progress:
            display_frame = current_movie_frame.copy()
            if results.detections:
                cv2.rectangle(display_frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
                detect_blinks_and_draw_landmarks(display_frame, face_roi, face_x, face_y)
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Blink Triggered Movie Frame", display_frame)
        else:
            cv2.imshow("Blink Triggered Movie Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    movie_cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from picamera2 import Picamera2
from libcamera import Transform
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque

def euclidean_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

def compute_ear(eye_points):
    # Compute Eye Aspect Ratio (EAR)
    p1, p2, p3, p4, p5, p6 = eye_points
    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def main():
    # Initialize MediaPipe Face Detection and Face Mesh.
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Define landmark indices for eye blink detection.
    left_eye_indices = [33, 159, 158, 133, 153, 144]
    right_eye_indices = [362, 386, 387, 263, 373, 380]
    
    # Increase threshold for blink detection for higher sensitivity.
    BLINK_THRESHOLD = 0.25  # Adjusted for more sensitivity

    # Initialize Picamera2 with vertical flip.
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        # Picamera2's RGB888 format is stored as BGR bytes, matching OpenCV.
        main={"size": (640, 480), "format": "RGB888"},
        # Keep the camera stream at 30 fps; low light may increase gain/noise.
        controls={"FrameDurationLimits": (33333, 33333)},
        transform=Transform(vflip=1))
    picam2.configure(config)
    picam2.start()
    print("Camera started successfully.")

    # Create display window.
    cv2.namedWindow("Blink Detection", cv2.WINDOW_NORMAL)
    prev_time = time.monotonic()
    fps_window_start = prev_time
    fps_frame_count = 0
    inference_interval = 2
    frame_number = 0
    ear_history = deque(maxlen=3)
    latest_state = None

    try:
        while True:
            frame = picam2.capture_array()
            ih, iw, _ = frame.shape
            run_inference = frame_number % inference_interval == 0
            frame_number += 1

            if run_inference:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detection_results = face_detector.process(image_rgb)
                latest_state = None
                if not detection_results.detections:
                    ear_history.clear()

                for detection in detection_results.detections or []:
                    bboxC = detection.location_data.relative_bounding_box
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    x_end = min(int((bboxC.xmin + bboxC.width) * iw), iw)
                    y_end = min(int((bboxC.ymin + bboxC.height) * ih), ih)

                    # Ensure ROI stays within frame boundaries.
                    roi_rgb = image_rgb[y:y_end, x:x_end]
                    actual_h, actual_w = roi_rgb.shape[:2]
                    if actual_h == 0 or actual_w == 0:
                        continue

                    # Pad ROI to square.
                    square_side = max(actual_w, actual_h)
                    square_roi = np.zeros((square_side, square_side, 3), dtype=np.uint8)
                    offset_x = (square_side - actual_w) // 2
                    offset_y = (square_side - actual_h) // 2
                    square_roi[offset_y:offset_y+actual_h, offset_x:offset_x+actual_w] = roi_rgb
                    
                    mesh_results = face_mesh.process(square_roi)
                    if mesh_results.multi_face_landmarks:
                        for face_landmarks in mesh_results.multi_face_landmarks:
                            landmarks = []
                            for lm in face_landmarks.landmark:
                                lm_x = lm.x * square_side
                                lm_y = lm.y * square_side
                                lm_x_adj = lm_x - offset_x
                                lm_y_adj = lm_y - offset_y
                                landmarks.append((int(lm_x_adj), int(lm_y_adj)))
                            
                            try:
                                left_eye_pts = [landmarks[i] for i in left_eye_indices]
                                right_eye_pts = [landmarks[i] for i in right_eye_indices]
                            except IndexError:
                                continue

                            left_eye_pts_full = [(pt[0] + x, pt[1] + y) for pt in left_eye_pts]
                            right_eye_pts_full = [(pt[0] + x, pt[1] + y) for pt in right_eye_pts]

                            ear_left = compute_ear(left_eye_pts_full)
                            ear_right = compute_ear(right_eye_pts_full)
                            ear_avg = (ear_left + ear_right) / 2.0
                            ear_history.append(ear_avg)
                            smoothed_ear = sum(ear_history) / len(ear_history)
                            display_text = "BLINK" if smoothed_ear < BLINK_THRESHOLD else f"EAR: {smoothed_ear:.2f}"
                            latest_state = {
                                "left_eye": left_eye_pts_full,
                                "right_eye": right_eye_pts_full,
                                "x": x,
                                "y": y,
                                "width": actual_w,
                                "height": actual_h,
                                "text": display_text,
                            }
                            break

            if latest_state:
                for pt in latest_state["left_eye"] + latest_state["right_eye"]:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)

                x, y = latest_state["x"], latest_state["y"]
                w, h = latest_state["width"], latest_state["height"]
                display_text = latest_state["text"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                font_scale = 2.0
                thickness = 3
                text_size, _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = x + (w - text_size[0]) // 2
                text_y = y - 20 if y - 20 > 20 else y + 20
                color = (0, 0, 255) if display_text == "BLINK" else (0, 255, 0)
                cv2.putText(frame, display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            current_time = time.monotonic()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            fps_frame_count += 1
            fps_window_elapsed = current_time - fps_window_start
            if fps_window_elapsed >= 5:
                fps = fps_frame_count / fps_window_elapsed
                print(f"Average FPS: {fps:.1f}", flush=True)
                fps_window_start = current_time
                fps_frame_count = 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Blink Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.005)
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")

if __name__ == '__main__':
    main()

from picamera2 import Picamera2
from libcamera import Transform
import cv2
import mediapipe as mp
import time
import math
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient


CAPTURE_FPS = 60
CLIP_PLAYBACK_FPS = 30
BUFFER_SECONDS = 1
BUFFER_FRAMES = CAPTURE_FPS * BUFFER_SECONDS
ANALYSIS_MAX_SIZE = (640, 360)
OSC_BIND_ADDRESS = "0.0.0.0"
OSC_PORT = 9993
OSC_RECORD_ADDRESS = "/recordBlink"
OSC_ACK_PORT = 9992
OSC_ACK_ADDRESS = "/recordBlinkAck"
PREVIEW_WINDOW = "Blink Detection"


class ClipWriter:
    """Writes completed clips away from the camera and inference loop."""

    def __init__(self):
        self._clips = Queue()
        self._thread = Thread(target=self._write_clips, daemon=True)
        self._thread.start()

    def submit(self, path, frames, ack_ip=None):
        self._clips.put((path, frames, ack_ip))

    def close(self):
        self._clips.join()
        self._clips.put(None)
        self._thread.join()

    def _write_clips(self):
        while True:
            clip = self._clips.get()
            if clip is None:
                self._clips.task_done()
                return

            path, frames, ack_ip = clip
            try:
                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    CLIP_PLAYBACK_FPS,
                    (width, height),
                )
                if not writer.isOpened():
                    print(f"Could not open video file: {path}", flush=True)
                    continue

                for frame in frames:
                    writer.write(frame)
                writer.release()
                if not path.is_file() or path.stat().st_size == 0:
                    print(f"Video file was not saved correctly: {path}", flush=True)
                    continue

                print(f"Saved blink clip: {path}", flush=True)
                if ack_ip is not None:
                    try:
                        ack_client = SimpleUDPClient(ack_ip, OSC_ACK_PORT)
                        ack_client.send_message(OSC_ACK_ADDRESS, path.name)
                        print(
                            f"Sent OSC {OSC_ACK_ADDRESS} to "
                            f"{ack_ip}:{OSC_ACK_PORT} for {path.name}.",
                            flush=True,
                        )
                    except OSError as error:
                        print(
                            f"Could not send OSC acknowledgement to "
                            f"{ack_ip}:{OSC_ACK_PORT}: {error}",
                            flush=True,
                        )
            finally:
                self._clips.task_done()


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


def zoom_crop(frame, zoom):
    """Return the centre crop used for both preview and analysis."""
    height, width = frame.shape[:2]
    crop_width = max(2, int(width / zoom))
    crop_height = max(2, int(height / zoom))
    x_start = (width - crop_width) // 2
    y_start = (height - crop_height) // 2
    return frame[y_start:y_start + crop_height, x_start:x_start + crop_width]


def apply_zoom(frame, zoom):
    """Centre-crop and resize a frame for the preview window."""
    if zoom <= 1.0:
        return frame

    cropped = zoom_crop(frame, zoom)
    height, width = frame.shape[:2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def resize_for_analysis(frame):
    """Limit MediaPipe input size while preserving the preview aspect ratio."""
    height, width = frame.shape[:2]
    max_width, max_height = ANALYSIS_MAX_SIZE
    scale = min(1.0, max_width / width, max_height / height)
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def main():
    external_triggers = Queue()

    def queue_external_trigger(client_address, address, *args):
        external_triggers.put(client_address[0])

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
    BLINK_THRESHOLD = 0.30  # Higher threshold for more sensitive blink detection.

    # Both modes keep the camera running for the rolling buffer. The selected
    # mode determines whether a local blink or an OSC message starts a clip.
    picam2 = None

    def start_camera():
        nonlocal picam2
        if picam2 is not None:
            return
        camera = Picamera2()
        config = camera.create_preview_configuration(
            # Picamera2's RGB888 format is stored as BGR bytes, matching OpenCV.
            main={"size": (960, 540), "format": "RGB888"},
            # Keep the camera stream at 60 fps; low light may increase gain/noise.
            controls={"FrameDurationLimits": (16666, 16666)},
            transform=Transform(vflip=1),
        )
        camera.configure(config)
        camera.start()
        picam2 = camera
        print("Camera started successfully.", flush=True)

    def stop_camera():
        nonlocal picam2
        if picam2 is None:
            return
        picam2.stop()
        picam2.close()
        picam2 = None
        print("Camera stopped.", flush=True)

    # Mode and overlay changes do not interrupt the camera or rolling buffer.
    overlays_enabled = True
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    start_camera()
    prev_time = time.monotonic()
    fps_window_start = prev_time
    fps_frame_count = 0
    # Run landmark inference at roughly 20 fps while preserving 60 fps capture.
    inference_interval = 3
    frame_number = 0
    ear_history = deque(maxlen=3)
    latest_state = None
    zoom = 1.0
    zoom_step = 0.1
    max_zoom = 3.0
    pre_buffer = deque(maxlen=BUFFER_FRAMES)
    clip_frames = None
    clip_ack_ip = None
    post_frames_remaining = 0
    blink_latched = False
    clip_directory = Path(__file__).resolve().parents[1] / "clips"
    clip_directory.mkdir(exist_ok=True)
    clip_writer = ClipWriter()
    osc_dispatcher = Dispatcher()
    osc_dispatcher.map(
        OSC_RECORD_ADDRESS,
        queue_external_trigger,
        needs_reply_address=True,
    )
    osc_server = ThreadingOSCUDPServer(
        (OSC_BIND_ADDRESS, OSC_PORT), osc_dispatcher
    )
    osc_thread = Thread(target=osc_server.serve_forever, daemon=True)
    osc_thread.start()
    print(
        f"Listening for OSC {OSC_RECORD_ADDRESS} on "
        f"{OSC_BIND_ADDRESS}:{OSC_PORT}.",
        flush=True,
    )
    mode = "osc"
    print(
        "OSC trigger mode enabled. Press c for blink trigger mode or g to "
        "toggle the GUI overlays.",
        flush=True,
    )

    try:
        while True:
            camera_frame = picam2.capture_array()
            frame = apply_zoom(camera_frame, zoom)
            ih, iw, _ = frame.shape
            analysis_frame = resize_for_analysis(zoom_crop(camera_frame, zoom))
            analysis_h, analysis_w = analysis_frame.shape[:2]
            analysis_to_preview_x = iw / analysis_w
            analysis_to_preview_y = ih / analysis_h
            run_inference = mode == "camera" and frame_number % inference_interval == 0
            frame_number += 1
            blink_now = False

            if run_inference:
                image_rgb = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2RGB)
                detection_results = face_detector.process(image_rgb)
                latest_state = None
                if not detection_results.detections:
                    ear_history.clear()

                for detection in detection_results.detections or []:
                    bboxC = detection.location_data.relative_bounding_box
                    x = max(0, int(bboxC.xmin * analysis_w))
                    y = max(0, int(bboxC.ymin * analysis_h))
                    x_end = min(int((bboxC.xmin + bboxC.width) * analysis_w), analysis_w)
                    y_end = min(int((bboxC.ymin + bboxC.height) * analysis_h), analysis_h)

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

                            left_eye_pts_full = [
                                (int((pt[0] + x) * analysis_to_preview_x),
                                 int((pt[1] + y) * analysis_to_preview_y))
                                for pt in left_eye_pts
                            ]
                            right_eye_pts_full = [
                                (int((pt[0] + x) * analysis_to_preview_x),
                                 int((pt[1] + y) * analysis_to_preview_y))
                                for pt in right_eye_pts
                            ]

                            ear_left = compute_ear(left_eye_pts_full)
                            ear_right = compute_ear(right_eye_pts_full)
                            ear_avg = (ear_left + ear_right) / 2.0
                            ear_history.append(ear_avg)
                            smoothed_ear = sum(ear_history) / len(ear_history)
                            blink_now = smoothed_ear < BLINK_THRESHOLD
                            display_text = "BLINK" if blink_now else f"EAR: {smoothed_ear:.2f}"
                            latest_state = {
                                "left_eye": left_eye_pts_full,
                                "right_eye": right_eye_pts_full,
                                "x": int(x * analysis_to_preview_x),
                                "y": int(y * analysis_to_preview_y),
                                "width": int(actual_w * analysis_to_preview_x),
                                "height": int(actual_h * analysis_to_preview_y),
                                "text": display_text,
                            }
                            break

            clip_start_pending = False
            clip_start_ack_ip = None
            if run_inference:
                if blink_now:
                    if not blink_latched and clip_frames is None and len(pre_buffer) == BUFFER_FRAMES:
                        blink_latched = True
                        clip_start_pending = True
                        print("Blink detected; recording the next 1 second.", flush=True)
                else:
                    blink_latched = False

            while True:
                try:
                    trigger_ip = external_triggers.get_nowait()
                except Empty:
                    break
                print("External OSC trigger /recordBlink received.", flush=True)
                if mode != "osc":
                    print(
                        "Ignoring external OSC trigger: blink trigger mode is active.",
                        flush=True,
                    )
                elif len(pre_buffer) < BUFFER_FRAMES:
                    print(
                        "Ignoring external OSC trigger: pre-buffer is still warming up.",
                        flush=True,
                    )
                elif clip_frames is not None or clip_start_pending:
                    print(
                        "Ignoring external OSC trigger: a clip is already being collected.",
                        flush=True,
                    )
                else:
                    clip_start_pending = True
                    clip_start_ack_ip = trigger_ip
                    print(
                        "External OSC trigger accepted; recording the next 1 second.",
                        flush=True,
                    )

            if overlays_enabled and latest_state:
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
            if overlays_enabled:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Zoom: {zoom:.1f}x", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                mode_text = (
                    "OSC trigger mode" if mode == "osc" else "Blink trigger mode"
                )
                cv2.putText(frame, mode_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                if clip_frames is not None or clip_start_pending:
                    cv2.putText(frame, "REC", (10, 135),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Store exactly what is shown so clips follow the overlay setting.
            pre_buffer.append(frame.copy())
            if clip_start_pending:
                clip_frames = list(pre_buffer)
                clip_ack_ip = clip_start_ack_ip
                post_frames_remaining = BUFFER_FRAMES
            elif clip_frames is not None:
                clip_frames.append(frame.copy())
                post_frames_remaining -= 1
                if post_frames_remaining == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clip_path = clip_directory / f"blink_{timestamp}.mp4"
                    clip_writer.submit(clip_path, clip_frames, clip_ack_ip)
                    clip_frames = None
                    clip_ack_ip = None
            
            cv2.imshow(PREVIEW_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(']'):
                zoom = min(max_zoom, round(zoom + zoom_step, 1))
                print(f"Zoom: {zoom:.1f}x", flush=True)
            elif key == ord('['):
                zoom = max(1.0, round(zoom - zoom_step, 1))
                print(f"Zoom: {zoom:.1f}x", flush=True)
            elif key == ord('o'):
                mode = "osc"
                latest_state = None
                ear_history.clear()
                blink_latched = False
                print("OSC trigger mode enabled.", flush=True)
            elif key == ord('c'):
                mode = "camera"
                latest_state = None
                ear_history.clear()
                blink_latched = False
                print("Blink trigger mode enabled.", flush=True)
            elif key == ord('g'):
                overlays_enabled = not overlays_enabled
                state = "enabled" if overlays_enabled else "disabled"
                print(f"GUI overlays {state}.", flush=True)
            elif key == ord('q'):
                break
            time.sleep(0.005)
    finally:
        osc_server.shutdown()
        osc_server.server_close()
        osc_thread.join()
        stop_camera()
        clip_writer.close()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")

if __name__ == '__main__':
    main()

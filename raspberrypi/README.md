# Blink Rolling Buffer

Real-time blink detection for a Raspberry Pi 5 and Camera Module 3 Wide. When a blink is detected, the application saves a slow-motion MP4 clip with the second before and the second after the event.

## Requirements

- Raspberry Pi 5 with a supported CSI camera
- Raspberry Pi OS Bookworm or later
- Picamera2 and libcamera installed through `apt`

Install the camera stack if necessary:

```bash
sudo apt update
sudo apt install python3-picamera2 rpicam-apps
```

## Setup

```bash
cd raspberrypi
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`--system-site-packages` makes the Raspberry Pi OS Picamera2 package available inside the virtual environment.

## Run

```bash
cd raspberrypi
source .venv/bin/activate
python3 src/blink.py
```

Press `q` in the preview window to stop the application.

## Controls

- `]`: zoom in by 0.1x
- `[`: zoom out by 0.1x
- `q`: quit

## Blink clips

The camera is configured for 60 fps and the application keeps a 60-frame circular buffer. A blink saves 60 frames before and 60 frames after the event. Clips are encoded at 30 fps, so all 120 captured frames play back as a four-second slow-motion video.

Saved clips are written to `raspberrypi/clips/` and include the same blink, landmark, FPS, zoom, and recording overlays shown in the preview.

The 960x540 preview and recording stream is reduced to a maximum of 640x360 before MediaPipe analysis, while captured clips retain the higher resolution. In testing, this higher-resolution mode ran below the 60 fps target; use 640x480 if maintaining the closest possible 2x slow-motion rate is more important than output resolution.

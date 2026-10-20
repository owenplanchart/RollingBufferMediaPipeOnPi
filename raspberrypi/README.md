# Blink Rolling Buffer

Real-time blink detection for a Raspberry Pi 5 and Camera Module 3 Wide. When a blink is detected, the application saves a slow-motion MP4 clip with 1.5 seconds before and 1.5 seconds after the event.

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

The application always keeps the camera and rolling buffer active. It starts in
OSC trigger mode: press `o` to record clips only when `/recordBlink` arrives,
or press `c` to record clips only when the camera detects a blink. The preview
shows the current mode. Press `g` to hide or show all overlays while keeping the
camera view open. The selected overlay setting also applies to recorded clips.
Press `q` to stop the application.

## Desktop launcher

After completing Setup, install the desktop and application-menu shortcuts once:

```bash
cd raspberrypi
python3 install-desktop.py
```

Open **Blink Rolling Buffer** from the desktop or application menu. If the desktop
asks, allow launching or mark the shortcut trusted. Press `q` in the preview
window to quit. The existing keyboard controls and OSC configuration still apply.
Only the shortcuts are installed; no application files or dependencies are copied.

The launcher runs the current `src/blink.py` using `.venv/bin/python` beside it.
Code changes and Git updates take effect the next time you launch the app; no
rebuild is needed. If you move the project, rerun `install-desktop.py` from the
new location. To use another environment, set `BLINK_PYTHON` to its absolute
Python executable path. Additional launcher arguments are forwarded to `blink.py`.
You can also run `python3 launch.py` directly.

Output is saved in `~/.local/state/blink-rolling-buffer/blink.log` (or under
`XDG_STATE_HOME` when set). Logs rotate at 2 MiB, keeping three backups.
Startup failures show a graphical message when `zenity` is available.
The launcher prevents duplicate launches through either shortcut; manually
running `src/blink.py` bypasses that lock, so quit any terminal-started instance
before using the desktop shortcut. Dependency installation remains the explicit
Setup step above; the launcher does not install or upgrade packages.

To remove the desktop integration, delete the **Blink Rolling Buffer** desktop
shortcut and `~/.local/share/applications/blink-rolling-buffer.desktop` (or its
equivalent under `XDG_DATA_HOME`). The project remains runnable from the terminal.

## Launch automatically at startup

On this Pi, `~/.config/autostart/blink-rolling-buffer.desktop` runs
`autostart.py` at graphical login. Desktop auto-login is already enabled.
The wrapper waits 10 seconds, then runs the existing `launch.py`, reusing the
virtual environment, rotating logs, and duplicate launch protection.
The delay starts at desktop login, so total time from power-on also includes
boot time. Source changes take effect on the next launch without a rebuild.

Closing the app leaves it closed until the next login or manual launch.
To disable automatic launching, remove
`~/.config/autostart/blink-rolling-buffer.desktop`; the manual desktop shortcut
will still work. If the project moves, update the autostart entry's `Exec` path.

A full reboot is needed to verify the complete power-on sequence, preview,
and Ethernet OSC triggers and acknowledgements.

## External OSC trigger

In addition to local MediaPipe blink detection, the application accepts an OSC
message at `/recordBlink` on UDP port `9993`. For the current installation, send
the message over the direct Ethernet connection from the Mac Mini at
`192.168.50.1` to the Raspberry Pi at `192.168.50.2`. Both Ethernet interfaces
use the `192.168.50.0/24` subnet with no gateway, so Wi-Fi can remain available
for internet access and other devices.

The server binds to `0.0.0.0:9993`, so it accepts OSC triggers arriving through
Ethernet or Wi-Fi. The direct Ethernet address is preferred for the installation
because it avoids relying on Wi-Fi for recording triggers.

To send a test trigger from another computer (including the Mac Mini), install
`python-osc` and run:

```bash
python3 -m pip install python-osc
python3 -c 'from pythonosc.udp_client import SimpleUDPClient; SimpleUDPClient("192.168.50.2", 9993).send_message("/recordBlink", 1.0)'
```

The Pi ignores triggers while its 1.5-second pre-buffer is warming up or while
a clip is already being collected.

After an externally triggered clip has been saved successfully, the Pi sends
`/recordBlinkAck` to UDP port `9992` at the IP address that sent the trigger.
For the direct connection, this is the Mac Mini at `192.168.50.1:9992`. The
saved filename is included as the sole OSC message argument. The acknowledgement
is sent only after the video writer has closed the file and confirmed that it
exists and is non-empty. No acknowledgement is sent for an ignored trigger or a
clip that could not be saved. Failure to send the acknowledgement is logged and
does not stop the rolling-buffer receiver.

## Controls

- `c`: camera blink-trigger mode (OSC messages are ignored)
- `o`: OSC-trigger mode (local blink detection is disabled)
- `g`: toggle overlays in the camera view and recorded clips
- `]`: zoom in by 0.1x
- `[`: zoom out by 0.1x
- `r`: rotate 90 degrees clockwise (0°, 90°, 180°, 270°, then back to 0°)
- `q`: quit

Click the preview window to give it keyboard focus before using these controls.
Rotation applies to the preview, blink detection, and saved clips, relative to
the existing camera vertical flip. Quarter turns produce a 540x960 portrait
image without stretching. The preview area automatically switches between
640x360 landscape and 360x640 portrait to match the image. You can still resize
the window manually between orientation changes. If a clip is being collected,
rotation waits until collection finishes. Each applied rotation clears the rolling buffer and
detection state; triggers are unavailable until the 90-frame buffer refills.
The rotation angle appears when overlays are enabled and defaults to 270° on restart.

## Blink clips

The camera is configured for 60 fps and the application keeps a 90-frame circular buffer. A trigger saves 90 frames before and 90 frames after the event (3 seconds total at the target capture rate). Clips are encoded at 30 fps, so all 180 captured frames play back as a six-second slow-motion video.

Saved clips are written to `raspberrypi/clips/` and include the same blink, landmark, FPS, zoom, and recording overlays shown in the preview.

The recording stream is 960x540 in landscape or 540x960 in portrait. MediaPipe
analysis uses at most 640x360 or 360x640 respectively; resizing the preview
window does not change the recording resolution. Capture targets 60 fps, but
actual throughput depends on processing load. The on-screen FPS is an
instantaneous reading; the log reports a steadier five-second average.

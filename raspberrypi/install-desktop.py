#!/usr/bin/env python3
"""Install or refresh shortcuts for this checkout; no root privileges needed."""

import os
from pathlib import Path
import shutil
import subprocess


def exec_quote(value):
    # Desktop Entry Exec quoting followed by the string-value escaping layer.
    value = str(value).replace("%", "%%")
    for char in ("\\", '"', "`", "$"):
        value = value.replace(char, "\\" + char)
    return '"' + value.replace("\\", "\\\\") + '"'


def main():
    launcher = Path(__file__).resolve().with_name("launch.py")
    data_root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share")
    applications = data_root / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    entry = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Blink Rolling Buffer\n"
        "Comment=Record rolling-buffer clips using OSC or blink detection\n"
        f"Exec=/usr/bin/python3 {exec_quote(launcher)}\n"
        "Icon=camera-video\n"
        "Terminal=false\n"
        "Categories=AudioVideo;Video;\n"
        "StartupNotify=false\n"
    )
    targets = [applications / "blink-rolling-buffer.desktop"]
    desktop = Path.home() / "Desktop"
    if shutil.which("xdg-user-dir"):
        result = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True)
        if result.stdout.strip():
            desktop = Path(result.stdout.strip())
    # XDG can disable the desktop by setting it to the home directory.
    if desktop != Path.home() and desktop.is_dir():
        targets.append(desktop / "blink-rolling-buffer.desktop")
    for target in targets:
        if target.exists() and "Name=Blink Rolling Buffer\n" not in target.read_text():
            raise SystemExit(f"Refusing to replace an unrelated shortcut: {target}")
    for target in targets:
        target.write_text(entry)
        target.chmod(0o755)
        print(f"Installed: {target}")
    print("Open Blink Rolling Buffer from the application menu or desktop. Press q to quit.")
    print("If the desktop requests permission to launch the shortcut, mark it trusted/allow launching.")


if __name__ == "__main__":
    main()

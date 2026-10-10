#!/usr/bin/env python3
"""Desktop entry point; runs the current source with the project's environment."""

import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent


def notify(message, error=False):
    print(message, file=sys.stderr, flush=True)
    if shutil.which("zenity") and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        subprocess.run(
            ["zenity", "--error" if error else "--info",
             "--title=Blink Rolling Buffer", f"--text={message}", "--no-markup"],
            check=False,
        )


def main():
    state_root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    state_dir = state_root / "blink-rolling-buffer"
    state_dir.mkdir(parents=True, exist_ok=True)
    # A user-wide lock also prevents copies of this checkout competing for port 9993.
    with (state_dir / "app.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            notify("Blink Rolling Buffer is already running. Use its preview window; press q to quit.")
            return 0

        log_path = state_dir / "blink.log"
        logger = logging.getLogger("blink-launcher")
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=3)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)

        # Optional override supports a different environment without editing this launcher.
        python = Path(os.environ.get("BLINK_PYTHON") or PROJECT / ".venv/bin/python").expanduser()
        if not python.is_file() or not os.access(python, os.X_OK):
            message = f"Python environment not found or executable: {python}\nFollow the Setup section in raspberrypi/README.md."
            logger.error(message)
            notify(message, error=True)
            return 1

        logger.info("Starting %s with %s", PROJECT / "src/blink.py", python)
        try:
            with subprocess.Popen(
                [str(python), "-u", str(PROJECT / "src/blink.py"), *sys.argv[1:]],
                cwd=PROJECT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                # Keep the lock while the app runs, even if the launcher is terminated.
                pass_fds=(lock.fileno(),),
            ) as process:
                for line in process.stdout:
                    logger.info("%s", line.rstrip("\n"))
                code = process.wait()
        except OSError as error:
            logger.exception("Could not launch application")
            notify(f"Could not start Blink Rolling Buffer: {error}\nLog: {log_path}", error=True)
            return 1

        logger.info("Application exited with code %s", code)
        if code:
            notify(f"Blink Rolling Buffer stopped with an error.\nDetails are in: {log_path}", error=True)
        return code if code >= 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except OSError as error:
        notify(f"Launcher error: {error}", error=True)
        sys.exit(1)

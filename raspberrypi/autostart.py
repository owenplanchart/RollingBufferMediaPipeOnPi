#!/usr/bin/env python3
"""Start the desktop launcher ten seconds after automatic graphical login."""

import os
from pathlib import Path
import time


if __name__ == "__main__":
    time.sleep(10)
    launcher = Path(__file__).resolve().with_name("launch.py")
    os.execv("/usr/bin/python3", ["/usr/bin/python3", str(launcher)])

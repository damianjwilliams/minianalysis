#!/usr/bin/env python
"""Pick a recording and a step from a window, then run it.

A thin wrapper so this step can be run as a plain script from the repo root:

    python launch.py path/to/recording.abf

Identical to `python -m minianalysis launch path/to/recording.abf`; all the
same options apply (`python launch.py --help`). The real code lives in
minianalysis/launch.py.
"""

import os
import sys

# Run-from-anywhere: put THIS file's directory (the repo root, which holds the
# minianalysis package) at the front of the import path, so double-clicking the
# script or calling it by absolute path works without installing anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minianalysis.launch import main  # noqa: E402  (import after the sys.path fix above)

if __name__ == "__main__":
    main()

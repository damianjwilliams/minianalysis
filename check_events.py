#!/usr/bin/env python
"""Check (and accept/reject) the events a detection run produced.

A thin wrapper so this step can be run as a plain script from the repo root:

    python check_events.py path/to/recording.abf

Identical to `python -m minianalysis check path/to/recording.abf`; all the
same options apply (`python check_events.py --help`). The real code lives in
minianalysis/check.py.
"""

import os
import sys

# Run-from-anywhere: put THIS file's directory (the repo root, which holds the
# minianalysis package) at the front of the import path, so double-clicking the
# script or calling it by absolute path works without installing anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minianalysis.check import main  # noqa: E402  (import after the sys.path fix above)

if __name__ == "__main__":
    main()

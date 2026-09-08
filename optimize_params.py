#!/usr/bin/env python
"""Tune the Mini-Analysis-style detection parameters interactively.

A thin wrapper so this step can be run as a plain script from the repo root:

    python optimize_params.py path/to/recording.abf

Identical to `python -m minianalysis optimize path/to/recording.abf`; all the
same options apply (`python optimize_params.py --help`). The real code lives in
minianalysis/optimize.py.
"""

import os
import sys

# Run-from-anywhere: put THIS file's directory (the repo root, which holds the
# minianalysis package) at the front of the import path, so double-clicking the
# script or calling it by absolute path works without installing anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minianalysis.optimize import main  # noqa: E402  (import after the sys.path fix above)

if __name__ == "__main__":
    main()

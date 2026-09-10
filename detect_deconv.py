#!/usr/bin/env python
"""Detect events by deconvolution, then write the events CSV.

A thin wrapper so this step can be run as a plain script from the repo root:

    python detect_deconv.py path/to/recording.abf
    python detect_deconv.py --benchmark

Identical to `python -m minianalysis deconvolve path/to/recording.abf`; all the
same options apply (`python detect_deconv.py --help`). The real code lives in
minianalysis/deconvolve.py.
"""

import os
import sys

# Run-from-anywhere: put THIS file's directory (the repo root, which holds the
# minianalysis package) at the front of the import path, so double-clicking the
# script or calling it by absolute path works without installing anything.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minianalysis.deconvolve import main  # noqa: E402  (import after the sys.path fix above)

if __name__ == "__main__":
    sys.exit(main())

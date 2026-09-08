"""Make the repo root importable, so `pytest` finds the minianalysis
package without the repo having to be pip-installed first (plain `pytest`
only puts THIS directory on sys.path, unlike `python -m pytest`, which also
adds the working directory)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

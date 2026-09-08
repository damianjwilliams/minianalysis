"""
Shared visual constants for every plot/GUI in this package -- one source of
truth so the trace, the annotations and the QC markers are drawn the same
way in the batch plots (run.py), the optimizer (optimize.py) and the event
checker (check.py).

Tokens come from a validated colorblind-safe light-mode palette.
"""

# chart chrome tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
TRACE = "#333333"  # raw-trace line color, distinct from ink/muted so it recedes behind markers

"""
minianalysis -- a standalone, Mini-Analysis-style synaptic event detector
for gap-free voltage-clamp .abf recordings, with the two tools you need
around it: one to tune the detection parameters, one to check what came out.

Three steps, in the order you'd use them
----------------------------------------
    optimize   tune the detection parameters live -- edit a value, click a
               peak in the trace, see it measured (and see why it would be
               rejected, if it would be) right away
    run        detect every event in the recording in one pass and write
               the events CSV, a sidecar recording the exact parameters
               used, and a whole-trace plot
    check      click any detected event to see every detection window and
               threshold drawn on its own trace, and accept or reject it

Run them as `python -m minianalysis <step> recording.abf` (see cli.py), as
the scripts in the repo root (optimize_params.py, run_detection.py,
check_events.py), or pick a recording and a step from a window with
`python -m minianalysis launch`.

The pieces underneath
---------------------
    core         the detector and the analysis functions -- pure
                 numpy/scipy/pandas, no matplotlib and no Qt, so it imports
                 cleanly into a script or a notebook
    preprocess   zero-phase Bessel low-pass + downsample, shared by all
                 three steps (and usable on its own)
    gui_utils    dialog-field helpers and the GUI exception guard
    style        the plot/GUI color tokens

Everything shares one convention: a gap-free voltage-clamp .abf, one
channel, events described by a peak sample index into whatever trace the
detector actually saw (raw, or filtered/resampled -- which is why the
filter settings have to match between detecting and checking).
"""

__all__ = ["cli", "core"]
__version__ = "1.0.0"

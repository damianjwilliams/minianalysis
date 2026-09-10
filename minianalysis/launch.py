"""
Front door: one small window to pick a recording and a step -- tune, detect
(either way), or check -- plus the filter/resample settings to use, then
launch that step exactly as its own command line would. Handy when you'd
rather not type paths, and the only thing standing between you and an
analysis is remembering which command comes first.

This is a convenience wrapper, not another set of options: each step's own
dialog and flags still apply afterward (`run`, for instance, shows its own
9-detection-parameter dialog next, unless it's launched with --no-gui).

    Tune parameters (optimize)   opens the live three-panel optimizer on
                                 this trace; its own "Run full detection
                                 with these parameters" button is what
                                 actually produces an events CSV.
    Run detection (run)          the batch run -- parameter dialog, then
                                 detection, then the CSV/sidecar/PNG.
    Deconvolution (deconvolve)   the other detector: estimates the event
                                 waveform from the recording and detects on
                                 the deconvolved trace. No thresholds to
                                 tune, and it separates overlapping events.
                                 Writes its own _deconv_* files.
    Check events (check)         the click-to-verify + accept/reject
                                 window, on an events CSV that already
                                 exists. Listed twice -- once per detector,
                                 since each keeps its own events and its own
                                 accept/reject decisions.

"When this finishes, open the event checker on ..." appears for the three
detection steps, naming whichever detector's events it would open. It starts
ticked for `run` and `deconvolve`, where reviewing is the normal next move,
and unticked for `optimize`, which only writes an events CSV if you press its
own "Run full detection" button.

Tick it and the step runs inside a small progress window -- its output as it
appears, an indeterminate progress bar, a Stop button, and an "Open checker
now" button. When the step exits, the checker opens automatically, but ONLY
if the step succeeded and actually left an events CSV behind. Otherwise the
window says which of those two things went wrong and stays open, rather than
launching a checker that would immediately error out. Leave it unticked and
the step is launched fire-and-forget as before, with no extra window in front
of its own GUI.

The filter/resample settings are passed through to whichever step you pick,
and they matter for every one of them: events are detected in, and their
peak_idx values index into, whatever trace the run used, so `check` has to
load that same trace to line up with them. One asymmetry to know about --
`deconvolve` filters by default and takes --no-filter to opt out, whereas
the other steps are opt-in with --filter; the checkbox reads the same either
way and build_argv emits the right flag (which is why picking `deconvolve`
pre-ticks it).

Usage
-----
    python launch.py
    python launch.py recording.abf --channel 0     # pre-filled, still editable
    python -m minianalysis launch                  # same thing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import Optional

from .gui_utils import FILTER_FIELD_SPECS, make_field_widget, read_field_widget
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ

# (command, label shown in the dropdown, explanatory note under it, extra args)
# `extra_args` is appended verbatim to that step's command line -- it exists so
# the two `check` entries can differ only by --source while still sharing one
# command, rather than needing a second dropdown.
STEP_CHOICES = [
    ("optimize", "Tune parameters (optimize) — live, click one peak at a time",
     "Opens the three-panel optimizer: edit any detection parameter on the left, click a peak in "
     "the full trace, and see it measured with those values right away — including why it would "
     "be rejected, if it would be. Its own 'Run full detection with these parameters' button "
     "writes the events CSV once you're happy.", []),
    ("run", "Run detection (run) — detect every event in one pass",
     "Shows the 9-detection-parameter dialog (pre-filled with the filter/resample choice below), "
     "then scans the whole trace and writes <recording>_minianalysis_events.csv, the parameters "
     "sidecar next to it, and a whole-trace PNG with every event marked.", []),
    ("deconvolve", "Detect by deconvolution (deconvolve) — no thresholds to tune",
     "A different detector: it estimates the event waveform from this recording, deconvolves the "
     "trace by it so every event collapses to a sharp impulse, then cleans up duplicate hits on "
     "one event's decay tail. Nothing to tune — its thresholds are in units of the measured "
     "noise. Separates overlapping events the threshold detector merges, and does not fragment "
     "decay tails. Writes <recording>_deconv_events.csv. NOTE: this one filters at 3 kHz / "
     "10 kHz by DEFAULT (like the other detectors in these scripts), so ticking the filter box "
     "below is the normal choice — unticking it passes --no-filter.", []),
    ("check", "Check events (check) — verify what the threshold detector found",
     "Opens the checker on an events CSV that already exists: click any event to see every "
     "detection window and threshold drawn on its own trace, and accept or reject it. Run "
     "'optimize' or 'run' first — this one needs their output.", ["--source", "minianalysis"]),
    ("check", "Check events (check) — verify what the DECONVOLUTION detector found",
     "The same checker, on the deconvolution detector's events instead. Each detector keeps its "
     "own reviewed/progress CSVs, so accepting or rejecting here never touches the decisions you "
     "made on the other one. Run 'deconvolve' first — this one needs its output.",
     ["--source", "deconv"]),
]


@dataclass
class LaunchChoice:
    abf_path: str
    channel: int
    step: str
    filter_enabled: bool
    cutoff_hz: float
    target_rate_hz: float
    filter_order: int
    check_after: bool = False
    extra_args: list[str] = dataclasses_field(default_factory=list)
    """Step-specific flags from the chosen STEP_CHOICES entry (e.g. --source)."""


def _prompt_for_launch(abf_path: str, channel: int) -> Optional[LaunchChoice]:
    from PyQt5 import QtWidgets

    # Must keep a live reference -- see run._prompt_for_settings for why an
    # unassigned QApplication([]) here can be garbage-collected before the
    # QDialog below is constructed.
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Mini Analysis — Choose a Step")
    layout = QtWidgets.QVBoxLayout(dialog)

    file_row = QtWidgets.QHBoxLayout()
    abf_edit = QtWidgets.QLineEdit(abf_path or "")
    browse_btn = QtWidgets.QPushButton("Browse...")

    def on_browse():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            dialog, "Select recording", abf_edit.text() or "", "Axon ABF (*.abf)")
        if path:
            abf_edit.setText(path)

    browse_btn.clicked.connect(on_browse)
    file_row.addWidget(QtWidgets.QLabel("Recording (.abf):"))
    file_row.addWidget(abf_edit)
    file_row.addWidget(browse_btn)
    layout.addLayout(file_row)

    channel_row = QtWidgets.QHBoxLayout()
    channel_spin = QtWidgets.QSpinBox()
    channel_spin.setRange(0, 15)
    channel_spin.setValue(channel)
    channel_row.addWidget(QtWidgets.QLabel("Channel:"))
    channel_row.addWidget(channel_spin)
    channel_row.addStretch(1)
    layout.addLayout(channel_row)

    step_group = QtWidgets.QGroupBox("Step")
    step_layout = QtWidgets.QVBoxLayout(step_group)
    step_combo = QtWidgets.QComboBox()
    step_combo.addItems([label for _, label, _, _ in STEP_CHOICES])
    step_note = QtWidgets.QLabel()
    step_note.setWordWrap(True)
    step_layout.addWidget(step_combo)
    step_layout.addWidget(step_note)
    check_after_w = QtWidgets.QCheckBox("Open the event checker when this finishes")
    step_layout.addWidget(check_after_w)
    layout.addWidget(step_group)

    filter_group = QtWidgets.QGroupBox("Filter / resample (preprocess.py)")
    filter_form = QtWidgets.QFormLayout(filter_group)
    enabled_w = QtWidgets.QCheckBox()
    filter_form.addRow("Apply Bessel low-pass + resample:", enabled_w)
    filter_widgets = {}
    filter_defaults = dict(cutoff_hz=DEFAULT_CUTOFF_HZ, target_rate_hz=DEFAULT_TARGET_RATE_HZ,
                            filter_order=DEFAULT_ORDER)
    for attr, label, kind, kwargs in FILTER_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, filter_defaults[attr])
        w.setEnabled(False)
        enabled_w.toggled.connect(w.setEnabled)
        filter_widgets[attr] = w
        filter_form.addRow(label + ":", w)
    filter_note = QtWidgets.QLabel(
        "Whatever you choose here has to be the same for detection and for checking: event "
        "positions are sample indices into the trace the detector actually saw.")
    filter_note.setWordWrap(True)
    filter_form.addRow(filter_note)
    layout.addWidget(filter_group)

    def on_step_changed(index):
        step, _, note, _extra = STEP_CHOICES[index]
        step_note.setText(note)
        # Nothing to follow `check` with -- it IS the checker.
        can_follow = step != "check"
        check_after_w.setVisible(can_follow)
        if not can_follow:
            check_after_w.setChecked(False)
        else:
            # Name the detector whose events would open, so it is clear which
            # of the two checkers this means, and start it ticked for the
            # steps where reviewing is the normal next move.
            which = "deconvolution" if step == "deconvolve" else "threshold detector's"
            check_after_w.setText(
                f"When this finishes, open the event checker on the {which} events")
            check_after_w.setChecked(step in CHECK_AFTER_BY_DEFAULT)
        # `deconvolve` filters by default on its own command line. Leaving the
        # box unticked here would silently run it differently from how running
        # it by hand runs it, so pre-tick it when that step is chosen. Still
        # freely untickable -- this only moves the starting point.
        if step == "deconvolve" and not enabled_w.isChecked():
            enabled_w.setChecked(True)

    step_combo.currentIndexChanged.connect(on_step_changed)
    on_step_changed(0)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)

    def on_accept():
        if not abf_edit.text().strip():
            QtWidgets.QMessageBox.warning(dialog, "Missing recording", "Choose an .abf file first.")
            return
        dialog.accept()

    buttons.accepted.connect(on_accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(520)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None

    step, _, _, extra_args = STEP_CHOICES[step_combo.currentIndex()]
    filter_values = {attr: read_field_widget(filter_widgets[attr], kind)
                     for attr, _, kind, _ in FILTER_FIELD_SPECS}
    return LaunchChoice(abf_path=abf_edit.text().strip(), channel=channel_spin.value(), step=step,
                         filter_enabled=enabled_w.isChecked(), check_after=check_after_w.isChecked(),
                         extra_args=list(extra_args), **filter_values)


def build_argv(choice: LaunchChoice, command: Optional[str] = None,
                extra_args: Optional[list[str]] = None) -> list[str]:
    """`python -m minianalysis ...` arguments for one step.

    `command` swaps the first word -- used to build the follow-up `check`
    command from the same choice. `extra_args` overrides the choice's own
    step-specific flags, which the follow-up needs (it wants check's
    --source, not the flags of the step that just ran).

    The filter checkbox means the same thing for every step, but the flag it
    produces does NOT: `deconvolve` filters at 3 kHz / 10 kHz by default and
    takes --no-filter to opt out, while optimize/run/check are opt-IN with
    --filter. The polarity is inverted here so the one checkbox stays
    truthful for all of them -- and it matters, because the filter settings
    go into the output filename (core.output_stem), so getting this wrong
    means the follow-up checker looks for a file that was never written.
    """
    cmd = command or choice.step
    argv = [cmd, choice.abf_path, "--channel", str(choice.channel)]
    filter_values = ["--cutoff-hz", str(choice.cutoff_hz),
                     "--target-rate-hz", str(choice.target_rate_hz),
                     "--filter-order", str(choice.filter_order)]
    if cmd == "deconvolve":
        argv += filter_values if choice.filter_enabled else ["--no-filter"]
    elif choice.filter_enabled:
        argv += ["--filter"] + filter_values
    argv += list(choice.extra_args if extra_args is None else extra_args)
    return argv


def followup_check_argv(choice: LaunchChoice) -> list[str]:
    """The `check` command to run after a detection step finishes.

    Which detector's events to open depends on which step just ran -- the
    deconvolution step writes its own CSV under its own suffix.
    """
    source = "deconv" if choice.step == "deconvolve" else "minianalysis"
    return build_argv(choice, command="check", extra_args=["--source", source])


def step_label(step: str) -> str:
    """Short human name for a step, for window titles and status lines."""
    return {"optimize": "parameter tuning", "run": "threshold detection",
            "deconvolve": "deconvolution detection",
            "check": "the event checker"}.get(step, step)


# Steps where reviewing the result afterwards is the normal thing to do, so
# the follow-up checkbox starts ticked. `optimize` is excluded on purpose:
# it is exploratory, and it only writes an events CSV if you press its own
# "Run full detection" button, so defaulting it on would usually end in the
# "no events CSV was written" message.
CHECK_AFTER_BY_DEFAULT = {"run", "deconvolve"}


def expected_events_csv(choice: LaunchChoice) -> str:
    """Where the chosen step will write its events CSV.

    Used to decide whether opening the checker afterwards makes any sense:
    the previous version launched it unconditionally, so a detection run that
    failed -- or an optimizer closed without ever clicking "Run full
    detection" -- was followed by a checker process that immediately errored
    out. Knowing the path up front lets us say why instead.
    """
    from .check import SOURCES
    from .core import output_stem

    source = "deconv" if choice.step == "deconvolve" else "minianalysis"
    stem = output_stem(choice.abf_path, choice.channel, choice.filter_enabled,
                       choice.cutoff_hz, choice.target_rate_hz)
    return stem + SOURCES[source][0]


def followup_decision(returncode: int, stopped: bool, events_csv: str,
                       has_followup: bool, label: str) -> tuple[bool, str]:
    """Should the checker open after a step, and if not, why not?

    Kept out of the Qt callback so it can be tested directly -- this is the
    logic that used to be absent altogether: the old code ran the checker
    unconditionally, so a failed detection was followed by a checker process
    that opened only to error out.

    Returns (open_the_checker, message_to_show).
    """
    if stopped:
        return False, f"Stopped. {label} did not finish."
    if returncode != 0:
        return False, (f"{label} exited with code {returncode}. "
                       f"Not opening the checker.")
    if not has_followup:
        return False, f"{label} finished."
    if not os.path.exists(events_csv):
        return False, (f"{label} finished, but no events CSV was written:"
                       f"\n  {events_csv}"
                       f"\nNothing to check. (If this was the optimizer, its "
                       f"'Run full detection with these parameters' button is "
                       f"what writes that file.)")
    return True, ""


def _run_step_with_progress(cmd: list[str], followup: Optional[list[str]],
                             step_label: str, events_csv: str) -> None:
    """Run `cmd` in a window that shows its output, then open `followup`.

    Replaces a bare `proc.wait()`: between clicking OK and the checker
    appearing there used to be no window at all, just console text, which on
    a ten-minute recording is a long silence with nothing to look at. This
    keeps the step's own stdout visible, offers a Stop button, and -- the
    part that actually matters -- only opens the checker if the step
    succeeded AND left an events CSV behind, reporting the reason when it
    didn't.

    Uses QProcess rather than subprocess so output streams into the widget
    without a reader thread fighting the Qt event loop.
    """
    from PyQt5 import QtCore, QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle(f"Mini Analysis — running {step_label}")
    layout = QtWidgets.QVBoxLayout(dialog)

    heading = QtWidgets.QLabel(f"<b>{step_label}</b> is running.")
    heading.setWordWrap(True)
    layout.addWidget(heading)

    log = QtWidgets.QPlainTextEdit()
    log.setReadOnly(True)
    log.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    log.setStyleSheet("font-family: monospace; font-size: 9pt;")
    log.setMinimumHeight(240)
    layout.addWidget(log)

    bar = QtWidgets.QProgressBar()
    bar.setRange(0, 0)            # indeterminate: no step reports progress
    layout.addWidget(bar)

    buttons = QtWidgets.QDialogButtonBox()
    stop_btn = buttons.addButton("Stop", QtWidgets.QDialogButtonBox.DestructiveRole)
    close_btn = buttons.addButton("Close", QtWidgets.QDialogButtonBox.RejectRole)
    check_now_btn = buttons.addButton("Open checker now",
                                      QtWidgets.QDialogButtonBox.ActionRole)
    check_now_btn.setEnabled(False)
    check_now_btn.setVisible(followup is not None)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(720)

    proc = QtCore.QProcess(dialog)
    proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
    state = {"stopped": False, "launched": False}

    def append(text: str) -> None:
        text = text.rstrip()
        if text:
            log.appendPlainText(text)

    def on_output():
        append(bytes(proc.readAll()).decode("utf-8", errors="replace"))

    def open_checker():
        if state["launched"] or followup is None:
            return
        state["launched"] = True
        append("\nLaunching: " + " ".join(followup))
        subprocess.Popen(followup)
        dialog.accept()

    def on_finished(code, _status):
        bar.setRange(0, 1)
        bar.setValue(1)
        stop_btn.setEnabled(False)
        close_btn.setDefault(True)
        should_open, message = followup_decision(
            code, state["stopped"], events_csv, followup is not None, step_label)
        if message:
            append("\n" + message)
        if should_open:
            check_now_btn.setEnabled(True)
            open_checker()
        elif followup is not None and os.path.exists(events_csv):
            # A stop or a bad exit code, but an events CSV is sitting there
            # anyway (e.g. from an earlier run) -- offer it rather than
            # opening it, since it may not be what this step produced.
            check_now_btn.setEnabled(True)
            append("An events CSV does exist at that path from before; "
                   "'Open checker now' will open that one.")

    def on_stop():
        state["stopped"] = True
        append("\nStopping...")
        proc.kill()

    proc.readyRead.connect(on_output)
    proc.finished.connect(on_finished)
    stop_btn.clicked.connect(on_stop)
    close_btn.clicked.connect(dialog.reject)
    check_now_btn.clicked.connect(open_checker)

    append("Running: " + " ".join(cmd) + "\n")
    proc.start(cmd[0], cmd[1:])
    dialog.exec_()

    # Don't leave the step orphaned if the window was closed while it ran.
    if proc.state() != QtCore.QProcess.NotRunning:
        proc.kill()
        proc.waitForFinished(2000)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("abf", nargs="?", default="", help="Path to the .abf file (pre-fills the dialog)")
    parser.add_argument("--channel", type=int, default=0)
    args = parser.parse_args(argv)

    choice = _prompt_for_launch(args.abf, args.channel)
    if choice is None:
        print("Cancelled — nothing launched.")
        return

    cmd = [sys.executable, "-m", "minianalysis"] + build_argv(choice)
    label = step_label(choice.step)

    if choice.check_after and choice.step != "check":
        # Waiting for the step anyway, so show it happening rather than
        # closing the dialog and going silent on the console.
        followup = [sys.executable, "-m", "minianalysis"] + followup_check_argv(choice)
        print("Launching:", " ".join(cmd))
        _run_step_with_progress(cmd, followup, label, expected_events_csv(choice))
    else:
        # Fire and forget: nothing to wait for, so don't hold a window open
        # in front of the step's own GUI.
        print("Launching:", " ".join(cmd))
        subprocess.Popen(cmd)


if __name__ == "__main__":
    main()

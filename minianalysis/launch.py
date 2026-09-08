"""
Front door: one small window to pick a recording and a step -- tune, run,
or check -- plus the filter/resample settings to use, then launch that step
exactly as its own command line would. Handy when you'd rather not type
paths, and the only thing standing between you and an analysis is
remembering which command comes first.

This is a convenience wrapper, not a fourth set of options: each step's own
dialog and flags still apply afterward (`run`, for instance, shows its own
9-detection-parameter dialog next, unless it's launched with --no-gui).

    Tune parameters (optimize)   opens the live three-panel optimizer on
                                 this trace; its own "Run full detection
                                 with these parameters" button is what
                                 actually produces an events CSV.
    Run detection (run)          the batch run -- parameter dialog, then
                                 detection, then the CSV/sidecar/PNG.
    Check events (check)         the click-to-verify + accept/reject
                                 window, on an events CSV that already
                                 exists.

"Open the event checker when this finishes" appears for the first two
steps: it waits for that process to exit, then opens `check` on the same
recording/channel/filter settings. If no events CSV was produced (e.g. the
optimizer was closed without ever clicking "Run full detection"), the
checker just says so and exits, exactly as it would if you ran it by hand.

The filter/resample settings are passed through to whichever step you pick,
and they matter for all three: events are detected in, and their peak_idx
values index into, whatever trace the run used, so `check` has to load that
same trace to line up with them.

Usage
-----
    python launch.py
    python launch.py recording.abf --channel 0     # pre-filled, still editable
    python -m minianalysis launch                  # same thing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from .gui_utils import FILTER_FIELD_SPECS, make_field_widget, read_field_widget
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ

# (command, label shown in the dropdown, explanatory note under it)
STEP_CHOICES = [
    ("optimize", "Tune parameters (optimize) — live, click one peak at a time",
     "Opens the three-panel optimizer: edit any detection parameter on the left, click a peak in "
     "the full trace, and see it measured with those values right away — including why it would "
     "be rejected, if it would be. Its own 'Run full detection with these parameters' button "
     "writes the events CSV once you're happy."),
    ("run", "Run detection (run) — detect every event in one pass",
     "Shows the 9-detection-parameter dialog (pre-filled with the filter/resample choice below), "
     "then scans the whole trace and writes <recording>_minianalysis_events.csv, the parameters "
     "sidecar next to it, and a whole-trace PNG with every event marked."),
    ("check", "Check events (check) — verify and accept/reject what was detected",
     "Opens the checker on an events CSV that already exists: click any event to see every "
     "detection window and threshold drawn on its own trace, and accept or reject it. Run one of "
     "the two steps above first — this one needs their output."),
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
    step_combo.addItems([label for _, label, _ in STEP_CHOICES])
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
        step, _, note = STEP_CHOICES[index]
        step_note.setText(note)
        # Nothing to follow `check` with -- it IS the checker.
        can_follow = step != "check"
        check_after_w.setVisible(can_follow)
        if not can_follow:
            check_after_w.setChecked(False)

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

    step = STEP_CHOICES[step_combo.currentIndex()][0]
    filter_values = {attr: read_field_widget(filter_widgets[attr], kind)
                     for attr, _, kind, _ in FILTER_FIELD_SPECS}
    return LaunchChoice(abf_path=abf_edit.text().strip(), channel=channel_spin.value(), step=step,
                         filter_enabled=enabled_w.isChecked(), check_after=check_after_w.isChecked(),
                         **filter_values)


def build_argv(choice: LaunchChoice, command: Optional[str] = None) -> list[str]:
    """`python -m minianalysis ...` arguments for one step. All three take
    the same abf/channel/filter flags (that's the point of keeping their
    CLIs aligned), so `command` only swaps the first word -- used to build
    the follow-up `check` command from the same choice."""
    argv = [command or choice.step, choice.abf_path, "--channel", str(choice.channel)]
    if choice.filter_enabled:
        argv += ["--filter", "--cutoff-hz", str(choice.cutoff_hz),
                 "--target-rate-hz", str(choice.target_rate_hz),
                 "--filter-order", str(choice.filter_order)]
    return argv


def _wait_then_launch(proc: subprocess.Popen, cmd: list[str]) -> None:
    """Block until `proc` exits, then launch `cmd` -- the checker after a
    tune/run. The checker reports and exits cleanly on its own if the run
    never produced an events CSV, so there's nothing to check for here.
    Deliberately blocking, not a background thread: a daemon thread would
    be killed the moment main() returns, and nothing else keeps this
    process alive while the first command runs."""
    proc.wait()
    print("Launching:", " ".join(cmd))
    subprocess.Popen(cmd)


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
    print("Launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd)

    if choice.check_after and choice.step != "check":
        followup = [sys.executable, "-m", "minianalysis"] + build_argv(choice, command="check")
        print(f"Waiting for {choice.step!r} to finish before opening the event checker...")
        _wait_then_launch(proc, followup)


if __name__ == "__main__":
    main()

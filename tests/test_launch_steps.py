"""Tests for the launcher's step list, argv building, and auto-check follow-up.

Two things here are easy to get wrong and impossible to notice by reading:

  * `deconvolve` filters by DEFAULT and takes --no-filter to opt out, while
    every other step is opt-IN with --filter. The filter settings go into the
    output filename, so an inverted flag means the follow-up checker looks for
    a file that was never written.
  * The checker must open after a detection run ONLY if that run succeeded and
    actually left an events CSV behind.

`test_the_launch_dialog_actually_builds` opens the real dialog, because the
argv-level tests below all passed while a stale tuple unpack in the dialog's
own combo-box setup was broken.
"""

from __future__ import annotations

import os

import pytest

# Qt must be told to run headless BEFORE any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

check = pytest.importorskip("minianalysis.check")
pytest.importorskip("PyQt5")

# ------------------------------------------------------ the launch front door

def _choice(step, filter_enabled=True, extra_args=()):
    from minianalysis.launch import LaunchChoice
    return LaunchChoice(abf_path="rec.abf", channel=0, step=step,
                        filter_enabled=filter_enabled, cutoff_hz=3000.0,
                        target_rate_hz=10000.0, filter_order=8,
                        check_after=True, extra_args=list(extra_args))


def test_launch_offers_the_deconvolution_step_and_its_own_checker():
    from minianalysis.launch import STEP_CHOICES
    commands = [c for c, _, _, _ in STEP_CHOICES]
    assert "deconvolve" in commands
    # two check entries, one per detector, distinguished only by --source
    sources = [tuple(extra) for cmd, _, _, extra in STEP_CHOICES if cmd == "check"]
    assert ("--source", "minianalysis") in sources
    assert ("--source", "deconv") in sources
    # every entry is a 4-tuple, or the dialog unpacking breaks
    assert all(len(entry) == 4 for entry in STEP_CHOICES)


def test_deconvolve_gets_no_filter_flag_when_filtering_is_on():
    """deconvolve filters by DEFAULT, so an explicit --filter would be an
    unknown flag; the values alone are correct."""
    from minianalysis.launch import build_argv
    argv = build_argv(_choice("deconvolve", filter_enabled=True))
    assert "--filter" not in argv
    assert "--no-filter" not in argv
    assert "--cutoff-hz" in argv and "3000.0" in argv


def test_deconvolve_gets_no_filter_when_filtering_is_off():
    from minianalysis.launch import build_argv
    argv = build_argv(_choice("deconvolve", filter_enabled=False))
    assert "--no-filter" in argv
    assert "--filter" not in argv


@pytest.mark.parametrize("step", ["optimize", "run", "check"])
def test_the_other_steps_keep_the_opt_in_filter_flag(step):
    from minianalysis.launch import build_argv
    assert "--filter" in build_argv(_choice(step, filter_enabled=True))
    off = build_argv(_choice(step, filter_enabled=False))
    assert "--filter" not in off and "--no-filter" not in off


def test_followup_checker_targets_the_detector_that_just_ran():
    from minianalysis.launch import followup_check_argv
    deconv = followup_check_argv(_choice("deconvolve"))
    assert deconv[0] == "check"
    assert deconv[deconv.index("--source") + 1] == "deconv"
    for step in ("run", "optimize"):
        mini = followup_check_argv(_choice(step))
        assert mini[mini.index("--source") + 1] == "minianalysis"


def test_followup_checker_is_opt_in_filtered_even_after_deconvolve():
    """The stem encodes the filter settings, so the checker must rebuild the
    same trace -- with ITS flag spelling, not deconvolve's."""
    from minianalysis.launch import followup_check_argv
    fu = followup_check_argv(_choice("deconvolve", filter_enabled=True))
    assert "--filter" in fu and "--no-filter" not in fu
    fu_raw = followup_check_argv(_choice("deconvolve", filter_enabled=False))
    assert "--filter" not in fu_raw and "--no-filter" not in fu_raw


def test_followup_stem_matches_where_deconvolve_actually_writes():
    """The regression that would silently break: check looking for a file
    under a different stem than the detector wrote."""
    from minianalysis.core import output_stem
    from minianalysis.launch import followup_check_argv
    for filter_enabled in (True, False):
        c = _choice("deconvolve", filter_enabled=filter_enabled)
        fu = followup_check_argv(c)
        checker_stem = output_stem("rec.abf", 0, "--filter" in fu, 3000.0, 10000.0)
        writer_stem = output_stem("rec.abf", 0, filter_enabled, 3000.0, 10000.0)
        assert checker_stem == writer_stem, filter_enabled


def test_step_extra_args_are_not_leaked_into_the_followup():
    """A check entry's --source must not ride along on a detection step."""
    from minianalysis.launch import build_argv
    argv = build_argv(_choice("deconvolve", extra_args=["--source", "deconv"]),
                      command="check", extra_args=["--source", "minianalysis"])
    assert argv.count("--source") == 1
    assert argv[argv.index("--source") + 1] == "minianalysis"


def test_the_launch_dialog_actually_builds(monkeypatch):
    """Constructs the real dialog, headless.

    Every other launch test here works on STEP_CHOICES and build_argv without
    ever opening the window -- which is exactly how a stale 3-tuple unpack in
    `step_combo.addItems` survived a green suite. This one opens it.
    """
    from PyQt5 import QtCore, QtWidgets
    import minianalysis.launch as L

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seen = {}

    def inspect_then_close():
        for w in app.topLevelWidgets():
            if isinstance(w, QtWidgets.QDialog) and w.isVisible():
                combo = w.findChild(QtWidgets.QComboBox)
                seen["items"] = [combo.itemText(i) for i in range(combo.count())]
                # selecting deconvolve must pre-tick the filter box, because
                # that step filters by default on its own command line
                combo.setCurrentIndex(
                    next(i for i, (c, _, _, _) in enumerate(L.STEP_CHOICES)
                         if c == "deconvolve"))
                app.processEvents()
                seen["filter_ticked"] = any(
                    b.isChecked() for b in w.findChildren(QtWidgets.QCheckBox))
                w.reject()

    QtCore.QTimer.singleShot(0, inspect_then_close)
    result = L._prompt_for_launch("rec.abf", 0)

    assert result is None, "rejected dialog must return None"
    assert len(seen["items"]) == len(L.STEP_CHOICES)
    assert any("deconvolution" in it.lower() for it in seen["items"])
    assert seen["filter_ticked"], "deconvolve should pre-tick the filter box"


# ------------------------------------- auto-check after detection completes

def test_followup_is_ticked_by_default_only_for_the_detection_steps():
    """Reviewing is the normal next move after a detection run, but not after
    the optimizer, which usually writes no events CSV at all."""
    from minianalysis.launch import CHECK_AFTER_BY_DEFAULT
    assert CHECK_AFTER_BY_DEFAULT == {"run", "deconvolve"}
    assert "optimize" not in CHECK_AFTER_BY_DEFAULT
    assert "check" not in CHECK_AFTER_BY_DEFAULT


def test_dialog_ticks_the_followup_per_step_and_names_the_detector():
    from PyQt5 import QtCore, QtWidgets
    import minianalysis.launch as L

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    seen = {}

    def probe():
        for w in app.topLevelWidgets():
            if isinstance(w, QtWidgets.QDialog) and w.isVisible():
                combo = w.findChild(QtWidgets.QComboBox)
                for i, (cmd, _, _, _) in enumerate(L.STEP_CHOICES):
                    combo.setCurrentIndex(i)
                    app.processEvents()
                    boxes = [b for b in w.findChildren(QtWidgets.QCheckBox)
                             if "checker" in b.text()]
                    seen.setdefault(cmd, []).append(
                        (bool(boxes) and boxes[0].isVisible(),
                         boxes[0].isChecked() if boxes else False,
                         boxes[0].text() if boxes else ""))
                w.reject()

    QtCore.QTimer.singleShot(0, probe)
    L._prompt_for_launch("rec.abf", 0)

    assert seen["run"][0][:2] == (True, True)
    assert seen["deconvolve"][0][:2] == (True, True)
    assert seen["optimize"][0][:2] == (True, False)
    assert seen["check"][0][0] is False, "nothing to follow the checker with"
    # the label says WHICH detector's events will open
    assert "deconvolution" in seen["deconvolve"][0][2]
    assert "threshold" in seen["run"][0][2]


def test_expected_events_csv_matches_the_step_and_filter_settings():
    from minianalysis.core import output_stem
    from minianalysis.launch import expected_events_csv

    for step, source in (("deconvolve", "deconv"), ("run", "minianalysis"),
                         ("optimize", "minianalysis")):
        for filt in (True, False):
            c = _choice(step, filter_enabled=filt)
            stem = output_stem("rec.abf", 0, filt, 3000.0, 10000.0)
            assert expected_events_csv(c) == stem + check.SOURCES[source][0]


def test_checker_opens_only_on_success_with_a_real_events_csv(tmp_path):
    """The bug this replaced: the checker used to be launched unconditionally,
    so a failed detection was followed by a process that only errored out."""
    from minianalysis.launch import followup_decision

    present = tmp_path / "events.csv"
    present.write_text("peak_idx\n1\n")
    absent = str(tmp_path / "nothing.csv")

    ok, msg = followup_decision(0, False, str(present), True, "detection")
    assert ok and msg == ""

    for code, stopped, csv in ((1, False, str(present)),      # bad exit code
                                (0, True, str(present)),      # user stopped it
                                (0, False, absent)):          # nothing written
        ok, msg = followup_decision(code, stopped, csv, True, "detection")
        assert not ok
        assert msg, "a refusal must say why"


def test_followup_decision_explains_each_refusal_distinctly():
    from minianalysis.launch import followup_decision
    msgs = {
        "bad_exit": followup_decision(3, False, __file__, True, "detection")[1],
        "stopped": followup_decision(0, True, __file__, True, "detection")[1],
        "no_csv": followup_decision(0, False, "/definitely/absent.csv", True, "detection")[1],
        "not_asked": followup_decision(0, False, __file__, False, "detection")[1],
    }
    assert "code 3" in msgs["bad_exit"]
    assert "Stopped" in msgs["stopped"]
    assert "no events CSV" in msgs["no_csv"]
    assert "Run full detection" in msgs["no_csv"], "should name the optimizer's button"
    assert len(set(msgs.values())) == 4, "each reason should read differently"


def test_step_label_covers_every_step():
    from minianalysis.launch import STEP_CHOICES, step_label
    for cmd, _, _, _ in STEP_CHOICES:
        label = step_label(cmd)
        assert label and label != cmd, f"{cmd} has no human label"

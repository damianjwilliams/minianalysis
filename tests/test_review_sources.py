"""Tests for reviewing a second detector's events in the same checker window.

`check_events.py --source deconv` opens the classical event checker on the
deconvolution detector's output. The window is detector-agnostic (it rebuilds
each event's windows from the CSV's own baseline+amplitude), but two things
must NOT leak across:

  * the QC files -- each source keeps its own reviewed/progress CSVs, so
    reviewing one never overwrites the other's decisions;
  * the accept/reject criteria -- the classical detector's amplitude (a),
    area (b) and local-max (c) parameters must not be shown, or claimed as
    PASSED, for a detector that never applied them.
"""

from __future__ import annotations

import json
import os

import pytest

# Qt must be told to run headless BEFORE any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from minianalysis.core import DetectionParams, events_frame
from minianalysis.deconvolve import DeconvParams, _synth, detect_events_deconv

check = pytest.importorskip("minianalysis.check")
pytest.importorskip("PyQt5")

DT = 1e-4


# ------------------------------------------------------------- the registry

def test_both_sources_are_registered_with_four_distinct_suffixes():
    assert set(check.SOURCES) == {"minianalysis", "deconv"}
    for name, suffixes in check.SOURCES.items():
        assert len(suffixes) == 4, name
        assert len(set(suffixes)) == 4, f"{name} has duplicate suffixes"
        assert name in check.SOURCE_LABELS


def test_no_suffix_is_shared_between_sources():
    """The whole point: reviewing one source cannot clobber the other."""
    mini = set(check.SOURCES["minianalysis"])
    deconv = set(check.SOURCES["deconv"])
    assert not (mini & deconv)


# --------------------------------------------------- criteria are not faked

def test_params_text_hides_classical_only_criteria_for_another_source():
    p = DetectionParams()
    full = check.params_text(p, show_thresholds=True)
    trimmed = check.params_text(p, show_thresholds=False)
    # the accept/reject criteria go
    assert "Amplitude threshold (a)" in full
    assert "Amplitude threshold (a)" not in trimmed
    assert "Area threshold (b)" not in trimmed
    assert "Period to search local maximum (c)" not in trimmed
    # the shared MEASUREMENT windows stay -- both detectors used these
    for kept in ("Time before peak for baseline (d)",
                 "Period to average baseline (e)",
                 "Period to search decay time (f)",
                 "Fraction to find decay (g)"):
        assert kept in trimmed, kept


def test_params_text_appends_a_sources_own_criteria():
    got = check.params_text(DetectionParams(), show_thresholds=False,
                            extra="Deconvolution criteria\nThreshold: 6.0 SD")
    assert got.endswith("Threshold: 6.0 SD")


def test_legend_key_drops_the_annotations_that_are_no_longer_drawn():
    full = [lbl for _, lbl in check.build_legend_handles(True)]
    trimmed = [lbl for _, lbl in check.build_legend_handles(False)]
    assert "Amplitude threshold (a)" in full
    assert "Amplitude threshold (a)" not in trimmed
    assert "Local-max search span (c)" not in trimmed
    assert not any("PASS" in lbl or "FAIL" in lbl for lbl in trimmed)
    # ...but the shared annotations remain, so the key still explains the plot
    for kept in ("Baseline level used", "Detected peak", "Onset (rise crossing)",
                 "Decay point", "Decay search window (f)"):
        assert kept in trimmed, kept


# ------------------------------------------------- the sidecar is tolerated

def test_load_params_ignores_a_foreign_sidecars_extra_keys(tmp_path, capsys):
    """The deconv sidecar holds DeconvParams plus run diagnostics.
    DetectionParams(**raw) would raise TypeError on every one of them."""
    stem = str(tmp_path / "rec")
    sidecar = dict(vars(DeconvParams()) if not hasattr(DeconvParams(), "__dataclass_fields__")
                   else {f: getattr(DeconvParams(), f)
                         for f in DeconvParams.__dataclass_fields__})
    sidecar.update(noise_sd=2.61, detection_limit_4sd=10.43, kernel_ms=18.4,
                   kernel_source="estimated", n_isolated_for_kernel=297)
    path = stem + check.SOURCES["deconv"][1]
    with open(path, "w") as fh:
        json.dump(sidecar, fh)

    params = check._load_params(None, stem, {}, check.SOURCES["deconv"][1])
    assert isinstance(params, DetectionParams)
    # shared measurement fields carried across from the deconv sidecar
    assert params.decay_search_ms == DeconvParams().decay_search_ms
    assert params.baseline_before_ms == DeconvParams().baseline_before_ms
    # classical-only fields fell back to their defaults, NOT to anything real
    assert params.amplitude_threshold == DetectionParams().amplitude_threshold
    assert "ignored" in capsys.readouterr().out


def test_source_criteria_text_reports_real_thresholds(tmp_path):
    stem = str(tmp_path / "rec")
    path = stem + check.SOURCES["deconv"][1]
    with open(path, "w") as fh:
        json.dump(dict(thresh_sd=6.0, min_amp_sd=2.5, min_amp_pa=0.0,
                       noise_sd=2.61, detection_limit_4sd=10.43,
                       kernel_ms=18.4, kernel_source="estimated",
                       n_isolated_for_kernel=297), fh)
    text = check._source_criteria_text("deconv", path)
    assert "6.0 SD of deconvolved trace" in text
    assert "2.5 SD of raw trace" in text
    assert "2.61" in text and "10.43" in text
    assert "18.4 ms" in text and "297" in text


def test_source_criteria_text_is_empty_for_the_classical_source(tmp_path):
    assert check._source_criteria_text("minianalysis", str(tmp_path / "nope")) == ""


def test_source_criteria_text_survives_a_missing_or_corrupt_sidecar(tmp_path):
    assert check._source_criteria_text("deconv", str(tmp_path / "absent.json")) == ""
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert check._source_criteria_text("deconv", str(bad)) == ""


# ------------------------------------------------------- end to end, headless

def test_reviewer_opens_on_deconv_output_and_saves_to_its_own_files(tmp_path):
    from PyQt5 import QtWidgets

    t, v, true_s = _synth(rate_hz=5.0, duration_s=20.0, noise_sd_pa=2.0, seed=4)
    events = detect_events_deconv(t, v, DT, DeconvParams())
    assert len(events) > 20, "need a few events to review"

    stem = str(tmp_path / "rec")
    ev_s, pa_s, rv_s, pg_s = check.SOURCES["deconv"]
    df = events_frame(events)
    df.to_csv(stem + ev_s, index=False)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    params = check._load_params(None, stem, {}, pa_s)
    w = check.EventInspector(
        t, v, DT, df, params, "pA", title="test", pre_ms=12.0, post_ms=53.0,
        reviewed_path=stem + rv_s, progress_path=stem + pg_s,
        show_thresholds=False,
        extra_param_text=check._source_criteria_text("deconv", stem + pa_s))
    try:
        w.accept_current()
        w.reject_current()
        assert len(w.decisions) == 2
        assert os.path.exists(stem + pg_s)
        assert os.path.exists(stem + rv_s)
        # and nothing was written under the classical detector's names
        for suffix in check.SOURCES["minianalysis"]:
            assert not os.path.exists(stem + suffix), suffix
    finally:
        w.close()


def test_deconv_detail_title_makes_no_pass_fail_claim():
    """A detector with no amplitude/area threshold must not report PASS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, v, _ = _synth(rate_hz=5.0, duration_s=10.0, noise_sd_pa=2.0, seed=6)
    events = detect_events_deconv(t, v, DT, DeconvParams())
    assert events
    row = events_frame(events).iloc[3]

    fig, ax = plt.subplots()
    try:
        check.plot_event_detail(ax, t, v, DT, row, DetectionParams(), "pA",
                                12.0, 53.0, show_thresholds=False)
        title = ax.get_title()
        assert "PASS" not in title and "FAIL" not in title
        assert "thr" not in title
        assert "rise=" in title and "decay=" in title
        # and the classical rendering still does say it
        ax.clear()
        check.plot_event_detail(ax, t, v, DT, row, DetectionParams(), "pA",
                                12.0, 53.0, show_thresholds=True)
        assert "PASS" in ax.get_title() or "FAIL" in ax.get_title()
    finally:
        plt.close(fig)

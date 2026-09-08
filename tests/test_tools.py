"""
Tests for the two GUI tools' non-GUI math -- the parts that duplicate
detect_events' per-candidate steps and so could silently drift away from
it: optimize.evaluate_candidate (single-click testing) and
check._event_geometry (the windows drawn on an already-detected event).

Nothing here opens a window; only the module-level PyQt5/pyqtgraph imports
are needed, and the whole file skips if those aren't installed.
"""

from __future__ import annotations

import pytest

from minianalysis.core import DetectionParams, detect_events

pytest.importorskip("PyQt5", reason="the optimize/check tools need PyQt5")
pytest.importorskip("pyqtgraph", reason="the optimizer needs pyqtgraph")

from minianalysis.check import _event_geometry  # noqa: E402  (after the importorskip guards)
from minianalysis.optimize import evaluate_candidate  # noqa: E402

from test_core import DT, synthetic_trace  # noqa: E402


def test_evaluate_candidate_accepts_what_detect_events_accepts():
    """Clicking a real event in the optimizer must reach the same verdict
    the full scan does -- same baseline, same amplitude, same area."""
    t, v = synthetic_trace()
    params = DetectionParams()
    events = detect_events(t, v, DT, params)
    assert events, "the fixture should produce events to compare against"

    for event in events:
        result = evaluate_candidate(v, DT, event.peak_idx, params)
        assert result.accepted, result.reasons
        assert result.peak_idx == event.peak_idx
        assert result.baseline == pytest.approx(event.baseline)
        assert result.amplitude == pytest.approx(event.amplitude)
        assert result.area == pytest.approx(event.area)
        assert result.rise_time_ms == pytest.approx(event.rise_time_ms)
        assert result.decay_time_ms == pytest.approx(event.decay_time_ms)


def test_evaluate_candidate_snaps_a_sloppy_click_to_the_nearest_peak():
    t, v = synthetic_trace()
    params = DetectionParams()
    peak_idx = detect_events(t, v, DT, params)[0].peak_idx
    result = evaluate_candidate(v, DT, peak_idx + 5, params)  # clicked 0.5 ms late
    assert result.peak_idx == peak_idx


def test_evaluate_candidate_explains_every_rejection_reason():
    """Unlike detect_events, which stops at the first failed check, a click
    collects them all -- that's the point of the tool."""
    t, v = synthetic_trace()
    peak_idx = detect_events(t, v, DT, DetectionParams())[0].peak_idx
    params = DetectionParams(amplitude_threshold=100.0, area_threshold=10_000.0)
    result = evaluate_candidate(v, DT, peak_idx, params)
    assert not result.accepted
    assert len(result.reasons) == 2
    assert any("amplitude" in r for r in result.reasons)
    assert any("area" in r for r in result.reasons)


def test_event_geometry_reproduces_the_onset_and_decay_the_detector_measured():
    """check.py redraws each event's windows from its CSV row; those
    onset/decay positions have to land where detect_events put them."""
    t, v = synthetic_trace()
    params = DetectionParams()
    for event in detect_events(t, v, DT, params):
        geo = _event_geometry(v, DT, event.peak_idx, event.baseline, event.amplitude, params)
        rise_ms = (event.peak_idx - geo["onset_idx"]) * DT * 1e3
        decay_ms = (geo["decay_idx"] - event.peak_idx) * DT * 1e3
        assert rise_ms == pytest.approx(event.rise_time_ms)
        assert decay_ms == pytest.approx(event.decay_time_ms)
        assert geo["b0"] < geo["b1"] <= event.peak_idx, "baseline window sits before the peak"


def test_event_geometry_windows_match_the_parameters_asked_for():
    t, v = synthetic_trace()
    params = DetectionParams(baseline_before_ms=2.0, baseline_avg_ms=3.0, decay_search_ms=20.0)
    peak_idx = detect_events(t, v, DT, params)[0].peak_idx
    geo = _event_geometry(v, DT, peak_idx, 0.0, -20.0, params)
    assert (peak_idx - geo["b1"]) * DT * 1e3 == pytest.approx(2.0, abs=0.1)     # (d)
    assert (geo["b1"] - geo["b0"]) * DT * 1e3 == pytest.approx(3.0, abs=0.1)    # (e)
    assert (geo["decay_search_end"] - peak_idx) * DT * 1e3 == pytest.approx(20.0, abs=0.1)  # (f)


def test_launch_builds_the_same_flags_for_every_step():
    from minianalysis.launch import LaunchChoice, build_argv
    choice = LaunchChoice(abf_path="rec.abf", channel=1, step="run", filter_enabled=True,
                           cutoff_hz=3000.0, target_rate_hz=10000.0, filter_order=8)
    run_argv = build_argv(choice)
    check_argv = build_argv(choice, command="check")
    assert run_argv[0] == "run" and check_argv[0] == "check"
    assert run_argv[1:] == check_argv[1:], "the follow-up checker must see the same trace settings"
    assert "--filter" in run_argv and "3000.0" in run_argv


def test_cli_rejects_an_unknown_command():
    from minianalysis.cli import main
    with pytest.raises(SystemExit) as excinfo:
        main(["nonsense"])
    assert excinfo.value.code == 2


def test_output_stem_distinguishes_runs_that_must_not_share_files():
    """run.py names its output files off a stem and check.py rebuilds the
    same one; two runs that describe different traces must not collide."""
    from minianalysis.core import output_stem
    assert output_stem("rec.abf") == "rec"
    assert output_stem("rec.abf", channel=0) == "rec", "channel 0 keeps the plain name"
    assert output_stem("rec.abf", channel=1) == "rec_ch1"
    assert output_stem("rec.abf", filter_enabled=True, cutoff_hz=3000.0,
                        target_rate_hz=10000.0) == "rec_filt3000Hz10000Hz"
    assert output_stem("rec.abf", channel=1, filter_enabled=True, cutoff_hz=3000.0,
                        target_rate_hz=10000.0) == "rec_ch1_filt3000Hz10000Hz"

    # The bug this guards: analysing a second channel used to overwrite the
    # first channel's events CSV, silently, with no warning.
    assert output_stem("rec.abf", channel=0) != output_stem("rec.abf", channel=1)
    # ...and a filtered run describes a different trace than a raw one, so
    # its event indices mean something different.
    assert output_stem("rec.abf") != output_stem("rec.abf", filter_enabled=True,
                                                  cutoff_hz=3000.0, target_rate_hz=10000.0)


def test_events_csv_round_trips_exactly(tmp_path):
    """check.py rebuilds each event's detection windows from the baseline
    and amplitude in the events CSV, so those numbers have to come back out
    of the file bit for bit. pandas' default C float parser is NOT correctly
    rounded and loses an ULP on a good fraction of cells (357 of 2292 on a
    real 573-event recording), which is enough to move a drawn onset/decay
    marker to a different sample than the detector actually measured."""
    import pandas as pd
    from minianalysis.core import events_frame

    t, v = synthetic_trace(noise_sd=1.0)          # noise -> ugly, full-precision floats
    events = detect_events(t, v, DT, DetectionParams())
    assert events
    df = events_frame(events)

    csv = tmp_path / "rec_minianalysis_events.csv"
    df.to_csv(csv, index=False)
    back = pd.read_csv(csv, float_precision="round_trip")

    for col in ["baseline", "amplitude", "area", "peak_time_s"]:
        assert (back[col].to_numpy() == df[col].to_numpy()).all(), f"{col} did not survive the CSV"


def test_geometry_matches_the_detector_after_a_csv_round_trip(tmp_path):
    """The end-to-end version of the above: detect, write the CSV, read it
    back the way check.py does, and confirm every redrawn window still lands
    on the sample the detector measured."""
    import pandas as pd
    from minianalysis.core import events_frame

    t, v = synthetic_trace(noise_sd=1.0)
    params = DetectionParams()
    events = detect_events(t, v, DT, params)
    csv = tmp_path / "rec_minianalysis_events.csv"
    events_frame(events).to_csv(csv, index=False)
    back = pd.read_csv(csv, float_precision="round_trip")

    for event, (_, row) in zip(events, back.iterrows()):
        geo = _event_geometry(v, DT, int(row["peak_idx"]), row["baseline"], row["amplitude"], params)
        assert (int(row["peak_idx"]) - geo["onset_idx"]) * DT * 1e3 == pytest.approx(event.rise_time_ms)
        assert (geo["decay_idx"] - int(row["peak_idx"])) * DT * 1e3 == pytest.approx(event.decay_time_ms)

"""
Tests for the detector and the analysis functions, on a synthetic trace
whose events are known exactly -- so a failure here means the detection
math changed, not that a recording looked different today.

Run with:   pytest        (from the repo root)
"""

from __future__ import annotations

import numpy as np
import pytest

from minianalysis.core import (
    DetectionParams, column_statistics, cumulative_histogram, detect_events, events_frame,
    extract_event_traces, fit_exponential_decay, frequency_histogram, group_by_criteria,
    group_random, scale_traces,
)

FS = 10_000.0          # Hz
DT = 1.0 / FS
EVENT_TIMES_S = [0.10, 0.40, 0.70, 1.00, 1.30]
AMPLITUDE_PA = -20.0   # inward, like a real sEPSC in voltage clamp
TAU_MS = 3.0
RISE_MS = 0.4


def synthetic_trace(amplitude_pa: float = AMPLITUDE_PA, noise_sd: float = 0.0,
                     event_times_s=EVENT_TIMES_S, duration_s: float = 1.6):
    """A flat baseline with alpha-ish events planted at known times: a
    linear rise over RISE_MS into the peak, then a single-exponential decay
    with time constant TAU_MS."""
    n = int(duration_s * FS)
    t = np.arange(n) * DT
    v = np.zeros(n)
    rise_n = max(1, int(RISE_MS / 1000.0 * FS))
    decay_n = int(10 * TAU_MS / 1000.0 * FS)  # 10 tau is fully decayed for our purposes
    for t0 in event_times_s:
        peak_i = int(round(t0 * FS))
        rise = np.linspace(0.0, 1.0, rise_n, endpoint=False)
        v[peak_i - rise_n:peak_i] += amplitude_pa * rise
        decay = np.exp(-np.arange(decay_n) * DT * 1000.0 / TAU_MS)
        v[peak_i:peak_i + decay_n] += amplitude_pa * decay
    if noise_sd:
        v = v + np.random.default_rng(0).normal(0.0, noise_sd, size=n)
    return t, v


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------

def test_detects_every_planted_event_at_the_right_time():
    t, v = synthetic_trace()
    events = detect_events(t, v, DT, DetectionParams())
    assert len(events) == len(EVENT_TIMES_S)
    found = np.array([e.peak_time_s for e in events])
    assert np.allclose(found, EVENT_TIMES_S, atol=0.001)  # within 1 ms


def test_measured_amplitude_matches_the_planted_one():
    t, v = synthetic_trace()
    events = detect_events(t, v, DT, DetectionParams(n_avg_peak=1))  # read the peak sample itself
    amps = np.array([e.amplitude for e in events])
    assert np.allclose(amps, AMPLITUDE_PA, rtol=0.01)
    assert (amps < 0).all(), "inward events must keep their negative sign"


def test_averaging_the_peak_over_more_points_clips_the_amplitude():
    """n_avg_peak > 1 averages the peak with its neighbours, which on a
    sharply-peaked event reads slightly SMALLER than the peak sample --
    the intended noise/bias trade-off, worth pinning down so it can't
    change silently."""
    t, v = synthetic_trace()
    one = detect_events(t, v, DT, DetectionParams(n_avg_peak=1))[0].amplitude
    three = detect_events(t, v, DT, DetectionParams(n_avg_peak=3))[0].amplitude
    assert abs(three) < abs(one)
    assert abs(three) == pytest.approx(abs(one), rel=0.15)


def test_measured_half_decay_matches_the_planted_tau():
    t, v = synthetic_trace()
    # decay_fraction 0.5 -> half-decay; n_avg_peak=1 so the level it is
    # measured against is half of the true peak, not half of an average.
    events = detect_events(t, v, DT, DetectionParams(n_avg_peak=1))
    half_decay_ms = np.array([e.decay_time_ms for e in events])
    expected = TAU_MS * np.log(2)
    assert np.allclose(half_decay_ms, expected, atol=0.15)


def test_amplitude_threshold_rejects_everything_above_it():
    t, v = synthetic_trace()
    params = DetectionParams(amplitude_threshold=abs(AMPLITUDE_PA) * 2)
    assert detect_events(t, v, DT, params) == []


def test_area_threshold_rejects_everything_above_it():
    t, v = synthetic_trace()
    # Each event's area is ~amplitude * tau = 60 pA*ms; ask for far more.
    params = DetectionParams(area_threshold=10_000.0)
    assert detect_events(t, v, DT, params) == []


def test_wrong_direction_finds_nothing():
    t, v = synthetic_trace()
    assert detect_events(t, v, DT, DetectionParams(direction="positive")) == []


def test_positive_direction_finds_outward_events():
    t, v = synthetic_trace(amplitude_pa=+20.0)
    events = detect_events(t, v, DT, DetectionParams(direction="positive"))
    assert len(events) == len(EVENT_TIMES_S)
    assert all(e.amplitude > 0 for e in events)


def test_decay_search_window_is_a_hard_limit():
    """An event whose decay never reaches decay_fraction within
    decay_search_ms is rejected outright, not measured with a truncated
    decay -- see detect_events step 5."""
    t, v = synthetic_trace()
    params = DetectionParams(decay_search_ms=0.5)  # half-decay takes ~2.1 ms
    assert detect_events(t, v, DT, params) == []


def test_onset_search_window_is_a_hard_limit():
    """The rise-side counterpart: no onset crossing within
    onset_search_ms means the candidate is rejected, same as an unreached
    decay -- see detect_events step 4."""
    t, v = synthetic_trace()
    # The trace sits AT baseline until RISE_MS before the peak, so an onset
    # search that starts inside the rise itself never sees the crossing.
    params = DetectionParams(onset_search_ms=0.2)
    assert detect_events(t, v, DT, params) == []


def test_survives_a_noisy_trace():
    t, v = synthetic_trace(noise_sd=1.0)
    events = detect_events(t, v, DT, DetectionParams(amplitude_threshold=8.0, area_threshold=20.0))
    found = np.array([e.peak_time_s for e in events])
    for planted in EVENT_TIMES_S:
        assert np.any(np.abs(found - planted) < 0.001), f"missed the event at {planted}s"


def test_empty_trace_detects_nothing():
    t, v = synthetic_trace(event_times_s=[])
    assert detect_events(t, v, DT, DetectionParams()) == []


# ---------------------------------------------------------------------------
# events_frame / grouping / data arrays
# ---------------------------------------------------------------------------

def test_events_frame_columns_and_inter_event_intervals():
    t, v = synthetic_trace()
    df = events_frame(detect_events(t, v, DT, DetectionParams()))
    for col in ["peak_idx", "peak_time_s", "baseline", "amplitude", "rise_time_ms",
                "decay_time_ms", "area", "inter_event_interval_ms"]:
        assert col in df.columns
    assert np.isnan(df["inter_event_interval_ms"].iloc[0]), "the first event has no preceding one"
    assert np.allclose(df["inter_event_interval_ms"].iloc[1:], 300.0, atol=1.0)


def test_events_frame_of_no_events_is_empty_but_well_shaped():
    df = events_frame([])
    assert df.empty
    assert "inter_event_interval_ms" in df.columns


def test_group_by_criteria_selects_a_range():
    t, v = synthetic_trace()
    df = events_frame(detect_events(t, v, DT, DetectionParams()))
    mask = group_by_criteria(df, peak_time_s=(0.35, 0.75))
    assert mask.sum() == 2
    assert np.allclose(df.loc[mask, "peak_time_s"], [0.40, 0.70], atol=0.001)


def test_group_random_is_reproducible_for_a_given_seed():
    a = group_random(100, 10, seed=1)
    b = group_random(100, 10, seed=1)
    assert a.sum() == 10
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# descriptive analysis
# ---------------------------------------------------------------------------

def test_column_statistics_on_a_known_sample():
    stats = column_statistics([1.0, 2.0, 3.0, 4.0, np.nan])
    assert stats.n == 4                                  # NaNs dropped, not counted
    assert stats.mean == pytest.approx(2.5)
    assert stats.variance == pytest.approx(5.0 / 3.0)    # sample variance, ddof=1
    assert stats.sd == pytest.approx(np.sqrt(5.0 / 3.0))


def test_column_statistics_of_nothing_is_not_an_error():
    stats = column_statistics([])
    assert stats.n == 0
    assert np.isnan(stats.mean)


def test_histograms_count_every_value_once():
    values = np.concatenate([np.full(10, 1.0), np.full(30, 2.0)])
    freq = frequency_histogram(values, bin_size=0.5)
    assert freq["count"].sum() == 40
    cum = cumulative_histogram(values, bin_size=0.5)
    assert cum["cumulative_fraction"].iloc[-1] == pytest.approx(1.0)
    assert (cum["cumulative_fraction"].diff().dropna() >= 0).all(), "must be non-decreasing"


# ---------------------------------------------------------------------------
# group analysis + decay fitting
# ---------------------------------------------------------------------------

def test_extract_and_scale_event_traces():
    t, v = synthetic_trace()
    events = detect_events(t, v, DT, DetectionParams(n_avg_peak=1))
    t_ms, traces, baselines, amplitudes = extract_event_traces(v, DT, events, pre_ms=5.0, post_ms=20.0)
    assert traces.shape[0] == len(events)
    assert traces.shape[1] == len(t_ms)
    assert t_ms[0] == pytest.approx(-5.0, abs=0.1) and t_ms[-1] == pytest.approx(20.0, abs=0.1)

    scaled = scale_traces(traces, baselines, amplitudes)
    peak_i = int(np.argmin(np.abs(t_ms)))
    assert np.allclose(scaled[:, peak_i], 1.0, atol=0.02), "every event should overlay at unit peak"


def test_fit_exponential_decay_recovers_the_planted_tau():
    t, v = synthetic_trace()
    events = detect_events(t, v, DT, DetectionParams())
    t_ms, traces, _, _ = extract_event_traces(v, DT, events, pre_ms=0.0, post_ms=25.0)
    avg = traces.mean(axis=0)
    fit = fit_exponential_decay(t_ms, avg, n_exp=1, fit_range="peak_to_end", direction="negative")
    assert fit.taus_ms[0] == pytest.approx(TAU_MS, rel=0.05)
    assert fit.offset == pytest.approx(0.0, abs=0.5)
    assert "exp(-t/" in fit.equation_str()


def test_double_exponential_fit_is_at_least_as_good_as_a_single_one():
    t, v = synthetic_trace()
    events = detect_events(t, v, DT, DetectionParams())
    t_ms, traces, _, _ = extract_event_traces(v, DT, events, pre_ms=0.0, post_ms=25.0)
    avg = traces.mean(axis=0)
    one = fit_exponential_decay(t_ms, avg, n_exp=1)
    two = fit_exponential_decay(t_ms, avg, n_exp=2)
    assert two.residual_sd <= one.residual_sd * 1.5  # more parameters shouldn't fit much worse

"""Tests for the deconvolution detector.

Mirrors tests/test_core.py's approach -- plant events of known amplitude, tau
and timing, then check they are recovered -- and adds the two cases the
classical detector's suite does not cover: OVERLAPPING events, and a
head-to-head precision comparison on a noisy trace.
"""

from __future__ import annotations

import numpy as np
import pytest

from minianalysis.core import DetectionParams, detect_events
from minianalysis.deconvolve import (
    DeconvParams,
    _synth,
    biexp_kernel,
    detect_events_deconv,
    estimate_kernel,
    noise_sd,
    robust_sd,
    score,
    track_baseline,
)
from test_core import DT, AMPLITUDE_PA, EVENT_TIMES_S, synthetic_trace


# --------------------------------------------------------------- basic recovery

def test_finds_the_planted_events_on_a_clean_trace():
    t, v = synthetic_trace()
    events = detect_events_deconv(t, v, DT)
    assert len(events) == len(EVENT_TIMES_S)
    found = sorted(e.peak_time_s for e in events)
    for got, want in zip(found, EVENT_TIMES_S):
        assert abs(got - want) < 1e-3          # within 1 ms


def test_amplitudes_are_signed_negative_and_about_right():
    t, v = synthetic_trace()
    events = detect_events_deconv(t, v, DT, DeconvParams(n_avg_peak=1))
    assert events
    for e in events:
        assert e.amplitude < 0                 # inward
        assert abs(e.amplitude - AMPLITUDE_PA) / abs(AMPLITUDE_PA) < 0.15


def test_area_and_kinetics_are_populated_and_signed():
    t, v = synthetic_trace()
    events = detect_events_deconv(t, v, DT)
    assert events
    for e in events:
        assert e.area < 0
        assert e.rise_time_ms > 0
        assert e.decay_time_ms > 0


def test_empty_trace_returns_no_events():
    assert detect_events_deconv(np.zeros(0), np.zeros(0), DT) == []


def test_flat_trace_returns_no_events():
    v = np.zeros(10_000)
    assert detect_events_deconv(None, v, DT) == []


def test_wrong_direction_finds_nothing():
    t, v = synthetic_trace()
    events = detect_events_deconv(t, v, DT, DeconvParams(direction="positive"))
    assert events == []


def test_positive_direction_finds_outward_events():
    t, v = synthetic_trace(amplitude_pa=+20.0)
    events = detect_events_deconv(t, v, DT, DeconvParams(direction="positive"))
    assert len(events) == len(EVENT_TIMES_S)
    assert all(e.amplitude > 0 for e in events)


# ------------------------------------------------------------------ the schema

def test_events_frame_schema_matches_the_classical_detector():
    from minianalysis.core import events_frame

    t, v = synthetic_trace(noise_sd=1.0)
    df_new = events_frame(detect_events_deconv(t, v, DT))
    df_old = events_frame(detect_events(t, v, DT, DetectionParams()))
    assert list(df_new.columns) == list(df_old.columns)
    assert df_new["peak_idx"].dtype == df_old["peak_idx"].dtype
    assert np.isnan(df_new["inter_event_interval_ms"].iloc[0])


# --------------------------------------------------------------- kernel & noise

def test_biexp_kernel_is_unit_peak_and_positive():
    k = biexp_kernel(DT, 0.6, 6.0)
    assert np.isclose(k.max(), 1.0)
    assert k.min() >= -1e-12
    assert int(np.argmax(k)) < k.size // 4       # peaks early, then decays


def test_estimate_kernel_falls_back_when_too_few_isolated_events():
    kern, n = estimate_kernel(np.zeros(5_000), DT)
    assert kern is None
    assert n < DeconvParams().kernel_min_events


def test_estimate_kernel_recovers_a_longer_kernel_for_slower_decay():
    _, v_fast, _ = _synth(tau_ms=3.0, duration_s=30.0, seed=3)
    _, v_slow, _ = _synth(tau_ms=25.0, duration_s=30.0, seed=3)
    k_fast, n_fast = estimate_kernel(v_fast, DT)
    k_slow, n_slow = estimate_kernel(v_slow, DT)
    assert k_fast is not None and k_slow is not None
    assert n_fast >= 5 and n_slow >= 5
    assert k_slow.size > k_fast.size             # adapts to the real decay


def test_noise_sd_beats_a_diff_based_estimate_on_a_filtered_trace():
    """diff(v)/sqrt(2) underestimates sigma once the trace is low-pass filtered,
    because filtering correlates adjacent samples. noise_sd must not."""
    from minianalysis.preprocess import bessel_lowpass

    rng = np.random.default_rng(0)
    fs = 1.0 / DT
    v = rng.normal(0, 5.0, 200_000)              # pure noise, sigma = 5 pA
    vf = bessel_lowpass(v, fs, cutoff_hz=1000.0, order=8)
    true_sd = float(vf.std())
    est = noise_sd(vf, DT)
    diff_est = robust_sd(np.diff(vf)) / np.sqrt(2)
    assert abs(est - true_sd) / true_sd < 0.25   # within 25%
    assert diff_est < 0.6 * true_sd              # the naive one really does fail


def test_track_baseline_follows_slow_drift():
    n = 100_000
    drift = 30.0 * np.sin(2 * np.pi * np.arange(n) / n)
    got = track_baseline(drift, DT, win_ms=300.0)
    assert np.max(np.abs(got - drift)) < 5.0


# ------------------------------------------- the cases the classical suite lacks

def test_resolves_overlapping_events_that_the_classical_detector_merges():
    """Two events 2 ms apart -- inside the classical detector's default
    search_local_max_ms of 3 ms, so find_peaks(distance=...) keeps only the
    taller one and the smaller is never even a candidate. Deconvolution
    separates them because each event becomes its own impulse.

    (At 6 ms apart the classical detector resolves both perfectly well; the
    failure is specific to separations below search_local_max_ms.)
    """
    fs = 1.0 / DT
    n = 20_000
    v = np.zeros(n)
    tau_ms, rise_n = 5.0, 6
    for onset_s, amp in ((0.50, -25.0), (0.502, -18.0)):
        oi = int(onset_s * fs)
        pi = oi + rise_n
        v[oi:pi] += amp * np.linspace(0, 1, rise_n, endpoint=False)
        dec = min(n - pi, int(10 * tau_ms / 1000 * fs))
        v[pi:pi + dec] += amp * np.exp(-np.arange(dec) * DT * 1e3 / tau_ms)

    new = detect_events_deconv(None, v, DT,
                               DeconvParams(fallback_tau_decay_ms=tau_ms))
    old = detect_events(None, v, DT,
                        DetectionParams(amplitude_threshold=5.0, area_threshold=5.0))
    assert len(new) == 2, f"deconvolution should resolve both, got {len(new)}"
    assert len(old) < 2, "classical detector is expected to merge them"


def test_precision_beats_the_classical_detector_on_a_noisy_trace():
    """The headline claim, as a regression test. On planted events in 2 pA
    noise the classical detector fragments decay tails; this must not."""
    t, v, true_s = _synth(rate_hz=5.0, duration_s=30.0, noise_sd_pa=2.0, seed=11)
    sigma = noise_sd(v, DT)

    new = detect_events_deconv(t, v, DT)
    s_new = score(true_s, [e.peak_time_s for e in new])
    old = detect_events(t, v, DT, DetectionParams(
        amplitude_threshold=max(5.0, 6.0 * sigma), area_threshold=10.0))
    s_old = score(true_s, [e.peak_time_s for e in old])

    assert s_new["f1"] > s_old["f1"]
    assert s_new["precision"] > 0.90
    assert s_new["recall"] > 0.85


@pytest.mark.parametrize("tau_ms", [3.0, 12.0, 25.0])
def test_holds_up_across_decay_time_constants(tau_ms):
    """The kernel is estimated, so performance must not depend on knowing tau."""
    t, v, true_s = _synth(rate_hz=5.0, duration_s=30.0, tau_ms=tau_ms,
                          noise_sd_pa=2.0, seed=5)
    s = score(true_s, [e.peak_time_s for e in detect_events_deconv(t, v, DT)])
    assert s["f1"] > 0.85, f"F1 {s['f1']:.3f} at tau={tau_ms} ms"


def test_drift_does_not_change_the_result_much():
    kw = dict(rate_hz=5.0, duration_s=30.0, noise_sd_pa=2.0, seed=9)
    t0, v0, ts0 = _synth(drift_pa=0.0, **kw)
    t1, v1, ts1 = _synth(drift_pa=40.0, **kw)
    f0 = score(ts0, [e.peak_time_s for e in detect_events_deconv(t0, v0, DT)])["f1"]
    f1 = score(ts1, [e.peak_time_s for e in detect_events_deconv(t1, v1, DT)])["f1"]
    assert abs(f0 - f1) < 0.05


def test_score_helper_is_sane():
    assert score([1.0, 2.0], [1.0, 2.0])["f1"] == 1.0
    assert score([1.0, 2.0], [])["f1"] == 0.0
    assert score([1.0], [1.0, 5.0])["fp"] == 1
    assert score([1.0, 5.0], [1.0])["fn"] == 1
    # one detection cannot match two true events
    assert score([1.0, 1.001], [1.0])["tp"] == 1

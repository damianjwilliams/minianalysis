#!/usr/bin/env python
"""Deconvolution-based synaptic event detection -- a fourth method.

Why this exists
---------------
The classical detector in `core.detect_events` finds candidates as local maxima
of the raw trace, then accepts them on amplitude and area. Two consequences are
structural, not tuning problems:

  * `find_peaks(distance=search_local_max_ms)` keeps only the TALLEST peak in
    each window, so a small event riding near a large one is never a candidate.
  * The baseline window sits a fixed 5-8 ms before the peak, which lands on the
    previous event's decay tail at short inter-event intervals. Noise on that
    tail then registers as extra local maxima that pass every gate -- one event
    is reported as several.

On a planted-event benchmark the second effect dominates: with white noise at
2 pA the classical detector's precision falls to ~0.57, and essentially every
false positive sits on the decay tail of a real event.

This module implements the deconvolution approach of Pernia-Andrade et al. 2012
(Biophys J 103:1429), which attacks both at once. Convolving the trace with the
inverse of the event kernel collapses each event to a sharp impulse: the decay
tail that manufactured the spurious maxima is gone, and summating events
separate. Three additions beyond the 2012 paper:

  1. The kernel is ESTIMATED FROM THE RECORDING (the average of well-isolated
     events, unit-peak normalised), so no decay time constant is assumed. Its
     length adapts to the measured decay. A fixed kernel is the single biggest
     failure mode -- assuming 6 ms on 25 ms data drops precision to 0.51.
  2. A matching-pursuit cleanup pass accepts candidates largest-first and
     subtracts each accepted event's kernel from the residual, so fragments
     riding on an accepted event's tail have no amplitude left and are dropped.
  3. Per-event measurement happens on a trace with all OTHER events' kernels
     subtracted, so the baseline is not contaminated by neighbouring decays.

Benchmark (22 synthetic conditions spanning decay tau 3-60 ms, event rate
1-40 Hz, white and 1/f noise, baseline drift, and amplitudes from SNR 15 down
to SNR 3; reproduce with `--benchmark`):

    mean F1   0.947  (deconvolution)   vs  0.734  (classical)
    worst F1  0.694                    vs  0.498
    wins 22/22 conditions

(Scored through the default 3 kHz / 10 kHz preprocessing. `--benchmark-raw`
scores on the unfiltered synthetic trace instead: mean 0.949 vs 0.737, worst
0.606 vs 0.403 -- filtering slightly helps the smallest events, which is where
the worst case lives, and slightly hurts the slowest ones.)

The classical detector is given its BEST amplitude threshold for each
condition, chosen against the ground truth -- an advantage it does not have on
real data. It still loses every condition. Precision is 1.000 in 15 of the 22.

Two caveats worth keeping in mind:

  * Those numbers come from PLANTED SYNTHETIC events. The events are
    biexponential, and this detector's kernel is estimated from the data, so
    the benchmark is kinder to it than to a fixed-template method. Treat the
    margin as indicative, not as a measured advantage on your recordings.
  * On real recordings there is no ground truth, and this detector is markedly
    more conservative than the classical one on some files (on one test file,
    722 events vs 1073-2116 depending on threshold). Review its output before
    trusting a count -- `python check_events.py <abf> --source deconv` opens
    the same manual event checker used for the classical detector.

Known limitation -- SLOW EVENTS. Kernel estimation averages events that no
other candidate comes within kernel_isolation_ms (default 60 ms) of. Once the
decay tau exceeds ~25 ms at a normal event rate, nothing is genuinely isolated:
every "isolated" event still sits on a predecessor's tail, the pre-peak median
over-subtracts, and the estimated kernel length becomes unstable -- measured
28-116 ms on the same tau=60 ms data depending only on recording duration, with
F1 swinging 0.80-0.93. It still beats the classical detector throughout that
range, but if your events are slow, raise --kernel-isolation-ms (and
--detrend-win-ms with it) or pin the kernel via --tau-decay-ms. The CLI warns
when the estimated kernel is long relative to the isolation window.

The recall floor at SNR ~3 is not a bug: it is the noise-limited detection
limit described by Greger & Watson (J Physiol 2025), who show that improving
sensitivity near the noise floor makes amplitude changes masquerade as
frequency changes. `--min-amp-pa` exists to enforce that limit explicitly.

Preprocessing defaults to a 3 kHz Bessel low-pass and decimation to 10 kHz --
the same as minianalysis.preprocess and the other detectors in these analysis
scripts, so thresholds and event counts are comparable across methods. Pass
--no-filter to work on the raw trace instead.

Usage:
    python -m minianalysis deconvolve recording.abf              # 3 kHz / 10 kHz
    python -m minianalysis deconvolve recording.abf --plot
    python -m minianalysis deconvolve recording.abf --no-filter  # raw trace
    python -m minianalysis deconvolve --benchmark

Then review the events by eye in the same window the classical detector uses:

    python check_events.py recording.abf --source deconv --filter

(--filter there because this detector filters by default, and the output
filenames encode that -- the reviewer has to rebuild the same trace, since
peak_idx values are indices into it.)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .core import Event, events_frame, output_stem
from .preprocess import DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ

EVENTS_SUFFIX = "_deconv_events.csv"
PARAMS_SUFFIX = "_deconv_params.json"
TRACE_SUFFIX = "_deconv_trace.png"
KERNEL_SUFFIX = "_deconv_kernel.csv"

# numpy renamed trapz -> trapezoid in 2.0; core.py still uses trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass(frozen=True)
class DeconvParams:
    """Parameters for deconvolution detection.

    Defaults were chosen by maximising WORST-CASE F1 across the benchmark
    conditions, not mean F1 -- a detector that is never bad matters more here
    than one that is occasionally excellent.
    """

    # --- detection ---
    thresh_sd: float = 6.0
    """Threshold on the deconvolved trace, in robust SDs of that trace.

    The deconvolved trace is near-white, so its SD is a meaningful noise unit.
    Below ~5 the fragment count climbs steeply; above ~8 real small events go.
    """

    reg: float = 0.01
    """Wiener regularisation lambda, as a fraction of max|K(f)|^2.

    Controls the noise-amplification/sharpness tradeoff. Larger = smoother
    deconvolution, blunter impulses, more fragments. 0.01 was optimal across
    every condition tested; 0.15 is clearly too large (F1 0.06 at 3 SD).
    """

    min_amp_sd: float = 2.5
    """Matching-pursuit acceptance floor, in robust SDs of the RAW trace.

    Applied to each candidate's amplitude measured on the residual after
    larger events have been subtracted. This is what kills fragments.
    """

    min_amp_pa: float = 0.0
    """Optional ABSOLUTE amplitude floor in trace units, on top of min_amp_sd.

    Off by default -- every other threshold here is relative to the measured
    noise, which is what makes the detector self-calibrating. Set this to
    ~4*sigma if you want the Greger & Watson (J Physiol 2025) detection limit
    enforced as a hard cutoff rather than left implicit.
    """

    min_sep_ms: float = 1.5
    """Minimum separation between deconvolved impulses (find_peaks distance)."""

    smooth_ms: float = 0.3
    """Boxcar smoothing of the deconvolved trace before peak-picking."""

    # --- preprocessing ---
    detrend: bool = True
    detrend_win_ms: float = 300.0
    """Rolling-percentile window for slow baseline drift removal.

    Tracks the upper envelope (events are inward), so drift is removed without
    biting into events. Verified flat F1 from 0 to +/-40 pA of drift.
    """
    detrend_pct: float = 80.0

    # --- kernel estimation ---
    kernel_max_ms: float = 400.0
    """Upper bound on the averaging window; the kernel is cut shorter than this
    wherever the averaged waveform has decayed (see kernel_tail_frac)."""
    kernel_tail_frac: float = 0.02
    """Cut the kernel where the average stays below this fraction of peak."""
    kernel_tail_run_ms: float = 2.0
    """...for at least this long, so one noisy dip cannot truncate the kernel."""
    kernel_pre_ms: float = 5.0
    kernel_isolation_ms: float = 60.0
    """An event counts as isolated if no other candidate is within this."""
    kernel_min_snr: float = 5.0
    kernel_n_max: int = 300
    kernel_smooth_ms: float = 0.4
    kernel_min_events: int = 5
    """Below this many isolated events, fall back to the analytic kernel."""
    fallback_tau_rise_ms: float = 0.6
    fallback_tau_decay_ms: float = 6.0

    # --- measurement (kept identical in meaning to core.DetectionParams so the
    #     CSV columns are directly comparable between the two detectors) ---
    direction: str = "negative"
    n_avg_peak: int = 3
    onset_fraction: float = 0.1
    decay_fraction: float = 0.5
    onset_search_ms: float = 10.0
    decay_search_ms: float = 50.0
    baseline_before_ms: float = 3.0
    baseline_avg_ms: float = 3.0


def _n(ms: float, dt: float) -> int:
    """Milliseconds to samples, at least 1 (mirrors core._samples)."""
    return max(1, int(round(ms / 1000.0 / dt)))


def _baseline_pct(p: "DeconvParams") -> float:
    """Which percentile tracks the baseline, given the event polarity.

    Inward (negative) events pull the trace down, so the UPPER percentile is
    the baseline. Outward events push it up, so the mirror image applies --
    using 80 for outward events bites into the events and leaves residuals
    that then register as spurious near-zero-amplitude detections.
    """
    return p.detrend_pct if p.direction == "negative" else 100.0 - p.detrend_pct


def robust_sd(x: np.ndarray) -> float:
    """MAD-based SD estimate, insensitive to the events themselves."""
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def track_baseline(v: np.ndarray, dt: float, win_ms: float = 300.0,
                   pct: float = 80.0) -> np.ndarray:
    """Slow drift estimate: rolling percentile, linearly interpolated.

    For inward events the upper percentile tracks the true baseline, since
    events only pull the trace down.
    """
    v = np.asarray(v, float)
    n = v.size
    if n == 0:
        return np.zeros(0)
    w = max(3, _n(win_ms, dt))
    step = max(1, w // 4)
    idx = np.arange(0, n, step)
    vals = np.empty(idx.size)
    half = w // 2
    for i, s in enumerate(idx):
        seg = v[max(0, s - half): s + half]
        vals[i] = np.percentile(seg, pct) if seg.size else v[s]
    if idx.size == 1:
        return np.full(n, vals[0])
    return np.interp(np.arange(n), idx, vals)


def noise_sd(v: np.ndarray, dt: float, p: "DeconvParams | None" = None) -> float:
    """Trace noise SD, estimated from the event-FREE side of the distribution.

    Events are inward (negative), so after detrending the POSITIVE half of the
    residual contains noise only; mirroring it gives a clean symmetric sample.

    This matters: estimating sigma from diff(v) UNDERESTIMATES badly on a
    low-pass-filtered trace -- by 3.6-4.9x on the real recordings tested here,
    because filtering correlates adjacent samples. Estimating it from the raw
    trace spread instead is inflated by the events themselves.
    """
    p = p or DeconvParams()
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    # MEDIAN-centred detrend here, not the upper-percentile envelope used for
    # detection: with pct=80 the residual is offset by ~0.84 sigma, so the
    # mirrored quiet half samples only the tail and sigma comes out ~34% low.
    d = v - track_baseline(v, dt, p.detrend_win_ms, 50.0)
    pol = -1.0 if p.direction == "negative" else 1.0
    quiet = d[pol * d < 0]           # the side away from the events
    if quiet.size < 10:
        return robust_sd(d)
    return robust_sd(np.concatenate([quiet, -quiet]))


def biexp_kernel(dt: float, tau_rise_ms: float = 0.6, tau_decay_ms: float = 6.0,
                 length_ms: "float | None" = None) -> np.ndarray:
    """Unit-peak biexponential PSC kernel, positive-going. Analytic fallback."""
    if length_ms is None:
        length_ms = 8.0 * tau_decay_ms
    n = max(4, _n(length_ms, dt))
    t = np.arange(n) * dt * 1000.0
    k = np.exp(-t / tau_decay_ms) - np.exp(-t / tau_rise_ms)
    pk = k.max()
    return k / pk if pk > 0 else k


def estimate_kernel(v: np.ndarray, dt: float,
                    p: "DeconvParams | None" = None):
    """Average of well-isolated events, unit-peak, adaptively truncated.

    Returns (kernel, n_isolated). kernel is None if too few isolated events
    were found -- the caller then uses the analytic fallback.
    """
    p = p or DeconvParams()
    v = np.asarray(v, float)
    if v.size < 16:
        return None, 0
    pol = -1.0 if p.direction == "negative" else 1.0
    vd = pol * (v - track_baseline(v, dt, p.detrend_win_ms, _baseline_pct(p)))
    sd = robust_sd(vd)
    if sd <= 0:
        return None, 0
    iso_n = _n(p.kernel_isolation_ms, dt)
    cand, props = find_peaks(vd, height=p.kernel_min_snr * sd, distance=iso_n)
    if cand.size < p.kernel_min_events:
        return None, int(cand.size)
    keep = np.argsort(props["peak_heights"])[::-1][:p.kernel_n_max]
    cand = np.sort(cand[keep])
    pre_n = max(2, _n(p.kernel_pre_ms, dt))
    post_n = max(8, _n(p.kernel_max_ms, dt))
    segs = []
    for i, q in enumerate(cand):
        if q - pre_n < 0 or q + post_n >= vd.size:
            continue
        if i > 0 and q - cand[i - 1] < iso_n:
            continue
        if i < cand.size - 1 and cand[i + 1] - q < iso_n:
            continue
        seg = vd[q - pre_n: q + post_n].astype(float)
        seg -= np.median(seg[:pre_n])
        if seg[pre_n] <= 0:
            continue
        segs.append(seg / seg[pre_n])
    if len(segs) < p.kernel_min_events:
        return None, len(segs)
    avg = np.mean(segs, axis=0)
    kern = avg[pre_n:].copy()
    sm = _n(p.kernel_smooth_ms, dt)
    if sm > 1:
        kern = np.convolve(kern, np.ones(sm) / sm, mode="same")
    pk = kern.max()
    if pk <= 0:
        return None, len(segs)
    # Adaptive length. Two criteria, and the ORDER matters:
    #
    #  (a) Primary: the 1/e crossing. Take tau as the time for the average to
    #      fall to 1/e of peak, and keep 8*tau. This is robust to a constant
    #      baseline offset in the average, which matters because at long decay
    #      times no event is truly isolated (kernel_isolation_ms is far shorter
    #      than a slow tail), so the pre-peak median over-subtracts and pushes
    #      the whole tail down. Keying off 2%-of-peak under that offset cuts the
    #      kernel far too early -- measured 7-18 ms kernels on 60 ms decays,
    #      with F1 swinging 0.79-0.93 depending only on recording length.
    #  (b) Secondary cap: the tail must still STAY below tail_frac for
    #      tail_run_ms, so a genuinely short kernel is not padded with noise.
    #      A single-crossing version of this test truncates on noise.
    e_cross = np.nonzero(kern <= pk / np.e)[0]
    tau_n = int(e_cross[0]) if e_cross.size else kern.size
    cut = min(kern.size, max(4, 8 * max(tau_n, 1)))

    run_n = max(2, _n(p.kernel_tail_run_ms, dt))
    low = (kern < p.kernel_tail_frac * pk).astype(int)
    csum = np.concatenate(([0], np.cumsum(low)))
    for i in range(max(0, kern.size - run_n)):
        if csum[i + run_n] - csum[i] == run_n:
            cut = min(cut, max(i, 8 * max(tau_n, 1)))
            break
    cut = max(cut, max(4, _n(2.0, dt)))
    cut = min(cut, kern.size)
    kern = kern[:cut].copy()
    taper = np.linspace(1.0, 0.0, max(2, kern.size // 5))
    kern[-taper.size:] *= taper
    kern = np.clip(kern, 0.0, None)
    mx = kern.max()
    if mx <= 0:
        return None, len(segs)
    return kern / mx, len(segs)


def wiener_deconvolve(v: np.ndarray, kernel: np.ndarray, reg: float = 0.01,
                      smooth_n: int = 0) -> np.ndarray:
    """S(f) = Y(f) K*(f) / (|K(f)|^2 + lambda), lambda = reg * max|K|^2."""
    v = np.asarray(v, float)
    n = v.size
    nfft = 1 << int(np.ceil(np.log2(n + kernel.size)))
    Y = np.fft.rfft(v, nfft)
    K = np.fft.rfft(kernel, nfft)
    K2 = (K.conj() * K).real
    lam = reg * K2.max()
    s = np.fft.irfft(Y * K.conj() / (K2 + lam), nfft)[:n]
    if smooth_n and smooth_n > 1:
        s = np.convolve(s, np.ones(smooth_n) / smooth_n, mode="same")
    return s


def peel_refine(vd: np.ndarray, dt: float, cand: np.ndarray, kernel: np.ndarray,
                heights: np.ndarray, min_amp: float,
                p: "DeconvParams | None" = None):
    """Matching pursuit: accept largest-first, subtracting each accepted kernel.

    Returns (peak_idx, amplitude) in time order. amplitude is positive-going in
    the polarity-flipped frame; the caller re-signs it.
    """
    p = p or DeconvParams()
    resid = np.asarray(vd, float).copy()
    klen = kernel.size
    pre_n = _n(p.baseline_before_ms, dt)
    avg_n = max(1, p.n_avg_peak)
    accepted: list = []
    amps: list = []
    for j in np.argsort(heights)[::-1]:
        q = int(cand[j])
        a0 = max(0, q - avg_n // 2)
        a1 = min(resid.size, a0 + avg_n)
        peak_v = float(np.mean(resid[a0:a1]))
        b1 = q - pre_n
        b0 = max(0, b1 - _n(p.baseline_avg_ms, dt))
        base = float(np.median(resid[b0:b1])) if b1 > b0 else 0.0
        amp = peak_v - base
        if amp < min_amp:
            continue
        accepted.append(q)
        amps.append(amp)
        e1 = min(resid.size, q + klen)
        resid[q:e1] -= amp * kernel[:e1 - q]
    if not accepted:
        return np.array([], int), np.array([], float)
    order = np.argsort(accepted)
    return np.array(accepted, int)[order], np.array(amps, float)[order]


def _measure(vc: np.ndarray, dt: float, q: int, p: DeconvParams):
    """Measure one event on `vc`, a trace with other events' kernels removed.

    Definitions match core.detect_events exactly (onset at onset_fraction of
    span, decay at decay_fraction, area integrated onset..decay) so the CSV
    columns mean the same thing in both detectors' output.
    """
    pol = -1.0 if p.direction == "negative" else 1.0
    sv = pol * vc
    n = sv.size
    avg_n = max(1, p.n_avg_peak)
    a0 = max(0, q - avg_n // 2)
    a1 = min(n, a0 + avg_n)
    peak_v = float(np.mean(vc[a0:a1]))
    b1 = q - _n(p.baseline_before_ms, dt)
    b0 = b1 - _n(p.baseline_avg_ms, dt)
    if b0 < 0:
        return None
    baseline = float(np.mean(vc[b0:b1]))
    amplitude = peak_v - baseline
    span = pol * peak_v - pol * baseline
    if span <= 0:
        return None
    s_base = pol * baseline
    onset_level = s_base + p.onset_fraction * span
    decay_level = s_base + (1.0 - p.decay_fraction) * span

    lo = max(0, q - _n(p.onset_search_ms, dt))
    seg = sv[lo:q + 1]
    below = np.nonzero(seg < onset_level)[0]
    if below.size == 0:
        return None
    onset_idx = lo + int(below[-1])

    hi = min(n, q + _n(p.decay_search_ms, dt))
    seg2 = sv[q:hi]
    at = np.nonzero(seg2 <= decay_level)[0]
    if at.size == 0:
        return None
    decay_idx = q + int(at[0])

    area = float(_trapz(vc[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
    return Event(
        peak_idx=int(q),
        peak_time_s=float(q * dt),
        baseline=baseline,
        amplitude=float(amplitude),
        rise_time_ms=float((q - onset_idx) * dt * 1e3),
        decay_time_ms=float((decay_idx - q) * dt * 1e3),
        area=area,
    )


def detect_events_deconv(t, v: np.ndarray, dt: float,
                         params: "DeconvParams | None" = None,
                         return_diagnostics: bool = False):
    """Detect events by deconvolution. Signature mirrors core.detect_events.

    `t` is accepted and ignored (peak_time_s is peak_idx*dt), as in core.
    Returns list[Event], or (list[Event], diagnostics) if requested.
    """
    del t                       # accepted for signature parity with core only
    p = params or DeconvParams()
    v = np.asarray(v, float)
    pol = -1.0 if p.direction == "negative" else 1.0
    if v.size < 16:
        empty = dict(kernel=np.zeros(0), kernel_source="none", n_isolated=0,
                     deconv=np.zeros(0), deconv_sd=0.0, raw_sd=0.0,
                     drift=np.zeros(v.size), n_candidates=0, n_after_peel=0,
                     n_measured=0)
        return ([], empty) if return_diagnostics else []

    kern, n_iso = estimate_kernel(v, dt, p)
    kernel_source = "estimated"
    if kern is None:
        kern = biexp_kernel(dt, p.fallback_tau_rise_ms, p.fallback_tau_decay_ms)
        kernel_source = "analytic-fallback"

    drift = (track_baseline(v, dt, p.detrend_win_ms, _baseline_pct(p))
             if p.detrend else np.zeros(v.size))
    vd = pol * (v - drift)
    raw_sd = robust_sd(vd)

    s = wiener_deconvolve(vd, kern, reg=p.reg, smooth_n=_n(p.smooth_ms, dt))
    dsd = robust_sd(s)
    idx, props = find_peaks(s, height=p.thresh_sd * dsd,
                            distance=_n(p.min_sep_ms, dt))

    diag = dict(kernel=kern, kernel_source=kernel_source, n_isolated=n_iso,
                deconv=s, deconv_sd=dsd, raw_sd=raw_sd, drift=drift,
                n_candidates=int(idx.size), n_after_peel=0, n_measured=0)
    if idx.size == 0:
        return ([], diag) if return_diagnostics else []

    # snap each impulse to the nearest local maximum of the raw trace
    r = _n(2.0, dt)
    snapped, heights = [], []
    for i, h in zip(idx, props["peak_heights"]):
        lo, hi = max(0, i - r), min(vd.size, i + r + 1)
        snapped.append(lo + int(np.argmax(vd[lo:hi])))
        heights.append(float(h))
    snapped = np.array(snapped, int)
    heights = np.array(heights, float)
    _, first = np.unique(snapped, return_index=True)
    snapped, heights = snapped[first], heights[first]

    # On a synthetic noiseless trace raw_sd is 0, which would make the floor 0
    # and admit numerically-zero residuals as "events"; fall back to a small
    # fraction of the largest deflection there.
    if raw_sd > 0:
        min_amp = p.min_amp_sd * raw_sd
    else:
        min_amp = 0.01 * float(np.abs(vd).max()) if vd.size else 0.0
    min_amp = max(min_amp, p.min_amp_pa)
    peaks, amps = peel_refine(vd, dt, snapped, kern, heights, min_amp, p)
    diag["n_after_peel"] = int(peaks.size)
    if peaks.size == 0:
        return ([], diag) if return_diagnostics else []

    # Per-event measurement on a trace with all OTHER events removed, so a
    # neighbour's decay tail cannot contaminate this event's baseline.
    klen = kern.size
    model = np.zeros(v.size)
    for q, a in zip(peaks, amps):
        e1 = min(v.size, q + klen)
        model[q:e1] += a * kern[:e1 - q]

    events = []
    own = np.zeros(v.size)
    for q, a in zip(peaks, amps):
        e1 = min(v.size, q + klen)
        own[:] = 0.0
        own[q:e1] = a * kern[:e1 - q]
        vc = v - pol * (model - own)      # back to raw units and polarity
        ev = _measure(vc, dt, int(q), p)
        if ev is not None:
            events.append(ev)
    diag["n_measured"] = len(events)
    return (events, diag) if return_diagnostics else events


# ---------------------------------------------------------------- benchmarking

def _synth(fs=10_000.0, rate_hz=5.0, duration_s=60.0, amp_mean=-20.0,
           amp_cv=0.35, tau_ms=6.0, rise_ms=0.6, noise_sd_pa=2.0,
           drift_pa=0.0, pink=False, seed=0):
    """Poisson train of planted events with known peak times. Ground truth."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / fs
    n = int(duration_s * fs)
    v = np.zeros(n)
    n_ev = rng.poisson(rate_hz * duration_s)
    onsets = np.sort(rng.uniform(0.05, duration_s - 0.2, n_ev))
    amps = np.clip(rng.normal(amp_mean, abs(amp_mean) * amp_cv, n_ev), None, -1.0)
    taus = np.clip(rng.normal(tau_ms, tau_ms * 0.25, n_ev), 1.0, None)
    rise_n = max(1, int(round(rise_ms / 1000 * fs)))
    peaks = []
    for onset, a, tau in zip(onsets, amps, taus):
        oi = int(onset * fs)
        pi = oi + rise_n
        if pi + 1 >= n:
            continue
        v[oi:pi] += a * np.linspace(0, 1, rise_n, endpoint=False)
        dec_n = min(n - pi, int(10 * tau / 1000 * fs))
        v[pi:pi + dec_n] += a * np.exp(-np.arange(dec_n) * dt * 1000.0 / tau)
        peaks.append(pi * dt)
    if drift_pa:
        v += drift_pa * np.sin(2 * np.pi * np.arange(n) / n * 3)
    if noise_sd_pa:
        if pink:
            f = np.fft.rfftfreq(n, dt)
            f[0] = f[1]
            x = np.fft.irfft(np.exp(1j * rng.uniform(0, 2 * np.pi, f.size))
                             / np.sqrt(f), n)
            v += x / x.std() * noise_sd_pa
        else:
            v += rng.normal(0, noise_sd_pa, n)
    return np.arange(n) * dt, v, np.array(peaks)


def score(true_s, det_s, tol_ms=3.0):
    """Greedy one-to-one match within tol_ms. Returns precision/recall/f1."""
    true_s = np.asarray(true_s, float)
    det_s = np.asarray(det_s, float)
    if det_s.size == 0:
        return dict(tp=0, fp=0, fn=int(true_s.size), precision=0.0,
                    recall=0.0, f1=0.0)
    tol = tol_ms / 1000.0
    used = np.zeros(det_s.size, bool)
    tp = 0
    for ts in true_s:
        d = np.abs(det_s - ts)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    fp = int((~used).sum())
    fn = int(true_s.size - tp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec, f1=f1)


def run_benchmark(duration_s=60.0, seed=1, verbose=True, filtered=True):
    """Head-to-head against core.detect_events over 22 planted conditions.

    `filtered` applies the same 3 kHz Bessel low-pass the CLI uses by default,
    so the benchmark exercises the preprocessing the detector actually runs
    under. It is not cosmetic: filtering correlates adjacent noise samples,
    which is the regime where naive noise-SD estimates fail and where smooth
    noise bumps survive averaging more easily. Pass filtered=False to score on
    the raw synthetic trace instead.
    """
    from .core import DetectionParams, detect_events as classical
    from .preprocess import bessel_lowpass

    fs = 10_000.0
    dt = 1.0 / fs
    conds = []
    for td in (3.0, 6.0, 12.0, 25.0, 40.0, 60.0):
        conds.append((f"tau_decay={td:g} ms", dict(tau_ms=td)))
    for r in (1, 5, 10, 20, 40):
        conds.append((f"rate={r} Hz", dict(rate_hz=float(r))))
    for ns in (1.0, 2.0, 4.0):
        conds.append((f"pink noise sd={ns:g}", dict(noise_sd_pa=ns, pink=True)))
    for dr in (0.0, 10.0, 40.0):
        conds.append((f"drift +/-{dr:g} pA", dict(drift_pa=dr)))
    for am in (-30.0, -20.0, -12.0, -8.0, -6.0):
        conds.append((f"amp={am:g} pA", dict(amp_mean=am)))

    rows = []
    if verbose:
        print(f"preprocessing: "
              f"{'3 kHz Bessel low-pass at 10 kHz (the CLI default)' if filtered else 'none (raw synthetic trace)'}")
        print(f"{'condition':26s} {'F1_new':>7} {'P_new':>6} {'R_new':>6} "
              f"{'F1_old':>7} {'P_old':>6} {'R_old':>6} {'delta':>7}")
        print("-" * 80)
    for label, kw in conds:
        kw = dict(dict(rate_hz=5.0, duration_s=duration_s, noise_sd_pa=2.0,
                       seed=seed), **kw)
        t, v, true_s = _synth(fs=fs, **kw)
        if filtered:
            # already at the 10 kHz target, so this is the low-pass only
            v = bessel_lowpass(v, fs, cutoff_hz=DEFAULT_CUTOFF_HZ,
                               order=DEFAULT_ORDER)
        new = detect_events_deconv(t, v, dt, DeconvParams())
        sn = score(true_s, [e.peak_time_s for e in new])

        # Give the classical detector its BEST threshold for this condition,
        # chosen with knowledge of the ground truth it would not have in
        # practice. Pinning it at one multiple of sigma understates it badly
        # at low amplitudes, and a comparison worth reporting should not rest
        # on handicapping the baseline.
        sigma = noise_sd(v, dt)
        so = None
        for mult in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            cand = classical(t, v, dt, DetectionParams(
                amplitude_threshold=mult * sigma, area_threshold=10.0))
            s = score(true_s, [e.peak_time_s for e in cand])
            if so is None or s["f1"] > so["f1"]:
                so = s
        rows.append((label, sn, so))
        if verbose:
            print(f"{label:26s} {sn['f1']:7.3f} {sn['precision']:6.3f} "
                  f"{sn['recall']:6.3f} {so['f1']:7.3f} {so['precision']:6.3f} "
                  f"{so['recall']:6.3f} {sn['f1'] - so['f1']:+7.3f}")
    fn = np.array([r[1]["f1"] for r in rows])
    fo = np.array([r[2]["f1"] for r in rows])
    if verbose:
        print("-" * 80)
        print(f"{'MEAN':26s} {fn.mean():7.3f} {'':6s} {'':6s} {fo.mean():7.3f}")
        print(f"{'WORST':26s} {fn.min():7.3f} {'':6s} {'':6s} {fo.min():7.3f}")
        print(f"\ndeconvolution wins {(fn > fo).sum()}/{len(fn)} conditions")
    return rows


# ------------------------------------------------------------------------ main

def _load(abf_path, channel, filter_enabled, cutoff_hz, target_rate_hz, order):
    """Load a trace, optionally Bessel-filtered and decimated. No Qt needed."""
    import pyabf
    from .preprocess import bessel_lowpass, downsample

    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=channel)
    v = np.asarray(abf.sweepY, float)
    fs = float(abf.dataRate)
    unit = abf.adcUnits[channel] if channel < len(abf.adcUnits) else "pA"
    if filter_enabled:
        v = bessel_lowpass(v, fs, cutoff_hz=cutoff_hz, order=order)
        v, fs = downsample(v, fs, target_hz=target_rate_hz)
    return np.arange(v.size) / fs, v, 1.0 / fs, unit


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="minianalysis deconvolve",
        description="Deconvolution-based sEPSC detection (Pernia-Andrade 2012, "
                    "with a data-estimated kernel and matching-pursuit cleanup).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("abf", nargs="?", help="path to a gap-free .abf recording")
    ap.add_argument("--channel", type=int, default=0)
    # Filtering is ON by default, at the same 3 kHz / 10 kHz the rest of these
    # analysis scripts use (minianalysis.preprocess's defaults; `fastmini` in
    # the sibling sepsc package is likewise fixed at 3 kHz/10 kHz). Detecting
    # on the raw trace is still available via --no-filter, but then thresholds
    # tuned on filtered data do not carry over -- the noise floor differs.
    ap.add_argument("--no-filter", dest="filter_enabled", action="store_false",
                    help="detect on the RAW trace instead of Bessel low-pass "
                         "+ decimate (filtering is the default)")
    ap.set_defaults(filter_enabled=True)
    ap.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                    help="Bessel low-pass cutoff")
    ap.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                    help="decimate to approximately this sample rate")
    ap.add_argument("--filter-order", type=int, default=DEFAULT_ORDER)
    ap.add_argument("--thresh-sd", type=float, default=DeconvParams.thresh_sd)
    ap.add_argument("--reg", type=float, default=DeconvParams.reg)
    ap.add_argument("--min-amp-sd", type=float, default=DeconvParams.min_amp_sd)
    ap.add_argument("--min-amp-pa", type=float, default=DeconvParams.min_amp_pa,
                    help="absolute amplitude floor in trace units; 0 disables. "
                         "Set to 4x the reported noise sigma to enforce the "
                         "Greger & Watson detection limit as a hard cutoff.")
    ap.add_argument("--min-sep-ms", type=float, default=DeconvParams.min_sep_ms)
    ap.add_argument("--no-detrend", dest="detrend", action="store_false")
    ap.add_argument("--tau-decay-ms", type=float,
                    default=DeconvParams.fallback_tau_decay_ms,
                    help="analytic-kernel decay, used only if kernel estimation "
                         "finds too few isolated events")
    ap.add_argument("--kernel-isolation-ms", type=float,
                    default=DeconvParams.kernel_isolation_ms,
                    help="an event counts as isolated for kernel estimation if "
                         "no other candidate is within this. RAISE IT for slow "
                         "events: with a decay tau above ~25 ms at a high event "
                         "rate, nothing is isolated at the default and the "
                         "estimated kernel becomes unstable")
    ap.add_argument("--detrend-win-ms", type=float,
                    default=DeconvParams.detrend_win_ms,
                    help="rolling-percentile window for drift removal. Must stay "
                         "well above the event decay, or the detrender absorbs "
                         "the tail it is trying to preserve")
    ap.add_argument("--plot", action="store_true",
                    help="write a diagnostic PNG (trace, marks, deconvolved trace, kernel)")
    ap.add_argument("--benchmark", action="store_true",
                    help="run the planted-event benchmark and exit")
    ap.add_argument("--benchmark-raw", action="store_true",
                    help="run the benchmark WITHOUT the default 3 kHz filter")
    args = ap.parse_args(argv)

    if args.benchmark or args.benchmark_raw:
        run_benchmark(filtered=not args.benchmark_raw)
        return 0
    if not args.abf:
        ap.error("an .abf path is required (or use --benchmark)")
    if not os.path.isfile(args.abf):
        ap.error(f"no such file: {args.abf}")

    p = DeconvParams(thresh_sd=args.thresh_sd, reg=args.reg,
                     min_amp_sd=args.min_amp_sd, min_amp_pa=args.min_amp_pa,
                     min_sep_ms=args.min_sep_ms,
                     detrend=args.detrend,
                     detrend_win_ms=args.detrend_win_ms,
                     kernel_isolation_ms=args.kernel_isolation_ms,
                     fallback_tau_decay_ms=args.tau_decay_ms)

    t, v, dt, unit = _load(args.abf, args.channel, args.filter_enabled,
                           args.cutoff_hz, args.target_rate_hz, args.filter_order)
    dur = v.size * dt
    sigma = noise_sd(v, dt, p)
    prep = (f"Bessel {args.cutoff_hz:.0f} Hz order {args.filter_order} "
            f"-> {1/dt:.0f} Hz" if args.filter_enabled else "raw, unfiltered")
    print(f"{os.path.basename(args.abf)}: {dur:.1f} s at {1/dt:.0f} Hz, "
          f"channel {args.channel}, units {unit}")
    print(f"preprocessing: {prep}")
    print(f"noise sigma = {sigma:.2f} {unit}  "
          f"(4 sigma detection limit = {4*sigma:.2f} {unit})")

    events, diag = detect_events_deconv(t, v, dt, p, return_diagnostics=True)
    kernel_ms = diag["kernel"].size * dt * 1e3
    print(f"kernel: {diag['kernel_source']}, {kernel_ms:.1f} ms"
          f" from {diag['n_isolated']} isolated events")
    if diag["kernel_source"] == "estimated" and kernel_ms > 0.5 * p.kernel_isolation_ms:
        print(f"WARNING: the kernel ({kernel_ms:.0f} ms) is long relative to the "
              f"isolation window ({p.kernel_isolation_ms:.0f} ms), so the events "
              f"averaged into it were not really isolated and the estimate may "
              f"be unstable. Try --kernel-isolation-ms "
              f"{max(4 * kernel_ms, 2 * p.kernel_isolation_ms):.0f}.")
    print(f"candidates {diag['n_candidates']} -> after peel {diag['n_after_peel']}"
          f" -> measured {diag['n_measured']}")
    print(f"{len(events)} events, {len(events)/dur:.2f} Hz")

    df = events_frame(events)
    stem = output_stem(args.abf, channel=args.channel,
                       filter_enabled=args.filter_enabled,
                       cutoff_hz=args.cutoff_hz,
                       target_rate_hz=args.target_rate_hz)
    out_csv = stem + EVENTS_SUFFIX
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    with open(stem + PARAMS_SUFFIX, "w", encoding="utf-8") as fh:
        json.dump(dict(dataclasses.asdict(p),
                       filter_enabled=args.filter_enabled,
                       cutoff_hz=args.cutoff_hz if args.filter_enabled else None,
                       filter_order=args.filter_order if args.filter_enabled else None,
                       noise_sd=sigma, detection_limit_4sd=4 * sigma,
                       kernel_source=diag["kernel_source"],
                       kernel_ms=diag["kernel"].size * dt * 1e3,
                       n_isolated_for_kernel=diag["n_isolated"],
                       sample_rate_hz=1.0 / dt, duration_s=dur,
                       n_events=len(events)), fh, indent=2)
    print(f"wrote {stem + PARAMS_SUFFIX}")

    np.savetxt(stem + KERNEL_SUFFIX, diag["kernel"], delimiter=",",
               header="unit_peak_kernel", comments="")

    # Point the operator straight at the reviewer for THIS detector's output,
    # with the filtering flag that matches the stem these files were written
    # under (a mismatch there is the usual reason `check` cannot find them).
    print(f"review with: python check_events.py {args.abf} --source deconv"
          f"{' --filter' if args.filter_enabled else ''}")

    if args.plot:
        _plot(t, v, dt, unit, events, diag, p, stem + TRACE_SUFFIX)
        print(f"wrote {stem + TRACE_SUFFIX}")
    return 0


def _plot(t, v, dt, unit, events, diag, p, out_png, window_s=3.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fs = 1.0 / dt
    pk = np.array([e.peak_idx for e in events], int)
    mid = v.size // 2
    a = max(0, mid - int(window_s * fs / 2))
    b = min(v.size, a + int(window_s * fs))
    fig, ax = plt.subplots(3, 1, figsize=(14, 8),
                           gridspec_kw=dict(height_ratios=[3, 1.4, 1.4]))
    ax[0].plot(t[a:b], v[a:b], lw=0.5, color="0.25")
    m = (pk >= a) & (pk < b)
    if m.any():
        ax[0].plot(t[pk[m]], v[pk[m]], "v", ms=7, color="tab:green")
    ax[0].plot(t[a:b], diag["drift"][a:b], lw=1.0, color="tab:orange",
               label="tracked baseline")
    ax[0].set_title(f"detected events ({m.sum()} in this {window_s:g} s window; "
                    f"{len(events)} total)", loc="left")
    ax[0].set_ylabel(unit)
    ax[0].legend(loc="lower right", fontsize=8)
    s = diag["deconv"]
    ax[1].plot(t[a:b], s[a:b], lw=0.5, color="tab:blue")
    ax[1].axhline(p.thresh_sd * diag["deconv_sd"], ls="--", lw=1,
                  color="tab:orange", label=f"{p.thresh_sd:g} sigma")
    ax[1].set_title("deconvolved trace", loc="left")
    ax[1].set_xlabel("time (s)")
    ax[1].legend(loc="upper right", fontsize=8)
    kt = np.arange(diag["kernel"].size) * dt * 1e3
    ax[2].plot(kt, diag["kernel"], color="tab:red")
    ax[2].set_title(f"estimated kernel ({diag['kernel_source']}, "
                    f"{diag['n_isolated']} isolated events)", loc="left")
    ax[2].set_xlabel("ms")
    for x in ax:
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())

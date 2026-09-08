"""
The detector itself: a classical local-maximum + baseline + amplitude/area-
threshold synaptic event detector, plus the downstream analysis stage,
reimplementing the method described in the Synaptosoft "Mini Analysis
Program" tutorial (C. Justin Lee, PhD; recovered from archive.org --
slides 0006/0007/0017-0019/0022, which cover the detector plus the analysis
stage below).

This is an independent reconstruction from that tutorial's published method
description, not a copy of Synaptosoft's source (which isn't available) -- a
handful of implementation details it doesn't specify (the baseline
statistic, the exact onset-crossing definition used for "time to peak", the
exact skewness/kurtosis estimator) are reasonable, clearly-flagged choices
below rather than verified originals. Everything the tutorial DOES specify
is implemented as described.

DETECTION -- sequence of peak detection (slide 0017):
    1. find a local maximum
    2. find a baseline
    3. compare amplitude to threshold
    4. compute time to peak -- if the trace never rises past onset_fraction
       within onset_search_ms before the peak, the candidate is REJECTED
       outright (same hard search-limit semantics as step 5's
       decay_search_ms, just on the rise side)
    5. compute time to decay -- if the trace never reaches decay_fraction
       (g) of the peak within decay_search_ms (f), the candidate is
       REJECTED outright (decay_search_ms is a hard search limit on
       whether an event counts at all, not just a display/measurement cap)
    6. compute area, compare to threshold

Detection parameters (slide 0019), all fields of DetectionParams:
    amplitude threshold (a), area threshold (b), peak direction,
    number of points to average peak, period to search local maximum (c),
    time before peak for baseline (d), period to average baseline (e),
    period to search decay time (f), fraction to find decay (g)

For closely-spaced/overlapping events, slide 0007 describes adjusting the
baseline by extrapolating the PREVIOUS event's decay as a single exponential
(Y = A*e^(-x/tau)) rather than trusting a baseline window that's still
contaminated by that decay -- see _overlap_adjusted_baseline. OFF by default
(--adjust-overlapping-baseline to opt in): baseline is otherwise ALWAYS the
plain (d)/(e) window average, exactly what those two named parameters
describe and nothing else.

ANALYSIS -- once events are detected, the tutorial describes a further
analysis stage built on two structures ("Analysis I: Grouping and Data
Array"):
    - Grouping: selecting a subset of detected events for focused analysis
      ("Different Ways to Group Events" -- by criteria search, random
      selection, or episode) -- see group_by_criteria / group_random.
    - Data Arrays: merging/combining event data for further analysis
      ("Data Arrays: Combining and Manipulating Data" -- inter-event
      intervals, +,-,x,/,sqr,sqrt) -- see events_frame / combine_data_arrays.
That feeds two further analyses:
    - Descriptive Analysis: column statistics (5 moments of a
      distribution), frequency/cumulative histograms, running average,
      auto-/cross-correlation histograms -- see column_statistics,
      frequency_histogram, cumulative_histogram, running_average,
      autocorrelation_histogram, cross_correlation_histogram.
    - Group Analysis: display of grouped traces (average/superimposed, raw
      or scaled) and single/double exponential decay fitting via a Simplex
      (Nelder-Mead) minimization of sum-of-squares over a selectable fit
      range -- see extract_event_traces, scale_traces, fit_exponential_decay.
The action-potential-waveform-analysis, amperometric-peak-analysis, and
random-walk-modeling items the tutorial also names are for entirely
different signal types (spiking traces, amperometry) outside this package's
scope (gap-free voltage-clamp sEPSC recordings) and aren't implemented.

This module is deliberately import-light: pure numpy/scipy/pandas, no
matplotlib and no Qt, so it can be imported from a script, a notebook, or
any of the three GUIs without dragging a plotting backend along.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats as spstats
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.signal import find_peaks

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

__all__ = [
    "DetectionParams", "Event", "detect_events",
    "events_frame", "group_by_criteria", "group_random", "combine_data_arrays",
    "ColumnStatistics", "column_statistics", "frequency_histogram", "cumulative_histogram",
    "running_average", "autocorrelation_histogram", "cross_correlation_histogram",
    "extract_event_traces", "scale_traces", "DecayFit", "fit_exponential_decay",
    "PARAM_FIELD_SPECS", "output_stem",
]


@dataclass(frozen=True)
class DetectionParams:
    """The 9 Mini-Analysis-style detection parameters (tutorial slide 0019).

    Defaults are a reasonable starting point for sEPSCs, not calibrated
    values -- the tutorial itself (slide 0018, "Optimizing Detection
    Parameters") is explicit that these are meant to be tuned per
    recording by inspecting detected vs. missed events, not left fixed.
    """
    amplitude_threshold: float = 5.0                          # (a) pA
    area_threshold: float = 10.0                                # (b) pA*ms
    direction: Literal["negative", "positive"] = "negative"
    n_avg_peak: int = 3            # points averaged to read the peak value
    search_local_max_ms: float = 3.0                              # (c)
    """Must be at least comparable to a real event's decay duration --
    too short (e.g. 1ms against a ~3ms decay tau) lets per-sample noise
    riding on a single event's own decaying tail masquerade as several
    additional "local maxima", fragmenting one event into several spurious
    detections. Tune this against your own events' typical decay time."""
    baseline_before_ms: float = 2.0                                # (d)
    baseline_avg_ms: float = 3.0                                    # (e)
    decay_search_ms: float = 20.0                                    # (f)
    decay_fraction: float = 0.5                                       # (g)
    onset_fraction: float = 0.1
    """Not one of the tutorial's 9 named parameters -- needed to define
    "time to peak" (step 4), which the tutorial doesn't give a formula
    for. Mirrors decay_fraction's role but on the rising side: the onset
    is where the trace first rises past this fraction of the peak
    amplitude (above baseline) and stays there."""
    onset_search_ms: float = 5.0
    """Also not one of the tutorial's 9 -- the rise-side counterpart of
    decay_search_ms (f): how far before the peak to search for the
    onset_fraction crossing. If the trace never rises past onset_fraction
    within this window, the candidate is REJECTED outright (same hard
    search-limit semantics as decay_search_ms), not just measured with a
    truncated/unreached onset."""
    adjust_overlapping_baseline: bool = False
    """Off by default: baseline is then ALWAYS the plain mean of the (d)/
    (e) window (peak_idx - before_n - avg_n : peak_idx - before_n), exactly
    the two named parameters describe -- nothing else. Set True to opt into
    slide 0007's extra step for closely-spaced events, which
    REPLACES that window average with an extrapolated value (Y=A*e^(-x/tau)
    from the previous event's own decay) when the window would otherwise
    still be contaminated by it -- i.e. a baseline computed a different way
    than (d)/(e) for those specific events, not a bug when it happens, but
    also not what (d)/(e) alone would produce, hence opt-in."""


@dataclass
class Event:
    peak_idx: int
    peak_time_s: float
    baseline: float
    amplitude: float
    rise_time_ms: float
    decay_time_ms: float
    area: float


def _samples(ms: float, dt: float) -> int:
    return max(1, int(round(ms / 1000.0 / dt)))


def _overlap_adjusted_baseline(resting_baseline: float, prev_root_event: Event,
                                prev_tau_s: float, peak_time_s: float) -> Optional[float]:
    """Slide 0007: predict what the trace would read at THIS candidate's
    own peak time if the previous (root) event's decay is still ongoing --
    Y = A*e^(-x/tau) extrapolated from that event's fitted single-
    exponential decay -- so amplitude can be measured against the
    instantaneous decay trajectory instead of a stale, contaminated
    baseline-window average.

    Deliberately evaluated at the CANDIDATE'S OWN peak time, not the
    baseline window's time: those differ by several ms whenever a decay is
    still in progress, and using the wrong one under-corrects baseline by
    nearly the full amplitude of the earlier event (the two times aren't
    interchangeable the way they are for a flat, unchanging baseline).

    Returns None (caller falls back to the flat window average) if the
    residual is negligible or `peak_time_s` doesn't come after the root
    event.
    """
    x_s = peak_time_s - prev_root_event.peak_time_s
    if x_s <= 0 or prev_tau_s <= 0:
        return None
    residual = prev_root_event.amplitude * float(np.exp(-x_s / prev_tau_s))
    if abs(residual) < 0.02 * abs(prev_root_event.amplitude):
        return None  # decayed away; nothing to correct for
    return resting_baseline + residual


def detect_events(t: np.ndarray, v: np.ndarray, dt: float,
                   params: DetectionParams = DetectionParams()) -> list[Event]:
    """Scan a raw trace for synaptic events using the Mini-Analysis-style
    6-step sequence (slide 0017). `t`/`v` are the full trace (seconds /
    trace units); `dt` is the sample interval in seconds.
    """
    pol = -1.0 if params.direction == "negative" else 1.0
    sv = pol * v

    search_n = _samples(params.search_local_max_ms, dt)
    before_n = _samples(params.baseline_before_ms, dt)
    avg_n = _samples(params.baseline_avg_ms, dt)
    decay_n = _samples(params.decay_search_ms, dt)
    onset_search_n = _samples(params.onset_search_ms, dt)
    n_avg_peak = max(1, params.n_avg_peak)

    # Step 1: local maxima at least search_n samples apart -- a point only
    # counts as a candidate peak if nothing taller exists within that
    # window (tutorial's "period to search local maximum").
    candidate_idxs, _ = find_peaks(sv, distance=search_n)

    events: list[Event] = []
    prev_root_event: Optional[Event] = None  # last ACCEPTED event, for extrapolation
    prev_tau_s: Optional[float] = None
    resting_baseline: Optional[float] = None  # last known uncontaminated baseline reading

    for peak_idx in candidate_idxs:
        peak_idx = int(peak_idx)
        peak_time_s = peak_idx * dt

        # Step 2: baseline, from a window ending `baseline_before_ms`
        # before the peak and spanning `baseline_avg_ms`.
        b0 = peak_idx - before_n - avg_n
        b1 = peak_idx - before_n
        if b0 < 0:
            continue
        flat_baseline = float(np.mean(v[b0:b1]))
        baseline = flat_baseline
        adjusted = None
        if params.adjust_overlapping_baseline and prev_root_event is not None and prev_tau_s is not None:
            adjusted = _overlap_adjusted_baseline(
                resting_baseline if resting_baseline is not None else flat_baseline,
                prev_root_event, prev_tau_s, peak_time_s)
        if adjusted is not None:
            baseline = adjusted
        else:
            resting_baseline = flat_baseline  # this window wasn't contaminated; a clean reading

        # Step 3: amplitude vs. threshold.
        p0 = max(0, peak_idx - n_avg_peak // 2)
        p1 = min(len(v), p0 + n_avg_peak)
        peak_v = float(np.mean(v[p0:p1]))
        amplitude = peak_v - baseline
        if abs(amplitude) < params.amplitude_threshold:
            continue

        s_baseline, s_peak = pol * baseline, pol * peak_v
        span = s_peak - s_baseline
        if span <= 0:
            continue  # degenerate; can happen if baseline adjustment overshoots

        # Step 4: time to peak -- walk backward from the peak, within
        # onset_search_ms, to where the trace first rises past
        # onset_fraction of the amplitude and stays above it. If it never
        # gets there within the window, this candidate is rejected outright
        # (onset_search_ms is a hard search limit, same as decay_search_ms
        # (f) on the decay side -- not just a display/measurement cap).
        onset_level = s_baseline + params.onset_fraction * span
        onset_search_start = max(0, peak_idx - onset_search_n)
        onset_idx = None
        for k in range(peak_idx, onset_search_start - 1, -1):
            if sv[k] < onset_level:
                onset_idx = k
                break
        if onset_idx is None:
            continue
        rise_time_ms = (peak_idx - onset_idx) * dt * 1e3

        # Step 5: time to decay -- walk forward within decay_search_ms for
        # where the trace decays to decay_fraction of the peak amplitude.
        # If it never gets there within the window, this candidate is
        # rejected outright (not counted as an event with a truncated/
        # unreached decay point) -- decay_search_ms (f) is a hard search
        # limit, not a display cap.
        decay_level = s_baseline + (1.0 - params.decay_fraction) * span
        search_end = min(len(sv), peak_idx + decay_n)
        decay_idx = None
        for k in range(peak_idx, search_end):
            if sv[k] <= decay_level:
                decay_idx = k
                break
        if decay_idx is None:
            continue
        decay_time_ms = (decay_idx - peak_idx) * dt * 1e3

        # Step 6: area (baseline-subtracted, onset to decay point) vs.
        # threshold.
        area = float(_trapz(v[onset_idx:decay_idx + 1] - baseline, dx=dt * 1e3))
        if abs(area) < params.area_threshold:
            continue

        event = Event(
            peak_idx=peak_idx, peak_time_s=float(peak_idx * dt),
            baseline=baseline, amplitude=amplitude,
            rise_time_ms=rise_time_ms, decay_time_ms=decay_time_ms, area=area,
        )
        events.append(event)

        # Fit this event's own decay as a single exponential (slide 0007's
        # Y = A*e^(-x/tau)) so the NEXT candidate can extrapolate it if
        # it's close enough to still be riding this one's tail. tau is
        # solved analytically from decay_fraction/decay_time_ms, which are
        # already exactly known -- no extra sampling needed.
        #
        # Only do this when THIS event's own baseline was clean (adjusted
        # is None): an event accepted while still riding a previous decay
        # has its own decay_time_ms measured on a signal that's the SUM of
        # both events' decays, not a clean read of its own kinetics alone,
        # so fitting tau from it and chaining forward would compound the
        # error into every subsequent candidate. Keep extrapolating from
        # the last genuinely clean event instead -- a documented
        # simplification for 3+ closely chained/overlapping events (this
        # single-exponential correction is a first-order approximation per
        # the tutorial's own description, not full multi-event
        # deconvolution).
        if adjusted is None:
            if decay_time_ms > 0 and 0.0 < params.decay_fraction < 1.0:
                prev_tau_s = -(decay_time_ms / 1000.0) / np.log(1.0 - params.decay_fraction)
            else:
                prev_tau_s = None
            prev_root_event = event

    return events


# ---------------------------------------------------------------------------
# Analysis I: Grouping and Data Arrays (tutorial: "Analysis I")
# ---------------------------------------------------------------------------

def events_frame(events: list[Event]) -> pd.DataFrame:
    """Event list -> tidy DataFrame, sorted by peak time, with an
    inter-event interval column added (p.9 "Data Arrays": "By calculating
    inter-event intervals"). The first event's interval is NaN -- there's
    no preceding event to measure it from."""
    df = pd.DataFrame([vars(e) for e in events])
    if df.empty:
        df["inter_event_interval_ms"] = pd.Series(dtype=float)
        return df
    df = df.sort_values("peak_time_s").reset_index(drop=True)
    df["inter_event_interval_ms"] = df["peak_time_s"].diff() * 1e3
    return df


def group_by_criteria(df: pd.DataFrame, **ranges: tuple[Optional[float], Optional[float]]) -> np.ndarray:
    """Boolean mask selecting rows whose columns fall within given
    (min, max) ranges, e.g. group_by_criteria(df, amplitude=(10, 50),
    decay_time_ms=(0, 15)) -- p.8 "Different Ways to Group Events": "Group
    by Criteria Search". Either bound may be None for an open end."""
    mask = np.ones(len(df), dtype=bool)
    for column, (lo, hi) in ranges.items():
        values = df[column].to_numpy(dtype=float)
        if lo is not None:
            mask &= values >= lo
        if hi is not None:
            mask &= values <= hi
    return mask


def group_random(n_total: int, n_select: int, seed: Optional[int] = None) -> np.ndarray:
    """Boolean mask selecting a random subset of n_total rows -- p.8
    "Different Ways to Group Events": "Group by Random Selection"."""
    rng = np.random.default_rng(seed)
    n_select = max(0, min(n_select, n_total))
    idx = rng.choice(n_total, size=n_select, replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[idx] = True
    return mask


def combine_data_arrays(a, b, op: Literal["+", "-", "x", "/", "sqr", "sqrt"]) -> np.ndarray:
    """The mathematical operations p.9 "Data Arrays" lists for combining/
    manipulating data arrays: (+,-,x,/,sqr,sqrt). `b` is ignored for
    sqr/sqrt (unary)."""
    a = np.asarray(a, dtype=float)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "x":
        return a * b
    if op == "/":
        return a / b
    if op == "sqr":
        return a ** 2
    if op == "sqrt":
        return np.sqrt(a)
    raise ValueError(f"unknown operation {op!r} (expected one of +,-,x,/,sqr,sqrt)")


# ---------------------------------------------------------------------------
# Analysis II: Descriptive Analysis (tutorial: "Descriptive Analysis")
# ---------------------------------------------------------------------------

@dataclass
class ColumnStatistics:
    """"Column Statistics (5 moments of a distribution: mean, variance,
    etc.)" (p.10) -- n plus the 5 moments: mean, variance, sd, skewness,
    kurtosis (excess, i.e. 0 for a normal distribution)."""
    n: int
    mean: float
    variance: float
    sd: float
    skewness: float
    kurtosis: float


def column_statistics(values) -> ColumnStatistics:
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return ColumnStatistics(0, np.nan, np.nan, np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    variance = float(np.var(x, ddof=1)) if n > 1 else 0.0
    sd = float(np.sqrt(variance))
    skewness = float(spstats.skew(x, bias=False)) if n > 2 else np.nan
    kurtosis = float(spstats.kurtosis(x, bias=False)) if n > 3 else np.nan
    return ColumnStatistics(n, mean, variance, sd, skewness, kurtosis)


def frequency_histogram(values, bin_size: Optional[float] = None) -> pd.DataFrame:
    """Frequency histogram (p.10): counts per bin_size-wide bin over the
    data's own range. bin_size=None auto-picks range/30 (or 1.0 if the
    data is degenerate/empty)."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return pd.DataFrame({"bin_start": [], "bin_center": [], "count": []})
    data_range = float(x.max() - x.min())
    if bin_size is None:
        bin_size = data_range / 30.0 if data_range > 0 else 1.0
    lo = np.floor(x.min() / bin_size) * bin_size
    hi = np.ceil(x.max() / bin_size) * bin_size + bin_size
    edges = np.arange(lo, hi, bin_size)
    counts, edges = np.histogram(x, bins=edges)
    return pd.DataFrame({"bin_start": edges[:-1], "bin_center": edges[:-1] + bin_size / 2, "count": counts})


def cumulative_histogram(values, bin_size: Optional[float] = None) -> pd.DataFrame:
    """Cumulative fraction histogram (p.10)."""
    hist = frequency_histogram(values, bin_size)
    hist = hist.copy()
    total = hist["count"].sum()
    hist["cumulative_fraction"] = hist["count"].cumsum() / total if total > 0 else hist["count"].astype(float)
    return hist


def running_average(values, window: int) -> np.ndarray:
    """Running average histogram (p.10)."""
    x = np.asarray(values, dtype=float)
    return uniform_filter1d(x, size=max(1, window), mode="nearest")


def autocorrelation_histogram(event_times_s, max_lag_ms: float, bin_ms: float) -> pd.DataFrame:
    """Auto-correlation histogram for periodicity (p.10): histogram of
    every pairwise time difference between events (both signs) out to
    +/-max_lag_ms."""
    t = np.sort(np.asarray(event_times_s, dtype=float)) * 1e3  # ms
    diffs: list[float] = []
    for i in range(len(t)):
        j = i + 1
        while j < len(t) and t[j] - t[i] <= max_lag_ms:
            d = t[j] - t[i]
            diffs.append(d)
            diffs.append(-d)
            j += 1
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(diffs, bins=edges)
    return pd.DataFrame({"lag_start_ms": edges[:-1], "lag_center_ms": edges[:-1] + bin_ms / 2, "count": counts})


def cross_correlation_histogram(times_a_s, times_b_s, max_lag_ms: float, bin_ms: float) -> pd.DataFrame:
    """Cross-correlation histogram for synaptic connection between a pair
    (p.10): histogram of every t_b - t_a pairwise time difference (events B
    relative to events A) out to +/-max_lag_ms."""
    a = np.asarray(times_a_s, dtype=float) * 1e3
    b = np.sort(np.asarray(times_b_s, dtype=float)) * 1e3
    diffs: list[np.ndarray] = []
    for ta in a:
        i0, i1 = np.searchsorted(b, [ta - max_lag_ms, ta + max_lag_ms])
        diffs.append(b[i0:i1] - ta)
    all_diffs = np.concatenate(diffs) if diffs else np.array([])
    edges = np.arange(-max_lag_ms, max_lag_ms + bin_ms, bin_ms)
    counts, edges = np.histogram(all_diffs, bins=edges)
    return pd.DataFrame({"lag_start_ms": edges[:-1], "lag_center_ms": edges[:-1] + bin_ms / 2, "count": counts})


# ---------------------------------------------------------------------------
# Analysis III: Group Analysis -- display of grouped traces (tutorial: "Analysis III")
# ---------------------------------------------------------------------------

def extract_event_traces(v: np.ndarray, dt: float, events: list[Event], pre_ms: float, post_ms: float):
    """Peak-centered raw windows for a group of events -- p.11 "Analysis
    III: Group Analysis": "Display of grouped traces (raw or scaled)".
    Returns (t_ms, traces, baselines, amplitudes); events too close to
    either recording edge are silently dropped, so the returned arrays may
    have fewer rows than `events`. baselines/amplitudes come from each
    Event's own already-computed detection values, not re-derived here."""
    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    t_ms = np.arange(-pre_n, post_n) * dt * 1e3
    rows, baselines, amplitudes = [], [], []
    for e in events:
        i0, i1 = e.peak_idx - pre_n, e.peak_idx + post_n
        if i0 < 0 or i1 > len(v):
            continue
        rows.append(v[i0:i1])
        baselines.append(e.baseline)
        amplitudes.append(e.amplitude)
    traces = np.array(rows) if rows else np.empty((0, pre_n + post_n))
    return t_ms, traces, np.array(baselines), np.array(amplitudes)


def scale_traces(traces: np.ndarray, baselines: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Baseline-subtract each trace and divide by its OWN detected
    amplitude, so events of different sizes overlay at unit peak
    deflection -- p.12 "Display of Grouped Traces": "Scaled Superimposed" /
    "Scaled Averaged"."""
    safe_amp = np.where(amplitudes == 0, 1.0, amplitudes)
    return (traces - baselines[:, None]) / safe_amp[:, None]


# ---------------------------------------------------------------------------
# Exponential Decay Fitting (tutorial): Simplex (Nelder-Mead) minimization
# of sum-of-squares, single or double exponential, selectable fit range.
# ---------------------------------------------------------------------------

@dataclass
class DecayFit:
    n_exp: int
    amplitudes: list  # [A1] or [A1, A2]
    taus_ms: list      # [tau1] or [tau1, tau2], ms
    offset: float       # C, steady-state asymptote
    residual_sd: float
    n_iterations: int
    fit_range_ms: tuple  # (start_ms, end_ms), relative to the trace's own t_ms[0]

    def equation_str(self) -> str:
        terms = "+".join(f"({a:.4g})*exp(-t/{tau:.4g}ms)" for a, tau in zip(self.amplitudes, self.taus_ms))
        return f"y={terms}+({self.offset:.4g})"

    def predict(self, t_ms: np.ndarray) -> np.ndarray:
        y = np.full_like(np.asarray(t_ms, dtype=float), self.offset, dtype=float)
        for a, tau in zip(self.amplitudes, self.taus_ms):
            y = y + a * np.exp(-np.asarray(t_ms, dtype=float) / tau)
        return y


def _decay_fit_bounds(t_ms: np.ndarray, y: np.ndarray, mode: str,
                       custom_start_ms: Optional[float] = None, custom_end_ms: Optional[float] = None,
                       direction: str = "negative") -> tuple[int, int]:
    """Resolve a named fit range to (start_idx, end_idx) into t_ms/y, where
    t_ms[0] is the peak -- p.13 "Flexible fitting range": "Peak to end /
    Decay 10-90 / Decay 20-80 / Custom range"."""
    pol = -1.0 if direction == "negative" else 1.0
    sy = pol * y
    speak = float(sy[0]) if len(sy) else 0.0

    if mode == "peak_to_end":
        return 0, len(t_ms) - 1
    if mode == "custom":
        if custom_start_ms is None or custom_end_ms is None:
            raise ValueError("fit_range='custom' requires custom_start_ms and custom_end_ms")
        i0 = int(np.searchsorted(t_ms, custom_start_ms))
        i1 = int(np.searchsorted(t_ms, custom_end_ms))
        i0 = max(0, min(i0, len(t_ms) - 2))
        i1 = max(i0 + 1, min(i1, len(t_ms) - 1))
        return i0, i1
    if mode in ("decay_10_90", "decay_20_80"):
        hi_frac, lo_frac = (0.9, 0.1) if mode == "decay_10_90" else (0.8, 0.2)
        hi_level, lo_level = hi_frac * speak, lo_frac * speak
        i0 = i1 = None
        for k in range(len(sy)):
            if i0 is None and sy[k] <= hi_level:
                i0 = k
            if i0 is not None and sy[k] <= lo_level:
                i1 = k
                break
        i0 = 0 if i0 is None else i0
        i1 = (len(t_ms) - 1) if i1 is None else i1
        return i0, max(i0 + 1, i1)
    raise ValueError(f"unknown fit_range {mode!r}")


def fit_exponential_decay(t_ms: np.ndarray, y: np.ndarray, n_exp: int = 1,
                           fit_range: Literal["peak_to_end", "decay_10_90", "decay_20_80", "custom"] = "peak_to_end",
                           custom_start_ms: Optional[float] = None, custom_end_ms: Optional[float] = None,
                           direction: Literal["negative", "positive"] = "negative",
                           p0: Optional["DecayFit"] = None) -> DecayFit:
    """Fit y(t) ~= sum_i A_i*exp(-t/tau_i) + C to the peak-to-tail portion
    of one (typically averaged) event trace -- p.13 "Exponential Decay
    Fitting": "Using Simplex fitting algorithm. Minimization of
    sum-of-squares", "Single and double exponential fitting", "Flexible
    fitting range", and "Ability to use last fit results as initial
    guesses" (pass a previous DecayFit as `p0`).

    `t_ms`/`y` must be peak-aligned (t_ms[0] == 0, e.g. from
    extract_event_traces's averaged output). Uses
    scipy.optimize.minimize(method="Nelder-Mead") -- the Nelder-Mead
    downhill Simplex the tutorial names -- on raw sum-of-squared residuals,
    not a gradient-based least-squares solver.
    """
    if n_exp not in (1, 2):
        raise ValueError("n_exp must be 1 or 2")
    i0, i1 = _decay_fit_bounds(t_ms, y, fit_range, custom_start_ms, custom_end_ms, direction)
    tt = np.asarray(t_ms[i0:i1 + 1], dtype=float) - t_ms[i0]
    yy = np.asarray(y[i0:i1 + 1], dtype=float)

    peak_amp = float(yy[0])
    tail = float(np.median(yy[-max(1, len(yy) // 10):]))
    span = max(tt[-1], 1e-3)

    if p0 is not None and p0.n_exp == n_exp:
        x0 = np.array([*p0.amplitudes, *p0.taus_ms, p0.offset], dtype=float)
    elif n_exp == 1:
        x0 = np.array([peak_amp - tail, span / 3.0, tail], dtype=float)
    else:
        x0 = np.array([0.7 * (peak_amp - tail), span / 6.0, 0.3 * (peak_amp - tail), span / 2.0, tail], dtype=float)

    def unpack(x):
        amps = x[0:n_exp]
        taus = np.abs(x[n_exp:2 * n_exp]) + 1e-6
        offset = x[-1]
        return amps, taus, offset

    def sse(x):
        amps, taus, offset = unpack(x)
        pred = np.full_like(tt, offset, dtype=float)
        for a, tau in zip(amps, taus):
            pred = pred + a * np.exp(-tt / tau)
        return float(np.sum((yy - pred) ** 2))

    result = minimize(sse, x0, method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 5000, "maxfev": 5000})
    amps, taus, offset = unpack(result.x)
    n, k = len(yy), 2 * n_exp + 1
    residual_sd = float(np.sqrt(result.fun / max(1, n - k)))
    return DecayFit(n_exp=n_exp, amplitudes=amps.tolist(), taus_ms=taus.tolist(), offset=float(offset),
                     residual_sd=residual_sd, n_iterations=int(result.nit),
                     fit_range_ms=(float(t_ms[i0]), float(t_ms[i1])))


# ---------------------------------------------------------------------------
# One shared description of DetectionParams' fields as editable dialog
# fields -- (attr, label, widget kind, kwargs for the spin box) -- so the
# parameter dialog in run.py and the live parameter panel in optimize.py
# always offer exactly the same fields with the same ranges/steps. Kept
# here, next to DetectionParams itself, so adding a parameter is a
# one-file change; no Qt is imported to define it (see gui_utils.
# make_field_widget, which turns one of these rows into a widget).
# ---------------------------------------------------------------------------

PARAM_FIELD_SPECS = [
    # (attr, label, widget kind, kwargs for the spin box)
    ("amplitude_threshold", "Amplitude threshold (a)", "double", dict(minimum=0.0, maximum=1e6, decimals=2, singleStep=0.5)),
    ("area_threshold", "Area threshold (b)", "double", dict(minimum=0.0, maximum=1e7, decimals=2, singleStep=0.5)),
    ("direction", "Peak direction", "choice", dict(choices=["negative", "positive"])),
    ("n_avg_peak", "Number of points to average peak", "int", dict(minimum=1, maximum=1000)),
    ("search_local_max_ms", "Period to search local maximum (c), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("baseline_before_ms", "Time before peak for baseline (d), ms", "double", dict(minimum=0.0, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("baseline_avg_ms", "Period to average baseline (e), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("decay_search_ms", "Period to search decay time (f), ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=1.0)),
    ("decay_fraction", "Fraction to find decay (g)", "double", dict(minimum=0.01, maximum=0.99, decimals=2, singleStep=0.05)),
    ("onset_fraction", "Onset fraction (rise crossing)", "double", dict(minimum=0.01, maximum=0.99, decimals=2, singleStep=0.05)),
    ("onset_search_ms", "Period to search onset before peak, ms", "double", dict(minimum=0.01, maximum=100000.0, decimals=2, singleStep=0.5)),
    ("adjust_overlapping_baseline", "Adjust baseline for overlapping events", "bool", dict()),
]


# ---------------------------------------------------------------------------
# Output naming, in one place. run.py writes the events CSV/params sidecar/
# plots, optimize.py's "Run full detection" button writes the first two, and
# check.py has to find all of them again later -- so the rule for turning a
# recording into a filename prefix lives here, where all three can import it
# without pulling in matplotlib or Qt.
# ---------------------------------------------------------------------------

def output_stem(abf_path: str, channel: int = 0, filter_enabled: bool = False,
                 cutoff_hz: float = 0.0, target_rate_hz: float = 0.0) -> str:
    """The prefix every output file for this run hangs off: the .abf path
    without its extension, plus what makes this run distinguishable from
    another on the same recording.

    Channel is included only when it isn't 0. That asymmetry is deliberate:
    channel 0 is the overwhelmingly common case and its filenames predate
    this, so leaving them unsuffixed keeps every already-analysed recording
    findable -- while a second channel no longer silently overwrites the
    first. On a dual-channel recording (a primary current channel plus a
    secondary voltage monitor, say) analysing channel 1 used to clobber
    channel 0's events CSV with no warning at all.

    Filter settings are included when filtering is on, because event
    positions are sample indices into the filtered/resampled trace: a
    filtered run and a raw run of the same recording describe different
    traces and must not share an output file.

    Pure string logic, no file I/O, so callers can build paths (and fail
    fast on a missing input) before doing any expensive loading.
    """
    stem = os.path.splitext(abf_path)[0]
    if channel:
        stem += f"_ch{channel}"
    if filter_enabled:
        stem += f"_filt{int(cutoff_hz)}Hz{int(target_rate_hz)}Hz"
    return stem

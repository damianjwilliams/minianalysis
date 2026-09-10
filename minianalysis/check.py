"""
Check the events a detection run produced: click any detected event and
see, drawn directly on that event's own trace, every window and threshold
the 6-step detection sequence actually used to accept it -- the Mini
Analysis Program tutorial's own "Optimizing Detection Parameters" workflow
("First use mouse-click to detect... Examine the location of the X's and
dots... Examine the amplitude and area") applied to this reimplementation.

Shown both as text and as annotated spans/lines on the trace: all 8 of the
tutorial's named detection-window parameters plus the rise-side onset
search window (this reimplementation's own addition, mirroring (f) on the
decay side) -- Amplitude threshold (a), Area threshold (b), Number of
points to average peak, Period to search local maximum (c), Time before
peak for baseline (d), Period to average baseline (e), Period to search
decay time (f), Fraction to find decay (g), Period to search onset before
peak.

The parameters shown are read from the `<stem>_minianalysis_params.json`
sidecar that run.py (and optimize.py's "Run full detection" button) saves
alongside every events CSV -- so what you see here is guaranteed to be what
actually produced those events, not just today's defaults. If no sidecar is
found (events from an older run), it falls back to DetectionParams()'s
defaults, overridable with the same --amplitude-threshold/--area-threshold/
etc. flags run.py itself takes.

Also a QC/curation tool: Accept/Reject each event (buttons or keys), or
Accept All Remaining in one click. Two files are autosaved after EVERY
decision, so nothing is ever lost, and either one is picked back up
automatically on the next run:

    <stem>_minianalysis_reviewed.csv          accepted events only
    <stem>_minianalysis_review_progress.csv   every decided event, with a
                                              `decision` column (1=accepted,
                                              0=rejected)

Events you haven't decided on yet simply don't appear in the progress file;
they aren't "rejected" by default.

Usage
-----
    python check_events.py path\\to\\recording.abf
    python -m minianalysis check path\\to\\recording.abf      # same thing
    python -m minianalysis check recording.abf --csv custom_minianalysis_events.csv
    python -m minianalysis check recording.abf --amplitude-threshold 8   # override one param
    python -m minianalysis check recording.abf --filter
        # ^ if the detection run also used --filter (3 kHz Bessel low-pass + 10 kHz resample by
        # default -- see preprocess.py). This MUST match whatever --filter settings that run used:
        # peak_idx values in the events CSV are indices into that filtered/resampled trace, not
        # the raw one, so a mismatch silently misaligns every event.

Controls:
    click a red X in the top overview -- inspect that event
    Right / N            -- next event
    Left / P              -- previous event
    A / Up                -- accept this event, then advance
    R / Down              -- reject this event, then advance
    "Accept All Remaining" button -- accept every still-undecided event
    scroll wheel over the top (full-trace) panel -- zoom in/out, centered
        on the cursor; time (X) and current (Y) together by default, hold
        Ctrl to zoom time only or Shift to zoom current only
    left-drag inside the top panel -- pan (both X and Y together); no
        toolbar tool needs to be selected first, and a plain click (no
        drag) still selects the nearest event as usual
    toolbar's Pan/Zoom-rectangle buttons -- optional, for the toolbar's own
        constrained/box-zoom behavior on the top panel (hold x or y while
        dragging to pan or zoom just one axis); while either is toggled on
        it takes over the drag instead of the always-on pan above. Home
        resets to the original view; Back/Forward step through previous
        views.
    The bottom (per-event detail) panel has no independent pan/zoom of its
        own -- it's always redrawn fresh, auto-fit around the selected peak.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import numpy as np
import pandas as pd
import pyabf

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets

from .core import DetectionParams, _samples, output_stem
from .gui_utils import safe_callback
from .preprocess import (
    DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, get_hardware_filter_hz,
    load_filtered_trace,
)
from .style import GRID, INK, MUTED, SURFACE, TRACE

# The output-file naming convention, in one place -- run.py and
# optimize.py's "Run full detection" button both write EVENTS_SUFFIX/
# PARAMS_SUFFIX, and this module both reads those and writes the two QC
# files beside them. All five hang off the same <stem> (the .abf path
# without its extension, plus a _filt<cutoff>Hz<rate>Hz suffix when the run
# was filtered, and _ch<n> for a channel other than 0 -- see
# core.output_stem).
EVENTS_SUFFIX = "_minianalysis_events.csv"
PARAMS_SUFFIX = "_minianalysis_params.json"
REVIEWED_SUFFIX = "_minianalysis_reviewed.csv"
PROGRESS_SUFFIX = "_minianalysis_review_progress.csv"

# Which detector's output to review (--source). Each entry is the
# (events, params, reviewed, progress) suffixes, all hanging off the same
# core.output_stem, so every detector gets its own QC files and none can
# clobber another's. The window itself is detector-agnostic: it redraws each
# event's baseline/onset/decay windows from the CSV's own baseline+amplitude
# columns via _event_geometry, which is why one reviewer serves both.
SOURCES = {
    "minianalysis": (EVENTS_SUFFIX, PARAMS_SUFFIX, REVIEWED_SUFFIX, PROGRESS_SUFFIX),
    "deconv": ("_deconv_events.csv", "_deconv_params.json",
               "_deconv_reviewed.csv", "_deconv_review_progress.csv"),
}
SOURCE_LABELS = {
    "minianalysis": "Mini Analysis event checker",
    "deconv": "Deconvolution event checker",
}

# Built directly on a PyQt5 QApplication/QMainWindow via matplotlib's Qt
# Figure/FigureCanvas, NOT pyplot's own mainloop: pyplot's backend is a
# single global, and run.py deliberately forces the non-interactive "Agg"
# backend for its headless CLI plotting. Building the Qt window explicitly
# means importing anything from this package can never downgrade this
# window's backend, regardless of import order.

OVERVIEW_MAX_POINTS = 200_000  # stride-decimated for display speed only; detail view always uses full-res data


def _load_params(params_json: str | None, stem: str, overrides: dict,
                  params_suffix: str = PARAMS_SUFFIX) -> DetectionParams:
    """Sidecar JSON (if present) as the base, with any explicitly-passed
    CLI flags (non-None in `overrides`) applied on top -- see module
    docstring.

    Unknown keys are IGNORED rather than fatal. A detector other than the
    classical one writes its own parameters plus run diagnostics into its
    sidecar, and only the measurement fields (direction, n_avg_peak, the
    baseline/onset/decay windows and fractions) are shared with
    DetectionParams -- those are the only ones this window draws from.
    Anything DetectionParams does not define falls back to its default.
    """
    path = params_json or f"{stem}{params_suffix}"
    if os.path.exists(path):
        with open(path) as fh:
            raw = json.load(fh)
        known = {f.name for f in dataclasses.fields(DetectionParams)}
        ignored = sorted(set(raw) - known)
        base = DetectionParams(**{k: val for k, val in raw.items() if k in known})
        print(f"Loaded detection parameters -> {path}")
        if ignored:
            shown = ", ".join(ignored[:6]) + (", ..." if len(ignored) > 6 else "")
            print(f"  ({len(ignored)} key(s) there are not detection parameters "
                  f"and were ignored: {shown})")
    else:
        base = DetectionParams()
        print(f"No params sidecar found ({path}) -- using DetectionParams() defaults "
              f"(pass --amplitude-threshold etc. to override, or re-run "
              f"`python -m minianalysis run` to regenerate the sidecar).")
    explicit = {k: v for k, v in overrides.items() if v is not None}
    return dataclasses.replace(base, **explicit) if explicit else base


def _event_geometry(v: np.ndarray, dt: float, peak_idx: int, baseline: float, amplitude: float,
                     params: DetectionParams) -> dict:
    """Recompute the index boundaries detect_events' steps 2/4/5 used for
    this peak (b0/b1 baseline window, p0/p1 peak-average window, onset_idx,
    decay_idx, and an illustrative local-max search span around the peak).

    Deliberately takes `baseline`/`amplitude` from the event's own CSV row
    rather than recomputing them: those are the exact (possibly
    overlap-adjusted, see core._overlap_adjusted_baseline) values actually
    used at detection time, and re-deriving them here without that same
    sequential state could silently disagree with what really produced this
    event. Only the WINDOW GEOMETRY is recomputed -- purely a function of
    peak_idx/params, safe to redo standalone.
    """
    pol = -1.0 if params.direction == "negative" else 1.0
    before_n = _samples(params.baseline_before_ms, dt)
    avg_n = _samples(params.baseline_avg_ms, dt)
    decay_n = _samples(params.decay_search_ms, dt)
    onset_search_n = _samples(params.onset_search_ms, dt)
    search_n = _samples(params.search_local_max_ms, dt)
    n_avg_peak = max(1, params.n_avg_peak)

    b0 = max(0, peak_idx - before_n - avg_n)
    b1 = max(0, peak_idx - before_n)
    p0 = max(0, peak_idx - n_avg_peak // 2)
    p1 = min(len(v), p0 + n_avg_peak)

    # span == pol * amplitude EXACTLY, in both directions -- written that
    # way rather than as the more obvious pol * (baseline + amplitude) -
    # pol * baseline. detect_events computes span as (pol * peak_v) -
    # (pol * baseline) and amplitude as peak_v - baseline; multiplying by
    # +-1.0 is exact and IEEE subtraction is antisymmetric, so this form
    # agrees with it bit for bit, while rebuilding peak_v from baseline +
    # amplitude rounds once more than it has to. The onset/decay levels
    # below are fractions of this span, and a level an ULP away from a
    # sample sitting right at it picks a different crossing sample -- so
    # this is the cheap way to guarantee the markers land where the
    # detector actually measured, rather than merely usually landing there.
    s_baseline = pol * baseline
    span = pol * amplitude
    sv = pol * v

    # Mirrors detect_events' own onset search exactly (peak_idx -
    # onset_search_n : peak_idx), NOT capped at b1 -- an accepted event is
    # guaranteed to have a real crossing in this window, so the fallback
    # (onset_search_start) only matters for degenerate/edge cases.
    onset_level = s_baseline + params.onset_fraction * span
    onset_search_start = max(0, peak_idx - onset_search_n)
    onset_idx = onset_search_start
    for k in range(peak_idx, onset_search_start - 1, -1):
        if sv[k] < onset_level:
            onset_idx = k
            break

    decay_level = s_baseline + (1.0 - params.decay_fraction) * span
    decay_search_end = min(len(sv), peak_idx + decay_n)
    decay_idx = decay_search_end - 1
    for k in range(peak_idx, decay_search_end):
        if sv[k] <= decay_level:
            decay_idx = k
            break

    c_half = max(1, search_n // 2)
    return dict(b0=b0, b1=b1, p0=p0, p1=p1, onset_idx=onset_idx, decay_idx=decay_idx,
                decay_search_end=decay_search_end, onset_search_start=onset_search_start,
                local_max_lo=max(0, peak_idx - c_half), local_max_hi=min(len(v), peak_idx + c_half))


PARAM_LINES = [
    ("Amplitude threshold (a)", "amplitude_threshold", ""),
    ("Area threshold (b)", "area_threshold", ""),
    ("Number of points to average peak", "n_avg_peak", ""),
    ("Period to search local maximum (c)", "search_local_max_ms", " ms"),
    ("Time before peak for baseline (d)", "baseline_before_ms", " ms"),
    ("Period to average baseline (e)", "baseline_avg_ms", " ms"),
    ("Period to search decay time (f)", "decay_search_ms", " ms"),
    ("Fraction to find decay (g)", "decay_fraction", ""),
    ("Period to search onset before peak", "onset_search_ms", " ms"),
]


# The classical detector's accept/reject criteria. A different detector does
# not apply these, so the panel must not claim it did -- see params_text.
CLASSICAL_ONLY_FIELDS = {"amplitude_threshold", "area_threshold", "search_local_max_ms"}


def params_text(params: DetectionParams, show_thresholds: bool = True,
                extra: str = "") -> str:
    """The parameter panel's text.

    `show_thresholds=False` drops the lines the classical detector uses to
    ACCEPT or REJECT a candidate -- amplitude (a), area (b) and the local-max
    search span (c). Those fields still exist on DetectionParams and still
    hold their defaults, but a detector that never consulted them would be
    misrepresented by a panel listing them: an operator reading
    "Amplitude threshold (a): 5.0" would reasonably conclude every event
    shown had cleared 5 pA.  The measurement windows (d, e, f, g, onset) are
    shared by both detectors, so those are always shown.

    `extra` is appended verbatim, for a source to state its own criteria.
    """
    shown = [(label, field, unit) for label, field, unit in PARAM_LINES
             if show_thresholds or field not in CLASSICAL_ONLY_FIELDS]
    text = "\n".join(f"{label}: {getattr(params, field)}{unit}"
                     for label, field, unit in shown)
    return f"{text}\n{extra}" if extra else text


# Colors used consistently between plot_event_detail's annotations and
# build_legend_handles' key, so the two never drift apart.
COLOR_BASELINE_WINDOW = "#7fbf7f"
COLOR_BEFORE_PEAK = "#c9c9c9"
COLOR_BASELINE_LINE = "#2a8f2a"
COLOR_LOCAL_MAX = "#8a6dc9"
COLOR_PEAK = "crimson"
COLOR_AMPLITUDE = "#c97a2a"
COLOR_DECAY_WINDOW = "#f0c674"
COLOR_DECAY_LEVEL = "#b8860b"
COLOR_ONSET = "#2a6dc9"
COLOR_ONSET_WINDOW = "#9fc9e8"
COLOR_AREA_PASS = "#4a90d9"
COLOR_AREA_FAIL = "#e0555a"

# Light, high-contrast backing so inline trace labels stay legible over the
# raw signal and colored spans, instead of just floating text.
_LABEL_BBOX = dict(boxstyle="round,pad=0.2", fc=SURFACE, ec=MUTED, lw=0.5, alpha=0.9)

# QC accept/reject color language -- shared between the overview's halo
# markers (accepted_overlay/rejected_overlay below) and the Accept/Reject
# buttons' own stylesheets (see EventInspector.__init__), so a decision
# reads the same way in both places. Kept as its own pair, separate from
# the per-event annotation colors above, even though COLOR_ACCEPTED happens
# to equal COLOR_BASELINE_LINE -- QC state and detection-window annotations
# are conceptually unrelated. Deliberately NOT reusing COLOR_PEAK (crimson,
# already the plain "undecided" candidate marker's own color) for rejected
# -- a rejected halo needs to read as "reject" without being confusable
# with an undecided marker.
COLOR_ACCEPTED = "#2a8f2a"
COLOR_REJECTED = "#c0392b"


def build_legend_handles(show_thresholds: bool = True):
    """Proxy artists explaining every annotation plot_event_detail draws --
    the 'key' shown once in the inspector window, built here so its colors
    can never drift out of sync with the trace annotations themselves.

    `show_thresholds=False` drops the three entries that describe classical
    accept/reject machinery -- the (c) local-max span, the (a) amplitude
    threshold, and the (b) area PASS/FAIL fills. plot_event_detail does not
    draw those for a detector that never applied them, and a key listing
    annotations absent from the plot is worse than no key.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    items = [
        (Patch(fc=COLOR_BASELINE_WINDOW, alpha=0.35), "Baseline averaging window (e)"),
        (Line2D([], [], color=COLOR_BASELINE_LINE, ls="--", lw=1.2), "Baseline level used"),
        (Patch(fc=COLOR_BEFORE_PEAK, alpha=0.35), "Time before peak for baseline (d)"),
    ]
    if show_thresholds:
        items.append((Line2D([], [], color=COLOR_LOCAL_MAX, lw=1.2, marker=">", ms=5),
                      "Local-max search span (c)"))
    items += [
        (Line2D([], [], color=COLOR_ONSET_WINDOW, ls="--", lw=1.4), "Onset search window boundary"),
        (Line2D([], [], color=COLOR_PEAK, marker="x", ls="None", mew=2, ms=9), "Detected peak"),
        (Patch(fc=COLOR_PEAK, alpha=0.15), "Peak-averaging window (n points)"),
    ]
    if show_thresholds:
        items.append((Line2D([], [], color=COLOR_AMPLITUDE, lw=1.2, marker=">", ms=5),
                      "Amplitude threshold (a)"))
    items += [
        (Patch(fc=COLOR_DECAY_WINDOW, alpha=0.35), "Decay search window (f)"),
        (Line2D([], [], color=COLOR_DECAY_LEVEL, ls=":", lw=1.4), "Decay-fraction level (g)"),
        (Line2D([], [], color=COLOR_ONSET, marker="o", ls="None", ms=6), "Onset (rise crossing)"),
        (Line2D([], [], color=COLOR_DECAY_LEVEL, marker="o", ls="None", ms=6), "Decay point"),
    ]
    if show_thresholds:
        items += [
            (Patch(fc=COLOR_AREA_PASS, alpha=0.35), "Measured area -- PASS (b)"),
            (Patch(fc=COLOR_AREA_FAIL, alpha=0.35), "Measured area -- FAIL (b)"),
        ]
    else:
        items.append((Patch(fc=COLOR_AREA_PASS, alpha=0.35), "Measured area"))
    return items


def plot_event_detail(ax, t: np.ndarray, v: np.ndarray, dt: float, row: pd.Series,
                       params: DetectionParams, y_unit: str, pre_ms: float, post_ms: float,
                       show_thresholds: bool = True):
    """Draw one detected event with every detection-parameter window/
    threshold annotated on it -- the interactive counterpart of
    this_method.pdf p.2's "Detection Parameters" diagram, but on the real
    trace that actually produced this event. See build_legend_handles() for
    what each color/marker means."""
    peak_idx = int(row["peak_idx"])
    baseline, amplitude = float(row["baseline"]), float(row["amplitude"])
    sign = -1.0 if params.direction == "negative" else 1.0
    peak_v = baseline + amplitude

    geo = _event_geometry(v, dt, peak_idx, baseline, amplitude, params)
    pre_n = _samples(pre_ms, dt)
    post_n = _samples(post_ms, dt)
    i0, i1 = max(0, peak_idx - pre_n), min(len(v), peak_idx + post_n)
    t_ms = (np.arange(i0, i1) - peak_idx) * dt * 1e3

    def rel(idx):
        return (idx - peak_idx) * dt * 1e3

    ax.clear()
    ax.set_facecolor(SURFACE)
    ax.plot(t_ms, v[i0:i1], color=TRACE, lw=1.0, zorder=3)

    # (e) baseline-averaging window + (d) time-before-peak-for-baseline gap
    ax.axvspan(rel(geo["b0"]), rel(geo["b1"]), color=COLOR_BASELINE_WINDOW, alpha=0.35, zorder=1)
    ax.axvspan(rel(geo["b1"]), rel(peak_idx), color=COLOR_BEFORE_PEAK, alpha=0.35, zorder=1)
    ax.axhline(baseline, color=COLOR_BASELINE_LINE, ls="--", lw=1.2, zorder=2)

    # Onset search window (rise-side counterpart of decay_search_ms (f)) --
    # a boundary line, not a filled span, since it can extend further back
    # than (or overlap) the (d)/(e) bands above and a second overlapping
    # fill would just muddy them. (See the annotation key for what this is.)
    ax.axvline(rel(geo["onset_search_start"]), color=COLOR_ONSET_WINDOW, ls="--", lw=1.4, zorder=2)

    # (c) illustrative local-maximum search span, centered on the peak.
    # Classical-only: it comes from search_local_max_ms, which is the window
    # find_peaks(distance=...) used to pick candidates. A detector that found
    # its candidates another way never applied it, so drawing it would imply
    # a constraint that was not there.
    if show_thresholds:
        y_c = baseline + sign * 0.08 * abs(amplitude if amplitude else 1.0)
        x_c_lo, x_c_hi = rel(geo["local_max_lo"]), rel(geo["local_max_hi"])
        ax.annotate("", xy=(x_c_lo, y_c), xytext=(x_c_hi, y_c),
                    arrowprops=dict(arrowstyle="<->", color=COLOR_LOCAL_MAX, lw=1.4))
        ax.text((x_c_lo + x_c_hi) / 2, y_c, "(c)", color=COLOR_LOCAL_MAX, fontsize=9,
                fontweight="bold", ha="center", va="bottom", bbox=_LABEL_BBOX, zorder=6)

    # peak marker + n_avg_peak averaging window
    ax.plot(0, peak_v, "x", color=COLOR_PEAK, ms=10, mew=2, zorder=5)
    ax.axvspan(rel(geo["p0"]), rel(geo["p1"]), color=COLOR_PEAK, alpha=0.15, zorder=1)

    # (a) amplitude threshold, drawn as a bracket from baseline
    thresh_v = baseline + sign * params.amplitude_threshold
    pass_amp = abs(amplitude) >= params.amplitude_threshold
    if show_thresholds:
        x_a = t_ms[0] * 0.5
        ax.annotate("", xy=(x_a, baseline), xytext=(x_a, thresh_v),
                    arrowprops=dict(arrowstyle="<->", color=COLOR_AMPLITUDE, lw=1.4))
        ax.text(x_a, (baseline + thresh_v) / 2, f" a={params.amplitude_threshold}",
                color=COLOR_AMPLITUDE, fontsize=9, fontweight="bold", va="center",
                ha="left", bbox=_LABEL_BBOX, zorder=6)
        ax.axhline(thresh_v, color=COLOR_AMPLITUDE, ls=":", lw=1.0, zorder=1)

    # (f) decay-search window, (g) decay-fraction level, onset/decay markers
    ax.axvspan(rel(peak_idx), rel(geo["decay_search_end"]), color=COLOR_DECAY_WINDOW, alpha=0.20, zorder=1)
    # NOTE: no `sign` factor here, unlike thresh_v above -- `amplitude` (unlike
    # params.amplitude_threshold) is already signed, so baseline + amplitude
    # == peak_v; multiplying by sign again would flip this to the wrong side
    # of baseline entirely instead of landing between baseline and the peak.
    decay_level = baseline + (1.0 - params.decay_fraction) * amplitude
    ax.axhline(decay_level, color=COLOR_DECAY_LEVEL, ls=":", lw=1.2, zorder=1)
    ax.text(rel(geo["decay_search_end"]), decay_level, f" g={params.decay_fraction}", color=COLOR_DECAY_LEVEL,
             fontsize=9, fontweight="bold", ha="left", va="center", bbox=_LABEL_BBOX, zorder=6)
    ax.plot(rel(geo["onset_idx"]), v[geo["onset_idx"]], "o", color=COLOR_ONSET, ms=6,
             mec=INK, mew=0.5, zorder=4)
    ax.plot(rel(geo["decay_idx"]), v[geo["decay_idx"]], "o", color=COLOR_DECAY_LEVEL, ms=6,
             mec=INK, mew=0.5, zorder=4)

    # (b) area under the curve between onset and decay, shaded pass/fail
    # (blue/red, deliberately distinct from the green baseline-window span
    # above -- the two can sit close together near the peak)
    area_ok = abs(row["area"]) >= params.area_threshold
    fill_color = COLOR_AREA_PASS if area_ok else COLOR_AREA_FAIL
    onset_i, decay_i = geo["onset_idx"], geo["decay_idx"] + 1
    ax.fill_between(rel(np.arange(onset_i, decay_i)), v[onset_i:decay_i], baseline,
                     color=fill_color, alpha=0.35, zorder=2)

    ax.set_xlabel("time from peak (ms)", color=INK, fontsize=10)
    ax.set_ylabel(f"current ({y_unit})", color=INK, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(alpha=0.15, color=GRID)

    if show_thresholds:
        verdict = f"amp {'PASS' if pass_amp else 'FAIL'} / area {'PASS' if area_ok else 'FAIL'}"
        title = (f"t={row['peak_time_s']:.3f}s   amplitude={amplitude:.2f} {y_unit} "
                 f"(thr {params.amplitude_threshold})   "
                 f"area={row['area']:.2f} (thr {params.area_threshold})   [{verdict}]")
    else:
        # This detector applied no amplitude/area threshold, so report the
        # measurements without a pass/fail verdict against one that was never
        # checked. Kinetics go here instead -- they are what an operator
        # actually judges a deconvolution candidate on.
        rise = row.get("rise_time_ms", float("nan"))
        decay = row.get("decay_time_ms", float("nan"))
        title = (f"t={row['peak_time_s']:.3f}s   amplitude={amplitude:.2f} {y_unit}   "
                 f"area={row['area']:.2f}   rise={rise:.2f} ms   decay={decay:.2f} ms")
    ax.set_title(title, color=INK, fontsize=10)


def _source_criteria_text(source: str, params_path: str) -> str:
    """The criteria a non-classical detector actually applied, for the panel.

    The shared DetectionParams fields cover how each event was MEASURED, but
    not what made it an event in the first place. Rather than leave that blank
    (or worse, let the classical amplitude/area defaults stand in for it),
    read the source's own sidecar and state its real thresholds.
    """
    if source == "minianalysis" or not os.path.exists(params_path):
        return ""
    try:
        with open(params_path) as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return ""
    if source != "deconv":
        return ""
    rows = [
        ("Threshold", f"{raw.get('thresh_sd', '?')} SD of deconvolved trace"),
        ("Min amplitude", f"{raw.get('min_amp_sd', '?')} SD of raw trace"),
    ]
    if raw.get("min_amp_pa"):
        rows.append(("Absolute floor", f"{raw['min_amp_pa']} (trace units)"))
    if raw.get("noise_sd") is not None:
        rows.append(("Noise SD", f"{raw['noise_sd']:.2f}"))
        rows.append(("4 SD limit", f"{raw.get('detection_limit_4sd', 0.0):.2f}"))
    if raw.get("kernel_ms") is not None:
        rows.append(("Kernel", f"{raw['kernel_ms']:.1f} ms, {raw.get('kernel_source', '?')}"
                               f" ({raw.get('n_isolated_for_kernel', '?')} isolated)"))
    body = "\n".join(f"{k}: {v}" for k, v in rows)
    return "\nDeconvolution criteria\n" + "-" * 26 + "\n" + body


def _load_prior_decisions(reviewed_path: str, progress_path: str) -> dict:
    """{peak_idx: accepted_bool} from an existing progress file, or from a
    reviewed-only file (all True) if that's all that exists -- so a QC pass
    interrupted halfway through picks up exactly where it left off."""
    if os.path.exists(progress_path):
        prior = pd.read_csv(progress_path)
        return dict(zip(prior["location"].astype(int), prior["decision"] == 1))
    if os.path.exists(reviewed_path):
        prior = pd.read_csv(reviewed_path)
        return dict(zip(prior["location"].astype(int), [True] * len(prior)))
    return {}


class EventInspector(QtWidgets.QMainWindow):
    """The window itself: full-trace overview on top, one annotated event
    below, Accept/Reject QC underneath. Kept separate from main() so it can
    also be opened from a script or a notebook on an already-loaded trace
    (see the README's "Use it as a library" section)."""

    def __init__(self, t: np.ndarray, v: np.ndarray, dt: float, df: pd.DataFrame,
                 params: DetectionParams, y_unit: str, title: str, pre_ms: float, post_ms: float,
                 reviewed_path: str, progress_path: str,
                 show_thresholds: bool = True, extra_param_text: str = ""):
        super().__init__()
        # False when reviewing a detector with no amplitude/area threshold to
        # pass or fail; see params_text and plot_event_detail.
        self.show_thresholds = show_thresholds
        self.extra_param_text = extra_param_text
        self.t, self.v, self.dt = t, v, dt
        self.reviewed_path, self.progress_path = reviewed_path, progress_path
        self.decisions: dict = _load_prior_decisions(reviewed_path, progress_path)
        self.df = df.reset_index(drop=True)
        self.params = params
        self.y_unit = y_unit
        self.pre_ms, self.post_ms = pre_ms, post_ms
        self.idx = 0
        self._pan_ax = None  # set while a left-button drag-to-pan is in progress; see _on_press

        self.setWindowTitle(title)
        self.resize(1600, 880)

        self.figure = Figure(figsize=(15, 8.5))
        self.figure.patch.set_facecolor(SURFACE)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)
        # Standard matplotlib pan/zoom-rectangle/home toolbar -- gives both
        # panels independent click-drag pan and box-zoom (each Axes keeps
        # its own view/history), on top of the scroll-wheel zoom wired up
        # below via _on_scroll.
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # -- QC row: Accept/Reject this event, or Accept All Remaining -----
        qc_row = QtWidgets.QHBoxLayout()
        self.accept_btn = QtWidgets.QPushButton("Accept  (A)")
        self.reject_btn = QtWidgets.QPushButton("Reject  (R)")
        self.accept_all_btn = QtWidgets.QPushButton("Accept All Remaining")
        for btn in (self.accept_btn, self.reject_btn, self.accept_all_btn):
            btn.setFocusPolicy(QtCore.Qt.NoFocus)  # keep keyboard focus on the canvas, not the button
        # Same green/red accept/reject language as COLOR_ACCEPTED/
        # COLOR_REJECTED's overview halos above, so a decision reads the
        # same way in the buttons and on the trace.
        self.accept_btn.setStyleSheet(
            "QPushButton { background-color: #a0d8a0; font-weight: bold; }"
            "QPushButton:hover { background-color: #66bb6a; }"
            "QPushButton:pressed { background-color: #4c9950; }")
        self.reject_btn.setStyleSheet(
            "QPushButton { background-color: #f2a0a0; font-weight: bold; }"
            "QPushButton:hover { background-color: #e57373; }"
            "QPushButton:pressed { background-color: #c85a5a; }")
        self.accept_all_btn.setStyleSheet(
            "QPushButton { background-color: #c8e6c9; }"
            "QPushButton:hover { background-color: #a5d6a7; }")
        self.accept_btn.clicked.connect(self.accept_current)
        self.reject_btn.clicked.connect(self.reject_current)
        self.accept_all_btn.clicked.connect(self.accept_all_remaining)
        qc_row.addWidget(self.accept_btn)
        qc_row.addWidget(self.reject_btn)
        qc_row.addWidget(self.accept_all_btn)
        qc_row.addStretch(1)
        self.qc_status_label = QtWidgets.QLabel("")
        self.qc_status_label.setStyleSheet("font-family: Consolas, monospace;")
        self.qc_status_label.setTextFormat(QtCore.Qt.RichText)
        layout.addLayout(qc_row)
        layout.addWidget(self.qc_status_label)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Click a red X in the top panel to inspect that event, or use Right/N, Left/P to step. "
            "A/Up = accept, R/Down = reject.")

        self.ax_over, self.ax_detail = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [1, 2]})
        # Pan/zoom (scroll, drag, and the toolbar's own Pan/Zoom-rectangle
        # tool) is for the full-trace overview only -- the detail view
        # below it is a per-event snapshot that plot_event_detail always
        # redraws fresh (auto-fit to pre_ms/post_ms around the selected
        # peak) on every navigation/QC action, so it stays exactly as
        # before: no independent view of its own.
        self.ax_detail.set_navigate(False)

        stride = max(1, len(v) // OVERVIEW_MAX_POINTS)
        self.ax_over.set_facecolor(SURFACE)
        self.ax_over.plot(t[::stride], v[::stride], color=TRACE, lw=0.3, zorder=1)
        # QC halos, drawn BEHIND the x markers (zorder 2 < 3) so an
        # undecided event still just shows its plain crimson x, while a
        # decided one gets a colored ring around it.
        self.accepted_overlay, = self.ax_over.plot(
            [], [], "o", mfc=COLOR_ACCEPTED, mec="none", ms=9, zorder=2, label="accepted")
        self.rejected_overlay, = self.ax_over.plot(
            [], [], "o", mfc=COLOR_REJECTED, mec="none", ms=9, zorder=2, label="rejected")
        self.markers, = self.ax_over.plot(
            self.df["peak_time_s"], v[self.df["peak_idx"].to_numpy()], "x", color="crimson",
            ms=6, mew=1.2, zorder=3, picker=5, label=f"{len(self.df)} detected events (click one)")
        self.selected_marker, = self.ax_over.plot([], [], "o", color="gold", ms=12, mfc="none",
                                                     mew=2, zorder=4)
        self.ax_over.set_xlabel("time (s)", color=INK)
        self.ax_over.set_ylabel(f"current ({y_unit})", color=INK)
        self.ax_over.tick_params(colors=MUTED)
        self.ax_over.legend(loc="upper right", fontsize=8, labelcolor=INK)
        self._refresh_overview_markers()

        # Sidebar reserved at figure fraction x >= 0.66 (5.1in wide at this
        # figure size) -- wide enough for the longest parameter line at this
        # font size, so the box's left edge (it's left-aligned, unlike the
        # old right-aligned version that could spill past its own margin)
        # never crosses into the plot area.
        self.figure.subplots_adjust(right=0.65, hspace=0.35)
        SIDEBAR_X = 0.675

        # y-anchors: params box starts near the figure's own top (it lives in
        # figure coordinates, disjoint in x from both subplots above, so
        # this is safe regardless of where ax_over/ax_detail sit) and the
        # legend starts far enough below it to clear all 9 parameter lines
        # at this font size -- 8 lines used to fit under the old y=0.60/0.42
        # split, but the 9th (onset_search_ms) made the box tall enough to
        # overlap the legend below it.
        sidebar_body = "Detection parameters\n" + "-" * 26 + "\n" + params_text(params, self.show_thresholds, self.extra_param_text)
        self.param_text = self.figure.text(
            SIDEBAR_X, 0.97, sidebar_body,
            transform=self.figure.transFigure, ha="left", va="top", fontsize=9,
            family="monospace", color=INK,
            bbox=dict(boxstyle="round,pad=0.5", fc="#f2f1ea", ec=MUTED, alpha=0.95))

        # Anchor the key BELOW whatever the parameter box actually occupies,
        # rather than at a fixed y. The box's height depends on the source (a
        # non-classical detector adds its own criteria block), and a hard-coded
        # anchor is exactly what put the 9th parameter line under the legend
        # once already -- see the note above. 0.0245 per line at fontsize 9,
        # plus the box's own padding, measured to reproduce the previous 0.68
        # for the classical detector's 11 lines.
        n_lines = sidebar_body.count("\n") + 1
        key_y = 0.97 - n_lines * 0.0245 - 0.03

        key_handles, key_labels = zip(*build_legend_handles(self.show_thresholds))
        self.figure.legend(
            key_handles, key_labels, loc="upper left", bbox_to_anchor=(SIDEBAR_X, key_y),
            bbox_transform=self.figure.transFigure, fontsize=8.5, labelcolor=INK,
            title="Annotation key", title_fontsize=9.5, frameon=True,
            facecolor="#f2f1ea", edgecolor=MUTED, framealpha=0.95, handlelength=1.8)

        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_pan_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_pan_motion)
        self.canvas.mpl_connect("button_release_event", self.on_pan_release)
        self.show_event()
        self.canvas.setFocus()

    def show_event(self):
        row = self.df.iloc[self.idx]
        self.selected_marker.set_data([row["peak_time_s"]], [self.v[int(row["peak_idx"])]])
        plot_event_detail(self.ax_detail, self.t, self.v, self.dt, row, self.params,
                           self.y_unit, self.pre_ms, self.post_ms,
                           show_thresholds=self.show_thresholds)

        peak_idx = int(row["peak_idx"])
        qc = self.decisions.get(peak_idx)
        qc_word = "ACCEPTED" if qc is True else "REJECTED" if qc is False else "undecided"
        self.ax_detail.set_title(
            f"Event {self.idx + 1}/{len(self.df)}   QC: {qc_word}   " + self.ax_detail.get_title(),
            fontsize=10, color=INK)
        self.canvas.draw_idle()

        n_decided = len(self.decisions)
        n_accepted = sum(1 for ok in self.decisions.values() if ok)
        n_rejected = n_decided - n_accepted
        qc_color = {"ACCEPTED": COLOR_ACCEPTED, "REJECTED": COLOR_REJECTED, "undecided": MUTED}[qc_word]
        self.qc_status_label.setText(
            f"This event: <span style='color:{qc_color}; font-weight:bold;'>{qc_word}</span>"
            f"    |    Reviewed: {n_decided}/{len(self.df)} "
            f"(<span style='color:{COLOR_ACCEPTED};'>{n_accepted} accepted</span>, "
            f"<span style='color:{COLOR_REJECTED};'>{n_rejected} rejected</span>, "
            f"{len(self.df) - n_decided} undecided)")

    def _refresh_overview_markers(self):
        accepted_idxs = [i for i, ok in self.decisions.items() if ok]
        rejected_idxs = [i for i, ok in self.decisions.items() if not ok]
        acc_rows = self.df[self.df["peak_idx"].isin(accepted_idxs)]
        rej_rows = self.df[self.df["peak_idx"].isin(rejected_idxs)]
        self.accepted_overlay.set_data(acc_rows["peak_time_s"], self.v[acc_rows["peak_idx"].to_numpy()])
        self.rejected_overlay.set_data(rej_rows["peak_time_s"], self.v[rej_rows["peak_idx"].to_numpy()])

    def _autosave(self):
        """Write <stem>_minianalysis_reviewed.csv (accepted events) and
        <stem>_minianalysis_review_progress.csv
        (every decided event + a decision column, 1=accepted/0=rejected) --
        called after every single decision, so nothing is ever lost."""
        if not self.decisions:
            return
        decided = self.df[self.df["peak_idx"].isin(self.decisions.keys())].copy()
        decided["decision"] = decided["peak_idx"].map(lambda i: int(self.decisions[i]))
        decided = decided.rename(columns={"peak_idx": "location", "peak_time_s": "location_s"})
        decided.to_csv(self.progress_path, index=False)
        decided[decided["decision"] == 1].drop(columns="decision").to_csv(self.reviewed_path, index=False)

    def _set_decision(self, peak_idx: int, accepted: bool):
        self.decisions[peak_idx] = accepted
        self._autosave()
        self._refresh_overview_markers()

    @safe_callback
    def accept_current(self, *_qt_args):
        self._set_decision(int(self.df.iloc[self.idx]["peak_idx"]), True)
        self.idx = min(self.idx + 1, len(self.df) - 1)
        self.show_event()
        self.canvas.setFocus()

    @safe_callback
    def reject_current(self, *_qt_args):
        self._set_decision(int(self.df.iloc[self.idx]["peak_idx"]), False)
        self.idx = min(self.idx + 1, len(self.df) - 1)
        self.show_event()
        self.canvas.setFocus()

    @safe_callback
    def accept_all_remaining(self, *_qt_args):
        newly = 0
        for peak_idx in self.df["peak_idx"]:
            peak_idx = int(peak_idx)
            if peak_idx not in self.decisions:
                self.decisions[peak_idx] = True
                newly += 1
        self._autosave()
        self._refresh_overview_markers()
        self.show_event()
        self.statusBar().showMessage(f"Accepted {newly} remaining undecided event(s).", 5000)
        self.canvas.setFocus()

    @safe_callback
    def on_scroll(self, scroll_event):
        """Zoom the full-trace overview, centered on the cursor's data
        position. Plain scroll zooms time (X) and current (Y) together;
        hold Ctrl to zoom time only, Shift to zoom current only. Overview
        only, deliberately: the detail view below it is a per-event
        snapshot (plot_event_detail always redraws it fresh, auto-fit to
        pre_ms/post_ms around the selected peak), not something meant to
        be independently panned/zoomed.

        Modifier state is read directly from Qt (QApplication.
        keyboardModifiers()), NOT scroll_event.key: the Qt backend's own
        wheelEvent (backend_qt.FigureCanvasQT.wheelEvent) builds its
        MouseEvent with only x/y/step, never a key= kwarg, so
        scroll_event.key is always None for a real scroll -- matplotlib's
        usual event.key modifier check silently never fires here."""
        ax = scroll_event.inaxes
        if ax is not self.ax_over or scroll_event.xdata is None or scroll_event.ydata is None:
            return
        scale = 1 / 1.2 if scroll_event.button == "up" else 1.2
        xdata, ydata = scroll_event.xdata, scroll_event.ydata

        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl = bool(modifiers & QtCore.Qt.ControlModifier)
        shift = bool(modifiers & QtCore.Qt.ShiftModifier)
        zoom_x, zoom_y = (True, False) if (ctrl and not shift) else \
                         (False, True) if (shift and not ctrl) else (True, True)

        if zoom_x:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(xdata - (xdata - x0) * scale, xdata + (x1 - xdata) * scale)
        if zoom_y:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(ydata - (ydata - y0) * scale, ydata + (y1 - ydata) * scale)
        self.canvas.draw_idle()

    @safe_callback
    def on_pan_press(self, press_event):
        """Start of a left-drag pan on the full-trace overview (see
        on_scroll for why the detail view is excluded) -- deliberately NOT
        gated on distance from on_pick's marker picking: pick_event fires
        independently on the same press (a plain click still selects an
        event as usual), this only starts tracking a possible drag.
        Deferred entirely to the toolbar's own Pan/Zoom-rectangle tool
        while either is toggled on (self.toolbar.mode is non-empty then),
        so the two never fight over the same drag."""
        if press_event.button != 1 or press_event.inaxes is not self.ax_over or self.toolbar.mode != "":
            return
        self._pan_ax = press_event.inaxes
        self._pan_x0_px, self._pan_y0_px = press_event.x, press_event.y
        self._pan_xlim0 = press_event.inaxes.get_xlim()
        self._pan_ylim0 = press_event.inaxes.get_ylim()

    @safe_callback
    def on_pan_motion(self, motion_event):
        if self._pan_ax is None or motion_event.x is None or motion_event.y is None:
            return
        bbox = self._pan_ax.get_window_extent()
        if bbox.width <= 0 or bbox.height <= 0:
            return
        dx_px = motion_event.x - self._pan_x0_px
        dy_px = motion_event.y - self._pan_y0_px
        x0, x1 = self._pan_xlim0
        y0, y1 = self._pan_ylim0
        dx_data = dx_px / bbox.width * (x1 - x0)
        dy_data = dy_px / bbox.height * (y1 - y0)
        self._pan_ax.set_xlim(x0 - dx_data, x1 - dx_data)
        self._pan_ax.set_ylim(y0 - dy_data, y1 - dy_data)
        self.canvas.draw_idle()

    @safe_callback
    def on_pan_release(self, release_event):
        self._pan_ax = None

    @safe_callback
    def on_pick(self, pick_event):
        if pick_event.artist is not self.markers or len(pick_event.ind) == 0:
            return
        self.idx = int(pick_event.ind[0])
        self.show_event()

    @safe_callback
    def on_key(self, key_event):
        if key_event.key in ("right", "n"):
            self.idx = min(self.idx + 1, len(self.df) - 1)
            self.show_event()
        elif key_event.key in ("left", "p"):
            self.idx = max(self.idx - 1, 0)
            self.show_event()
        elif key_event.key in ("a", "up"):
            self.accept_current()
        elif key_event.key in ("r", "down"):
            self.reject_current()



def load_display_trace(abf_path: str, channel: int, filter_flag: bool, cutoff_hz: float,
                        target_rate_hz: float, filter_order: int):
    """Load (t, v, dt, y_unit) -- the raw trace, or the same Bessel-filtered
    + resampled one run.py's own --filter produces, if filter_flag. Callers
    resolve the stem (and so can fail fast on a missing CSV) via
    core.output_stem BEFORE calling this, so no time is spent filtering a
    multi-million-sample trace for a run that was never going to find its
    events file. A raw-vs-filtered mismatch between detection and checking
    silently misaligns every event, so this deliberately shares its
    defaults with run.py rather than having any of its own."""
    abf = pyabf.ABF(abf_path)
    abf.setSweep(0, channel=channel)
    y_unit = abf.adcUnits[channel]

    if filter_flag:
        hardware_filter_hz = get_hardware_filter_hz(abf, channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= cutoff_hz:
            print(f"NOTE: channel {channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz, at or below the requested {cutoff_hz:.0f} Hz.")
        t, v, fs, _ = load_filtered_trace(abf_path, channel=channel, cutoff_hz=cutoff_hz,
                                           target_hz=target_rate_hz, order=filter_order)
        dt = 1.0 / fs
        print(f"Filtered: {abf.dataRate:.0f} Hz raw -> {cutoff_hz:.0f} Hz Bessel "
              f"(order {filter_order}, zero-phase) -> {fs:.0f} Hz", flush=True)
    else:
        t = np.asarray(abf.sweepX, float)
        v = np.asarray(abf.sweepY, float)
        dt = 1.0 / abf.dataRate

    return t, v, dt, y_unit




def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--source", choices=sorted(SOURCES), default="minianalysis",
                   help="which detector's events to review: 'minianalysis' (the "
                        "classical threshold detector, from `run`) or 'deconv' "
                        "(the deconvolution detector, from `deconvolve`). Each "
                        "keeps its own reviewed/progress CSVs, so reviewing one "
                        "never touches the other's decisions.")
    p.add_argument("--csv", default=None,
                   help="Path to the events CSV (default: <stem> + the chosen "
                        "--source's events suffix)")
    p.add_argument("--params-json", default=None,
                   help="Path to the params sidecar (default: <stem> + the chosen "
                        "--source's params suffix)")
    p.add_argument("--pre-ms", type=float, default=None,
                   help="detail-view window before the peak, ms (default: auto, from the baseline "
                        "parameters with 3 ms headroom)")
    p.add_argument("--post-ms", type=float, default=None,
                   help="detail-view window after the peak, ms (default: auto, from decay_search_ms "
                        "with 3 ms headroom)")
    # Same override flags as run.py, applied on top of the sidecar (or its own defaults) only when
    # explicitly passed -- see _load_params.
    p.add_argument("--direction", choices=["negative", "positive"], default=None)
    p.add_argument("--amplitude-threshold", type=float, default=None)
    p.add_argument("--area-threshold", type=float, default=None)
    p.add_argument("--n-avg-peak", type=int, default=None)
    p.add_argument("--search-local-max-ms", type=float, default=None)
    p.add_argument("--baseline-before-ms", type=float, default=None)
    p.add_argument("--baseline-avg-ms", type=float, default=None)
    p.add_argument("--decay-search-ms", type=float, default=None)
    p.add_argument("--decay-fraction", type=float, default=None)
    p.add_argument("--onset-fraction", type=float, default=None)
    p.add_argument("--onset-search-ms", type=float, default=None)
    p.add_argument("--filter", action="store_true",
                   help="load the SAME Bessel-filtered + resampled trace the detection run's own "
                        "--filter produced (must match that run's settings exactly, since event "
                        "peak_idx values are indices into that filtered/resampled trace, not the "
                        "raw one)")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    args = p.parse_args(argv)

    # Same stem the run that produced these files built (core.output_stem
    # is shared with run.py and optimize.py precisely so this can't drift).
    # Resolved before load_display_trace below, so a missing events CSV is
    # reported instantly instead of after filtering a huge trace.
    stem = output_stem(args.abf, args.channel, args.filter, args.cutoff_hz, args.target_rate_hz)
    events_suffix, params_suffix, reviewed_suffix, progress_suffix = SOURCES[args.source]
    csv_path = args.csv or f"{stem}{events_suffix}"
    if not os.path.exists(csv_path):
        # Name the command that produces THIS source's CSV, including the
        # filtering flag, since the stem encodes it and a mismatch is the
        # usual reason the file "isn't there".
        if args.source == "deconv":
            hint = (f"python -m minianalysis deconvolve {args.abf}"
                    f"{'' if args.filter else ' --no-filter'}")
        else:
            hint = (f"python -m minianalysis run {args.abf}"
                    f"{' --filter' if args.filter else ''}")
        p.error(f"events CSV not found: {csv_path!r} (run `{hint}` first, or pass --csv)")

    t, v, dt, y_unit = load_display_trace(args.abf, args.channel, args.filter, args.cutoff_hz,
                                           args.target_rate_hz, args.filter_order)

    # float_precision="round_trip", not the default: pandas' fast C float
    # parser is not correctly rounded, so a baseline/amplitude written at
    # full precision by run.py comes back an ULP or two off (357 of 2292
    # numeric cells, on a real 573-event recording). Every window this tool
    # draws is derived from those two numbers, and a level that lands an ULP
    # away from a sample sitting right at it moves the onset/decay marker to
    # a different sample -- i.e. the tool would draw something the detector
    # did not do, on exactly the marginal events an operator looks at
    # hardest. Reading exactly what was written costs nothing here.
    df = pd.read_csv(csv_path, float_precision="round_trip")
    if df.empty:
        p.error("no events in that CSV")

    overrides = dict(
        direction=args.direction, amplitude_threshold=args.amplitude_threshold,
        area_threshold=args.area_threshold, n_avg_peak=args.n_avg_peak,
        search_local_max_ms=args.search_local_max_ms, baseline_before_ms=args.baseline_before_ms,
        baseline_avg_ms=args.baseline_avg_ms, decay_search_ms=args.decay_search_ms,
        decay_fraction=args.decay_fraction, onset_fraction=args.onset_fraction,
        onset_search_ms=args.onset_search_ms,
    )
    params = _load_params(args.params_json, stem, overrides, params_suffix)
    pre_ms = args.pre_ms if args.pre_ms is not None else params.baseline_before_ms + params.baseline_avg_ms + 3.0
    post_ms = args.post_ms if args.post_ms is not None else params.decay_search_ms + 3.0

    reviewed_path = f"{stem}{reviewed_suffix}"
    progress_path = f"{stem}{progress_suffix}"

    print(f"{len(df)} events from {os.path.basename(csv_path)}", flush=True)
    print("Click a red X in the top panel to inspect that event, or use Right/N, Left/P to step. "
          "A/Up = accept, R/Down = reject.", flush=True)
    if os.path.exists(progress_path) or os.path.exists(reviewed_path):
        print(f"Resuming QC from {os.path.basename(progress_path)}"
              f"{' / ' + os.path.basename(reviewed_path) if os.path.exists(reviewed_path) else ''}", flush=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = EventInspector(t, v, dt, df, params, y_unit,
                             title=f"{SOURCE_LABELS[args.source]} -- {os.path.basename(args.abf)}",
                             pre_ms=pre_ms, post_ms=post_ms,
                             reviewed_path=reviewed_path, progress_path=progress_path,
                             show_thresholds=(args.source == "minianalysis"),
                             extra_param_text=_source_criteria_text(
                                 args.source, args.params_json or f"{stem}{params_suffix}"))
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()

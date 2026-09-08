"""
Run detection on a recording: the batch/one-shot command. Loads a gap-free
voltage-clamp .abf, optionally Bessel-filters + resamples it
(preprocess.py), runs the Mini-Analysis-style detector (core.detect_events)
over the whole trace, and writes the results next to the .abf:

    <stem>_minianalysis_events.csv    one row per detected event
                                      (peak_idx, peak_time_s, baseline,
                                      amplitude, rise_time_ms,
                                      decay_time_ms, area,
                                      inter_event_interval_ms)
    <stem>_minianalysis_params.json   the exact DetectionParams used
    <stem>_minianalysis_trace.png     whole trace with every event marked

where <stem> is the .abf path without its extension, plus a
`_filt<cutoff>Hz<rate>Hz` suffix when --filter was used. check.py reads the
params sidecar back, so the windows it draws on an event are guaranteed to
be the ones that actually produced it.

By default a modal dialog opens first, pre-filled with the 9 detection
parameters plus the filter/resample settings, so they can be reviewed and
edited before anything runs -- the tutorial's own "enter parameters, click
OK" workflow. --no-gui skips it and runs straight from the flags below,
for scripts and batch jobs (and needs no PyQt5 at all: it's imported lazily
inside the dialog function).

The optional analysis stage (--stats, --histogram-column, --autocorr,
--cross-corr-csv, --group-analysis, --fit-decay) runs the tutorial's
downstream analyses on the events just detected -- see core.py.

Usage
-----
    python run_detection.py path\to\recording.abf
    python -m minianalysis run path\to\recording.abf        # same thing
        # ^ both open the parameters dialog first; edit or leave the values, then OK

    python -m minianalysis run recording.abf --no-gui --amplitude-threshold 8 --area-threshold 15
    python -m minianalysis run recording.abf --no-gui --filter --cutoff-hz 3000 --target-rate-hz 10000
    python -m minianalysis run recording.abf --no-gui --stats --histogram-column amplitude \
        --autocorr --group-analysis --fit-decay peak_to_end
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pyabf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import (
    PARAM_FIELD_SPECS, DetectionParams, _samples, autocorrelation_histogram, column_statistics,
    cross_correlation_histogram, cumulative_histogram, detect_events, events_frame,
    extract_event_traces, fit_exponential_decay, frequency_histogram, scale_traces,
)
from .gui_utils import FILTER_FIELD_SPECS, make_field_widget, read_field_widget
from .preprocess import (
    DEFAULT_CUTOFF_HZ, DEFAULT_ORDER, DEFAULT_TARGET_RATE_HZ, get_hardware_filter_hz,
    load_filtered_trace,
)
from .style import INK, MUTED, SURFACE, TRACE

# ---------------------------------------------------------------------------
# --no-gui (default OFF, i.e. the dialog shows unless disabled): a modal
# parameters dialog, pre-filled with defaults (or whatever --amplitude-
# threshold/--filter/etc. were also passed), shown before detection runs --
# the tutorial's own "enter parameters, click OK" workflow ("First use
# mouse-click to detect"). Covers both the 9 detection parameters AND the
# filter/resample settings (the Bessel low-pass step doesn't have a
# tutorial page of its own, but belongs in the same up-front dialog since
# it's the other thing you'd want to set before detection runs). PyQt5 is
# imported lazily, INSIDE this function, so `--no-gui` runs (scripts,
# automation) never need PyQt5 installed at all.


@dataclass
class GuiSettings:
    params: DetectionParams
    filter_enabled: bool
    cutoff_hz: float
    target_rate_hz: float
    filter_order: int


def _prompt_for_settings(defaults: DetectionParams, filter_enabled: bool, cutoff_hz: float,
                          target_rate_hz: float, filter_order: int) -> Optional[GuiSettings]:
    """Modal dialog covering both the 9 detection parameters and the
    filter/resample settings, pre-filled from the arguments above --
    returns a GuiSettings on OK, or None if the user cancelled (caller
    should abort without running detection)."""
    from PyQt5 import QtWidgets

    # Must keep a live reference -- an unassigned QApplication([]) here can
    # be garbage-collected before the QDialog below is constructed (no
    # QApplication existed yet in a plain `python -m minianalysis run ...`
    # run, unlike in tests that pre-create one), which crashes with "QWidget:
    # Must construct a QApplication before a QWidget".
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("Mini Analysis -- Detection Parameters")
    layout = QtWidgets.QVBoxLayout(dialog)

    note = QtWidgets.QLabel(
        "Values are pre-filled with defaults (or any --flags already passed).\n"
        "Leave a field as-is to use that default.")
    note.setWordWrap(True)
    layout.addWidget(note)

    param_group = QtWidgets.QGroupBox("Detection parameters")
    param_form = QtWidgets.QFormLayout(param_group)
    param_widgets = {}
    for attr, label, kind, kwargs in PARAM_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, getattr(defaults, attr))
        param_widgets[attr] = w
        param_form.addRow(label + ":", w)
    layout.addWidget(param_group)

    filter_group = QtWidgets.QGroupBox("Filter / resample (preprocess.py)")
    filter_form = QtWidgets.QFormLayout(filter_group)
    enabled_w = QtWidgets.QCheckBox()
    enabled_w.setChecked(filter_enabled)
    filter_form.addRow("Apply Bessel low-pass + resample:", enabled_w)
    filter_widgets = {}
    filter_current = dict(cutoff_hz=cutoff_hz, target_rate_hz=target_rate_hz, filter_order=filter_order)
    for attr, label, kind, kwargs in FILTER_FIELD_SPECS:
        w = make_field_widget(kind, kwargs, filter_current[attr])
        w.setEnabled(filter_enabled)
        enabled_w.toggled.connect(w.setEnabled)
        filter_widgets[attr] = w
        filter_form.addRow(label + ":", w)
    layout.addWidget(filter_group)

    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    dialog.setMinimumWidth(440)

    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None

    params = DetectionParams(**{
        attr: read_field_widget(param_widgets[attr], kind) for attr, _, kind, _ in PARAM_FIELD_SPECS
    })
    filter_values = {attr: read_field_widget(filter_widgets[attr], kind) for attr, _, kind, _ in FILTER_FIELD_SPECS}
    return GuiSettings(params=params, filter_enabled=enabled_w.isChecked(), **filter_values)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("abf", help="Path to the gap-free voltage-clamp .abf file")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--direction", choices=["negative", "positive"], default="negative",
                   help="peak direction; sEPSCs in voltage clamp are inward/negative (default)")
    p.add_argument("--amplitude-threshold", type=float, default=5.0,
                   help="(a) minimum |amplitude| to accept, in trace units (default 5)")
    p.add_argument("--area-threshold", type=float, default=10.0,
                   help="(b) minimum |area| to accept, in trace-units*ms (default 10)")
    p.add_argument("--n-avg-peak", type=int, default=3,
                   help="points averaged to read the peak value (default 3)")
    p.add_argument("--search-local-max-ms", type=float, default=3.0,
                   help="(c) period to search for a local maximum, in ms (default 3 -- should be "
                        "at least comparable to your events' decay duration, or a single event's "
                        "noisy tail can fragment into multiple spurious detections)")
    p.add_argument("--baseline-before-ms", type=float, default=2.0,
                   help="(d) time before the peak where the baseline window ends, in ms (default 2)")
    p.add_argument("--baseline-avg-ms", type=float, default=3.0,
                   help="(e) duration of the baseline-averaging window, in ms (default 3)")
    p.add_argument("--decay-search-ms", type=float, default=20.0,
                   help="(f) how far past the peak to search for the decay point, in ms (default 20)")
    p.add_argument("--decay-fraction", type=float, default=0.5,
                   help="(g) fraction of peak amplitude defining the decay point (default 0.5 = half-decay)")
    p.add_argument("--onset-fraction", type=float, default=0.1,
                   help="fraction of peak amplitude defining rise onset, for time-to-peak (default 0.1)")
    p.add_argument("--onset-search-ms", type=float, default=5.0,
                   help="how far before the peak to search for the onset_fraction crossing, in ms "
                        "(default 5) -- the rise-side counterpart of decay_search_ms (f); if the "
                        "onset amplitude is never reached within this window, the candidate is "
                        "rejected outright, same as an unreached decay point")
    p.add_argument("--adjust-overlapping-baseline", action="store_true",
                   help="opt into slide 0007's exponential-decay baseline extrapolation for "
                        "closely-spaced events. Off by default: baseline is then ALWAYS the plain "
                        "(d)/(e) window average as those two parameters describe, nothing else; "
                        "turning this on REPLACES that window average with an extrapolated value "
                        "for events still riding a previous one's decay tail")
    p.add_argument("--filter", action="store_true",
                   help="Bessel low-pass + downsample the trace (see preprocess.py) before "
                        "detection, instead of detecting on the raw trace -- lower-noise input "
                        "can reduce spurious local-maximum detections on a noisy recording, at "
                        "the cost of attenuating genuinely fast/small events")
    p.add_argument("--cutoff-hz", type=float, default=DEFAULT_CUTOFF_HZ,
                   help=f"only with --filter: Bessel low-pass cutoff, Hz (default {DEFAULT_CUTOFF_HZ:.0f})")
    p.add_argument("--target-rate-hz", type=float, default=DEFAULT_TARGET_RATE_HZ,
                   help=f"only with --filter: output sampling rate, Hz (default {DEFAULT_TARGET_RATE_HZ:.0f})")
    p.add_argument("--filter-order", type=int, default=DEFAULT_ORDER,
                   help=f"only with --filter: Bessel filter order (default {DEFAULT_ORDER})")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-gui", dest="gui", action="store_false", default=True,
                   help="skip the parameters dialog and run immediately from the flags/defaults "
                        "above -- for scripts/automation. By default (no --no-gui) a dialog shows "
                        "to review/edit the 9 detection parameters plus the filter/resample "
                        "settings, pre-filled with the values above (or their defaults), before "
                        "running -- the tutorial's own 'enter parameters, click OK' workflow. "
                        "Cancel aborts without detecting.")

    analysis = p.add_argument_group(
        "Analysis", "grouping/data-array, descriptive, and group analyses on the detected events (see core.py)")
    analysis.add_argument("--stats", action="store_true",
                   help="save column statistics (n, mean, variance, sd, skewness, kurtosis) "
                        "for amplitude/area/rise_time_ms/decay_time_ms/inter_event_interval_ms")
    analysis.add_argument("--histogram-column",
                   choices=["amplitude", "area", "rise_time_ms", "decay_time_ms", "inter_event_interval_ms"],
                   default=None, help="save a frequency + cumulative histogram for this column")
    analysis.add_argument("--hist-bin-size", type=float, default=None,
                   help="bin width for --histogram-column, in that column's own units "
                        "(default: auto, data range / 30)")
    analysis.add_argument("--autocorr", action="store_true",
                   help="save an auto-correlation histogram of event times, for periodicity")
    analysis.add_argument("--cross-corr-csv", default=None,
                   help="path to another *_events.csv (must have a peak_time_s or location_s column) "
                        "-- save a cross-correlation histogram against THIS run's events")
    analysis.add_argument("--corr-max-lag-ms", type=float, default=500.0,
                   help="+/- window for --autocorr/--cross-corr-csv, ms (default 500)")
    analysis.add_argument("--corr-bin-ms", type=float, default=5.0,
                   help="bin width for --autocorr/--cross-corr-csv, ms (default 5)")
    analysis.add_argument("--group-analysis", action="store_true",
                   help="save a superimposed/averaged (raw and scaled) trace plot across all "
                        "detected events")
    analysis.add_argument("--group-pre-ms", type=float, default=5.0,
                   help="only with --group-analysis: window before each peak, ms (default 5)")
    analysis.add_argument("--group-post-ms", type=float, default=20.0,
                   help="only with --group-analysis: window after each peak, ms (default 20)")
    analysis.add_argument("--fit-decay", choices=["peak_to_end", "decay_10_90", "decay_20_80", "custom"],
                   default=None, help="fit a single/double exponential decay (Simplex) to the "
                        "averaged trace from --group-analysis (implies --group-analysis)")
    analysis.add_argument("--fit-n-exp", type=int, choices=[1, 2], default=1,
                   help="only with --fit-decay: number of exponential components (default 1)")
    analysis.add_argument("--fit-custom-start-ms", type=float, default=None,
                   help="only with --fit-decay custom: fit-window start, ms relative to the peak")
    analysis.add_argument("--fit-custom-end-ms", type=float, default=None,
                   help="only with --fit-decay custom: fit-window end, ms relative to the peak")
    args = p.parse_args(argv)
    if args.fit_decay is not None:
        args.group_analysis = True

    params = DetectionParams(
        amplitude_threshold=args.amplitude_threshold,
        area_threshold=args.area_threshold,
        direction=args.direction,
        n_avg_peak=args.n_avg_peak,
        search_local_max_ms=args.search_local_max_ms,
        baseline_before_ms=args.baseline_before_ms,
        baseline_avg_ms=args.baseline_avg_ms,
        decay_search_ms=args.decay_search_ms,
        decay_fraction=args.decay_fraction,
        onset_fraction=args.onset_fraction,
        onset_search_ms=args.onset_search_ms,
        adjust_overlapping_baseline=args.adjust_overlapping_baseline,
    )

    if args.gui:
        settings = _prompt_for_settings(params, args.filter, args.cutoff_hz,
                                         args.target_rate_hz, args.filter_order)
        if settings is None:
            print("Cancelled in the parameters dialog -- no detection run.")
            return
        params = settings.params
        args.filter = settings.filter_enabled
        args.cutoff_hz = settings.cutoff_hz
        args.target_rate_hz = settings.target_rate_hz
        args.filter_order = settings.filter_order

    abf = pyabf.ABF(args.abf)
    abf.setSweep(0, channel=args.channel)

    if args.filter:
        hardware_filter_hz = get_hardware_filter_hz(abf, args.channel)
        if hardware_filter_hz is not None and hardware_filter_hz <= args.cutoff_hz:
            print(f"NOTE: channel {args.channel} is already hardware-filtered at "
                  f"{hardware_filter_hz:.0f} Hz (amplifier's telegraphed setting), at or below "
                  f"the requested {args.cutoff_hz:.0f} Hz -- filtering further is a no-op.")
        t, v, fs, hardware_filter_hz = load_filtered_trace(
            args.abf, channel=args.channel, cutoff_hz=args.cutoff_hz,
            target_hz=args.target_rate_hz, order=args.filter_order)
        dt = 1.0 / fs
        sweep_len_s = len(v) * dt
        print(f"Filtered: {abf.dataRate:.0f} Hz raw -> {args.cutoff_hz:.0f} Hz Bessel "
              f"(order {args.filter_order}, zero-phase) -> {fs:.0f} Hz", flush=True)

        # Robust (MAD-based, so a few genuinely large events don't inflate
        # it) noise-floor estimate on the FILTERED trace -- filtering
        # correlates adjacent samples, so a filtered noise bump survives
        # this detector's n_avg_peak averaging and area check far more
        # easily than a raw noise spike does. amplitude_threshold/
        # area_threshold tuned against raw-trace noise silently stop being
        # a meaningful bar once the trace is filtered; only warn (never
        # override -- the right value is a real scientific choice, not
        # something to guess on the caller's behalf).
        noise_sd = float(np.median(np.abs(v - np.median(v))) * 1.4826)
        if args.amplitude_threshold < 3 * noise_sd:
            print(f"WARNING: --amplitude-threshold {args.amplitude_threshold:.1f} pA is only "
                  f"{args.amplitude_threshold / noise_sd:.1f}x the filtered trace's noise SD "
                  f"(~{noise_sd:.1f} pA, robust estimate) -- thresholds tuned for the RAW trace "
                  f"don't transfer to filtered data (filtering correlates adjacent noise samples, "
                  f"so noise survives the amplitude/area checks more easily, not less). Consider "
                  f"--amplitude-threshold >= {3 * noise_sd:.0f} (3x SD) as a starting point, and "
                  f"re-tune --area-threshold too.", flush=True)
    else:
        t = np.asarray(abf.sweepX, float)
        v = np.asarray(abf.sweepY, float)
        dt = 1.0 / abf.dataRate
        fs = abf.dataRate
        sweep_len_s = abf.sweepLengthSec

    print(f"Detecting events in {os.path.basename(args.abf)} "
          f"({sweep_len_s:.1f}s @ {fs:.0f} Hz) "
          f"using the Mini-Analysis-style detector", flush=True)

    events = detect_events(t, v, dt, params)

    stem = os.path.splitext(args.abf)[0]
    if args.filter:
        stem += f"_filt{int(args.cutoff_hz)}Hz{int(args.target_rate_hz)}Hz"
    out_csv = f"{stem}_minianalysis_events.csv"
    df = events_frame(events)  # adds inter_event_interval_ms
    df.to_csv(out_csv, index=False)

    # Sidecar with the exact DetectionParams used -- check.py reads this
    # so its click-to-verify view shows the REAL parameters behind these
    # events, not just today's argparse defaults.
    params_path = f"{stem}_minianalysis_params.json"
    with open(params_path, "w") as fh:
        json.dump(asdict(params), fh, indent=2)

    if events:
        rate_hz = len(events) / sweep_len_s
        amps = np.array([e.amplitude for e in events])
        print(f"Detected {len(events)} events ({rate_hz:.3f} Hz), "
              f"mean amplitude {amps.mean():.2f} {abf.adcUnits[args.channel]}, "
              f"median {np.median(amps):.2f}")
    else:
        print("Detected 0 events -- check direction/thresholds.")
    print(f"Saved -> {out_csv}")

    if not args.no_plot:
        fig, ax = plt.subplots(figsize=(16, 4.5))
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        ax.plot(t, v, color=TRACE, lw=0.3, zorder=1)
        if events:
            peak_t = np.array([e.peak_time_s for e in events])
            peak_v = v[np.array([e.peak_idx for e in events])]
            ax.plot(peak_t, peak_v, "x", color="crimson", ms=6, mew=1.2, zorder=3,
                     label=f"{len(events)} detected events")
            ax.legend(loc="upper right", fontsize=9, labelcolor=INK)
        ax.set_xlabel("time (s)", color=INK)
        ax.set_ylabel(f"current ({abf.adcUnits[args.channel]})", color=INK)
        ax.set_title(f"{os.path.basename(args.abf)} -- Mini-Analysis-style detection", color=INK)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        out_png = f"{stem}_minianalysis_trace.png"
        fig.savefig(out_png, dpi=130, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved -> {out_png}")

    # -- Analysis stage (see core.py) ------------------------------------
    if not events and (args.stats or args.histogram_column or args.autocorr
                        or args.cross_corr_csv or args.group_analysis):
        print("No events detected -- skipping requested analysis steps.", flush=True)
        return

    if args.stats:
        rows = {}
        for col in ["amplitude", "area", "rise_time_ms", "decay_time_ms", "inter_event_interval_ms"]:
            rows[col] = vars(column_statistics(df[col]))
        stats_df = pd.DataFrame(rows).T
        stats_df.index.name = "metric"
        out = f"{stem}_minianalysis_stats.csv"
        stats_df.to_csv(out)
        print(f"Saved -> {out}")
        print(stats_df.round(3).to_string())

    if args.histogram_column:
        col = args.histogram_column
        freq = frequency_histogram(df[col], args.hist_bin_size)
        cum = cumulative_histogram(df[col], args.hist_bin_size)
        out_csv2 = f"{stem}_minianalysis_hist_{col}.csv"
        freq.assign(cumulative_fraction=cum["cumulative_fraction"]).to_csv(out_csv2, index=False)
        print(f"Saved -> {out_csv2}")
        if not args.no_plot and len(freq):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
            for ax in (ax1, ax2):
                fig.patch.set_facecolor(SURFACE)
                ax.set_facecolor(SURFACE)
                ax.tick_params(colors=MUTED)
            bin_w = float(freq["bin_center"].diff().median()) if len(freq) > 1 else 1.0
            ax1.bar(freq["bin_center"], freq["count"], width=bin_w, color=TRACE)
            ax1.set_xlabel(col, color=INK)
            ax1.set_ylabel("count", color=INK)
            ax1.set_title("Frequency histogram", color=INK)
            ax2.plot(cum["bin_center"], cum["cumulative_fraction"], color=TRACE)
            ax2.set_xlabel(col, color=INK)
            ax2.set_ylabel("cumulative fraction", color=INK)
            ax2.set_title("Cumulative histogram", color=INK)
            fig.tight_layout()
            out_png2 = f"{stem}_minianalysis_hist_{col}.png"
            fig.savefig(out_png2, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png2}")

    if args.autocorr:
        ac = autocorrelation_histogram(df["peak_time_s"], args.corr_max_lag_ms, args.corr_bin_ms)
        out_csv3 = f"{stem}_minianalysis_autocorr.csv"
        ac.to_csv(out_csv3, index=False)
        print(f"Saved -> {out_csv3}")
        if not args.no_plot and len(ac):
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            ax.bar(ac["lag_center_ms"], ac["count"], width=args.corr_bin_ms, color=TRACE)
            ax.set_xlabel("lag (ms)", color=INK)
            ax.set_ylabel("count", color=INK)
            ax.set_title("Auto-correlation histogram", color=INK)
            ax.tick_params(colors=MUTED)
            fig.tight_layout()
            out_png3 = f"{stem}_minianalysis_autocorr.png"
            fig.savefig(out_png3, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png3}")

    if args.cross_corr_csv:
        other = pd.read_csv(args.cross_corr_csv)
        time_col = "peak_time_s" if "peak_time_s" in other.columns else "location_s"
        cc = cross_correlation_histogram(df["peak_time_s"], other[time_col],
                                          args.corr_max_lag_ms, args.corr_bin_ms)
        out_csv4 = f"{stem}_minianalysis_crosscorr.csv"
        cc.to_csv(out_csv4, index=False)
        print(f"Saved -> {out_csv4}")
        if not args.no_plot and len(cc):
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(SURFACE)
            ax.set_facecolor(SURFACE)
            ax.bar(cc["lag_center_ms"], cc["count"], width=args.corr_bin_ms, color=TRACE)
            ax.set_xlabel("lag, other - this (ms)", color=INK)
            ax.set_ylabel("count", color=INK)
            ax.set_title(f"Cross-correlation vs. {os.path.basename(args.cross_corr_csv)}", color=INK)
            ax.tick_params(colors=MUTED)
            fig.tight_layout()
            out_png4 = f"{stem}_minianalysis_crosscorr.png"
            fig.savefig(out_png4, dpi=130, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved -> {out_png4}")

    if args.group_analysis:
        t_ms, traces, baselines, amplitudes = extract_event_traces(
            v, dt, events, args.group_pre_ms, args.group_post_ms)
        if len(traces) == 0:
            print("Group analysis: no events far enough from the recording edges to extract -- skipped.")
        else:
            scaled = scale_traces(traces, baselines, amplitudes)
            avg_raw, avg_scaled = traces.mean(axis=0), scaled.mean(axis=0)
            print(f"Group analysis: {len(traces)} traces "
                  f"({args.group_pre_ms:.1f}ms before / {args.group_post_ms:.1f}ms after peak)")

            decay_fit = None
            if args.fit_decay is not None:
                pre_n = _samples(args.group_pre_ms, dt)
                t_decay, y_decay = t_ms[pre_n:], avg_raw[pre_n:]
                decay_fit = fit_exponential_decay(
                    t_decay, y_decay, n_exp=args.fit_n_exp, fit_range=args.fit_decay,
                    custom_start_ms=args.fit_custom_start_ms, custom_end_ms=args.fit_custom_end_ms,
                    direction=args.direction)
                summary = (
                    f"Number of traces averaged: {len(traces)}\n"
                    f"# of exponentials: {decay_fit.n_exp}   Fit range: {args.fit_decay} "
                    f"({decay_fit.fit_range_ms[0]:.2f} to {decay_fit.fit_range_ms[1]:.2f} ms post-peak)\n"
                    f"{decay_fit.equation_str()}\n"
                    f"Std deviation: {decay_fit.residual_sd:.4g}   Iterations: {decay_fit.n_iterations}")
                print(summary)
                out_txt = f"{stem}_minianalysis_decayfit.txt"
                with open(out_txt, "w") as fh:
                    fh.write(summary + "\n")
                print(f"Saved -> {out_txt}")

            if not args.no_plot:
                fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
                fig.patch.set_facecolor(SURFACE)
                titles = [["Superimposed", "Scaled Superimposed"], ["Averaged", "Scaled Averaged"]]
                data = [[traces, scaled], [avg_raw[None, :], avg_scaled[None, :]]]
                for r in range(2):
                    for c in range(2):
                        ax = axes[r, c]
                        ax.set_facecolor(SURFACE)
                        for row in data[r][c]:
                            ax.plot(t_ms, row, color=TRACE, lw=(0.3 if r == 0 else 1.2), alpha=(0.3 if r == 0 else 1.0))
                        ax.set_title(titles[r][c], color=INK)
                        ax.tick_params(colors=MUTED)
                        if r == 1:
                            ax.set_xlabel("time from peak (ms)", color=INK)
                if decay_fit is not None:
                    pre_n = _samples(args.group_pre_ms, dt)
                    fit_t = t_ms[pre_n:]
                    axes[1, 0].plot(fit_t, decay_fit.predict(fit_t - fit_t[0]), color="crimson", lw=1.0,
                                     label=f"{decay_fit.n_exp}-exp fit")
                    axes[1, 0].legend(fontsize=8, labelcolor=INK)
                fig.suptitle(f"{os.path.basename(args.abf)} -- group analysis ({len(traces)} events)", color=INK)
                fig.tight_layout()
                out_png5 = f"{stem}_minianalysis_group.png"
                fig.savefig(out_png5, dpi=130, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"Saved -> {out_png5}")


if __name__ == "__main__":
    main()

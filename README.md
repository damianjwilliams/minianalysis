# minianalysis

Synaptic event detection for gap-free voltage-clamp `.abf` recordings
(pCLAMP/Clampex, read with [pyabf](https://pypi.org/project/pyabf/)), using the
classical local-maximum + baseline + amplitude/area-threshold method from the
Synaptosoft **Mini Analysis Program** — plus the two tools you actually need
around a detector: one to *tune* its parameters against your own recording, and
one to *check* every event it found.

Three steps, in the order you'd use them:

| Step | Command | What it does |
|---|---|---|
| **optimize** | `python optimize_params.py rec.abf` | Live three-panel window: edit a parameter, click a peak, see it measured — and see *why* it would be rejected, if it would be |
| **run** | `python run_detection.py rec.abf` | Scan the whole trace, write the events CSV, a sidecar recording the exact parameters used, and a whole-trace plot |
| **check** | `python check_events.py rec.abf` | Click any detected event to see every detection window and threshold drawn on its own trace; accept or reject it |

Everything is self-contained: no other repository, model file, or second
environment is needed.

---

## Install

Python 3.10+ (developed on Windows 11 with 3.11; nothing here is
Windows-specific except the example paths).

```bash
git clone https://github.com/<you>/minianalysis.git
cd minianalysis
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

That's enough to run everything from the repo root. Optionally, to get a
`minianalysis` command on your PATH and be able to import the package from
anywhere:

```bash
pip install -e .
```

The interactive windows need PyQt5 and a real display — they won't run over a
headless SSH session. Detection itself doesn't: `run ... --no-gui` needs only
numpy/scipy/pandas/pyabf/matplotlib, so batch jobs work on a headless machine.

---

## Quick start

```bash
# 1. Tune the parameters against a peak you can see with your own eyes
python optimize_params.py recording.abf

# 2. Detect every event (opens a parameter dialog first; --no-gui skips it)
python run_detection.py recording.abf

# 3. Check what it found, accepting or rejecting each event
python check_events.py recording.abf
```

Prefer one window that does the choosing for you? `python launch.py` asks for a
recording, a step and the filter settings, then runs it — and can open the
checker automatically when the run finishes.

Every step is also a subcommand, if you'd rather type one name:

```bash
python -m minianalysis optimize recording.abf
python -m minianalysis run recording.abf --no-gui --amplitude-threshold 8
python -m minianalysis check recording.abf
python -m minianalysis --help          # all commands
python -m minianalysis run --help      # one command's own options
```

---

## The three steps in detail

### `optimize` — tune the parameters

The Mini Analysis tutorial is explicit that the detection parameters are meant
to be **tuned per recording**, by looking at detected vs. missed events, not
left at someone else's defaults. This is that workflow, live:

- **Left panel** — every detection parameter, editable, read fresh on each
  click. No Apply button.
- **Top right** — the full trace, pan and zoom in both axes (pyqtgraph, so a
  multi-million-sample recording stays responsive).
- **Bottom right** — click any peak above and it's measured *here, right now*,
  with whatever values are currently in the left panel: baseline, amplitude,
  onset, decay, area, all annotated on the event's own trace. If the candidate
  would be rejected, the title says which check it failed — every one of them,
  not just the first.

Edit a value, click the same peak again, and see the difference immediately.
When you're happy, **"Run full detection with these parameters"** commits them
to a real run (same output as `run`, below).

There's also a Filter / resample panel with its own Apply button — filtering a
whole trace is too expensive to redo on every click.

> One deliberate simplification: a single click always uses the plain (d)/(e)
> baseline window average, never the overlapping-event baseline extrapolation,
> because that needs sequential state from scanning the whole trace in order.
> "Run full detection" *does* apply it if enabled, so the two can differ
> slightly on an event that closely follows another.

### `run` — detect every event

Opens the parameter dialog (pre-filled; `--no-gui` skips it), then scans the
trace and writes, next to the `.abf`:

| File | Contents |
|---|---|
| `<stem>_minianalysis_events.csv` | one row per event: `peak_idx`, `peak_time_s`, `baseline`, `amplitude`, `rise_time_ms`, `decay_time_ms`, `area`, `inter_event_interval_ms` |
| `<stem>_minianalysis_params.json` | the exact parameters used — read back by `check` |
| `<stem>_minianalysis_trace.png` | the whole trace with every event marked |

`<stem>` is the `.abf` path without its extension, plus `_filt<cutoff>Hz<rate>Hz`
when `--filter` was used.

Optional analysis flags run the tutorial's downstream analyses on the events
just detected:

```bash
python -m minianalysis run rec.abf --no-gui \
    --stats \                          # n, mean, variance, sd, skewness, kurtosis per column
    --histogram-column amplitude \     # frequency + cumulative histogram
    --autocorr \                       # auto-correlation histogram (periodicity)
    --group-analysis \                 # superimposed/averaged traces, raw and scaled
    --fit-decay peak_to_end            # single/double exponential decay fit (Nelder-Mead simplex)
```

### `check` — verify and curate

Click any event and see, drawn on its own trace, every window and threshold the
detector used to accept it: the baseline window (d)+(e) and the level it
produced, the local-max search span (c), the peak-averaging window, the
amplitude threshold (a), the onset search boundary and crossing, the decay
search window (f) and fraction level (g), and the measured area shaded
pass/fail against (b). A key in the sidebar says what every color means, and
the parameters themselves are listed beside it — read from the sidecar the run
wrote, so what you see is guaranteed to be what actually produced these events.

It doubles as QC. Accept or reject each event (`A`/`R`, or the buttons), or
accept every remaining one at once. Two files are autosaved after *every*
decision, and picked back up automatically next time:

| File | Contents |
|---|---|
| `<stem>_minianalysis_reviewed.csv` | accepted events only |
| `<stem>_minianalysis_review_progress.csv` | every decided event, with `decision` (1 = accepted, 0 = rejected) |

Events you haven't decided on don't appear in the progress file — they aren't
"rejected" by default.

**Controls:** click a red X to inspect it · `←`/`→` or `P`/`N` to step ·
`A`/`↑` accept · `R`/`↓` reject · scroll to zoom the overview (Ctrl = time only,
Shift = current only) · drag to pan · toolbar Home resets the view.

---

## Filtering

All three steps take the same optional preprocessing: a zero-phase Bessel
low-pass (Bessel for its flat group delay, so event shape and timing survive)
applied at the native sampling rate, *then* decimation to a target rate.

```bash
python -m minianalysis run rec.abf --filter --cutoff-hz 3000 --target-rate-hz 10000
```

Before filtering, the recording's own header is checked for the amplifier's
telegraphed hardware low-pass setting, so a redundant re-filter below what was
already applied during acquisition is never silently done without saying so.
Filtered runs also warn if your amplitude threshold is small relative to the
filtered trace's noise floor — thresholds tuned on a raw trace don't transfer,
because filtering correlates adjacent noise samples.

> **The filter settings must match between detecting and checking.** Event
> positions are sample indices into whatever trace the detector saw. Run with
> `--filter` and you must check with the same `--filter` settings, or every
> event lands in the wrong place. `launch.py` passes them through for you.

`preprocess` can also be used on its own to produce a filtered `.npz` plus a
before/after preview plot:

```bash
python -m minianalysis preprocess rec.abf --cutoff-hz 3000 --target-rate-hz 10000
```

---

## The detection parameters

The 9 parameters from the tutorial, plus two this reimplementation adds to pin
down the rise side (the tutorial specifies "time to peak" as a step but not a
formula for it):

| Parameter | Flag | Default | What it does |
|---|---|---|---|
| Amplitude threshold (a) | `--amplitude-threshold` | 5 | minimum \|amplitude\| to accept, in trace units |
| Area threshold (b) | `--area-threshold` | 10 | minimum \|area\| to accept, trace units × ms |
| Peak direction | `--direction` | negative | inward (sEPSC in voltage clamp) or outward |
| Points to average peak | `--n-avg-peak` | 3 | samples averaged to read the peak value |
| Period to search local max (c) | `--search-local-max-ms` | 3 ms | minimum spacing between candidate peaks |
| Time before peak for baseline (d) | `--baseline-before-ms` | 2 ms | gap between the baseline window and the peak |
| Period to average baseline (e) | `--baseline-avg-ms` | 3 ms | length of the baseline window |
| Period to search decay (f) | `--decay-search-ms` | 20 ms | how far past the peak to look for the decay point |
| Fraction to find decay (g) | `--decay-fraction` | 0.5 | fraction of amplitude defining the decay point |
| Onset fraction | `--onset-fraction` | 0.1 | fraction of amplitude defining rise onset |
| Onset search period | `--onset-search-ms` | 5 ms | how far before the peak to look for that crossing |

Two things worth knowing:

- **(c) must be at least comparable to your events' decay duration.** Set it too
  short (say 1 ms against a 3 ms decay) and per-sample noise riding on one
  event's own decaying tail registers as several more "local maxima",
  fragmenting a single event into several spurious detections.
- **(f) and the onset search window are hard limits, not display caps.** A
  candidate whose decay never reaches (g) within (f), or whose rise never
  crosses the onset fraction within the onset window, is *rejected outright* —
  not recorded with a truncated measurement.

There's also `--adjust-overlapping-baseline`, **off by default**: with it off,
the baseline is always the plain (d)/(e) window average, exactly what those two
parameters describe. Turning it on opts into the tutorial's extra step for
closely-spaced events, which replaces that average with a value extrapolated
from the previous event's own fitted exponential decay when the window would
still be contaminated by it.

---

## How detection works

Per the tutorial's 6-step sequence, for every candidate peak:

1. **Find a local maximum** — `scipy.signal.find_peaks` with a minimum spacing
   of (c).
2. **Find a baseline** — mean of the window ending (d) before the peak and
   spanning (e).
3. **Compare amplitude to threshold** — peak (averaged over `n_avg_peak`
   samples) minus baseline, against (a).
4. **Compute time to peak** — walk back from the peak to where the trace first
   crosses `onset_fraction` of the amplitude; reject if it never does within the
   onset search window.
5. **Compute time to decay** — walk forward to where the trace decays to (g) of
   the amplitude; reject if it never does within (f).
6. **Compute area** — baseline-subtracted, onset to decay point (trapezoid),
   against (b).

This is an independent reconstruction from the tutorial's published method
description, not a copy of Synaptosoft's source (which isn't available). The
few details it doesn't specify — the baseline statistic, the exact
onset-crossing definition, the skewness/kurtosis estimator — are reasonable
choices, flagged as such in the code. Everything the tutorial *does* specify is
implemented as described. Page references in the source (`p.7`, `p.13`, …) point
at the tutorial handout's own sections.

---

## Use it as a library

`minianalysis.core` is pure numpy/scipy/pandas — no matplotlib, no Qt — so it
imports cleanly into a script or a notebook:

```python
import pyabf
from minianalysis.core import DetectionParams, detect_events, events_frame

abf = pyabf.ABF("recording.abf")
abf.setSweep(0, channel=0)

params = DetectionParams(amplitude_threshold=8.0, area_threshold=15.0)
events = detect_events(abf.sweepX, abf.sweepY, 1 / abf.dataRate, params)
df = events_frame(events)          # tidy DataFrame + inter-event intervals

print(f"{len(df)} events, {len(df) / abf.sweepLengthSec:.2f} Hz, "
      f"mean amplitude {df.amplitude.mean():.1f} pA")
```

The analysis functions are importable the same way: `column_statistics`,
`frequency_histogram`, `cumulative_histogram`, `running_average`,
`autocorrelation_histogram`, `cross_correlation_histogram`,
`group_by_criteria`, `group_random`, `combine_data_arrays`,
`extract_event_traces`, `scale_traces`, `fit_exponential_decay`.

---

## Layout

```
optimize_params.py      run_detection.py     check_events.py     launch.py
   └─ thin wrappers, so each step is runnable as a plain script from the repo root

minianalysis/
    core.py         the detector + the analysis functions (no matplotlib, no Qt)
    run.py          the batch run: parameter dialog, detection, CSV/sidecar/plot output
    optimize.py     the live three-panel parameter optimizer
    check.py        the click-to-verify + accept/reject window
    preprocess.py   Bessel low-pass + downsample, shared by all three
    gui_utils.py    dialog-field helpers, GUI exception guard
    style.py        plot/GUI color tokens
    cli.py          `python -m minianalysis <command>` dispatch

tests/              synthetic-trace tests: run `pytest` from the repo root
```

## Tests

```bash
pip install pytest
pytest
```

The suite plants events of known amplitude, tau and timing in a synthetic trace
and checks the detector recovers them, that each threshold and search window
rejects what it should, and that the two GUI tools' duplicated per-candidate
math still agrees with `detect_events` exactly. The GUI tests need no display —
they only exercise the math, and skip entirely if PyQt5 isn't installed.

## Credit

The detection method is from the Synaptosoft **Mini Analysis Program** tutorial
by C. Justin Lee, PhD (recovered from archive.org). This is an independent
reimplementation of the published method, not affiliated with Synaptosoft.

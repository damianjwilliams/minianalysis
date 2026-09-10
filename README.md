# minianalysis

Synaptic event detection for gap-free voltage-clamp `.abf` recordings
(pCLAMP/Clampex, read with [pyabf](https://pypi.org/project/pyabf/)) — with
**two independent detectors** and the tools you actually need around them: one
to *tune* parameters against your own recording, and one to *check* every event
that was found.

| Step | Command | What it does |
|---|---|---|
| **optimize** | `python optimize_params.py rec.abf` | Live three-panel window: edit a parameter, click a peak, see it measured — and see *why* it would be rejected, if it would be |
| **run** | `python run_detection.py rec.abf` | The classical detector. Scan the whole trace, write the events CSV, a sidecar recording the exact parameters used, and a whole-trace plot |
| **deconvolve** | `python detect_deconv.py rec.abf` | The other detector. Estimates the event waveform from the recording, deconvolves the trace by it, and detects on the result — no thresholds to tune |
| **check** | `python check_events.py rec.abf` | Click any detected event to see every window and threshold drawn on its own trace; accept or reject it. `--source deconv` reviews the other detector's events |

### Two detectors, and when to use which

**`run`** is the classical method from the Synaptosoft **Mini Analysis
Program**: local maxima, a baseline window before each peak, and
amplitude/area thresholds. It is fully explainable — every accept or reject
traces to a parameter you set — which is exactly why `optimize` and `check`
are built around it.

**`deconvolve`** implements Pernía-Andrade et al. 2012 and targets two
failure modes of the classical method that are structural rather than tuning
problems: `find_peaks` keeps only the tallest peak per search window, so a
small event beside a large one is never a candidate; and the fixed baseline
window lands on the previous event's decay tail at short inter-event
intervals. Deconvolving with the event kernel collapses each event to a sharp
impulse, which separates summating events and removes the decay tail that
otherwise gets reported as several events. On a planted-event benchmark it
scores mean F1 **0.947 vs 0.734** and wins all 22 conditions
(`python detect_deconv.py --benchmark`) — but read the caveats in
[its own section](#deconvolve) before trusting that on your own recordings.

They write separate files and are reviewed separately, so you can run both on
one recording and compare.

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

With the classical detector:

```bash
# 1. Tune the parameters against a peak you can see with your own eyes
python optimize_params.py recording.abf

# 2. Detect every event (opens a parameter dialog first; --no-gui skips it)
python run_detection.py recording.abf

# 3. Check what it found, accepting or rejecting each event
python check_events.py recording.abf
```

Or with the deconvolution detector, which has nothing to tune — so there is no
step 1:

```bash
# 1. Detect (filters at 3 kHz / 10 kHz by default; --no-filter opts out)
python detect_deconv.py recording.abf --plot

# 2. Check what it found
python check_events.py recording.abf --source deconv --filter
```

That `--filter` is not optional bookkeeping: the filter settings go into the
output filename, and event positions are sample indices into whatever trace
the detector saw, so the checker has to rebuild the same one.

Prefer one window that does the choosing for you? `python launch.py` asks for a
recording, a step and the filter settings, then runs it — and after a detection
run it opens the event checker on that detector's events automatically, in a
window that shows the run's output as it happens.

Every step is also a subcommand, if you'd rather type one name:

```bash
python -m minianalysis optimize recording.abf
python -m minianalysis run recording.abf --no-gui --amplitude-threshold 8
python -m minianalysis deconvolve recording.abf
python -m minianalysis deconvolve --benchmark      # the two detectors, head to head
python -m minianalysis check recording.abf --source deconv
python -m minianalysis --help                      # all commands
python -m minianalysis run --help                  # one command's own options
```

---

## The steps in detail

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

`<stem>` is the `.abf` path without its extension, plus `_ch<n>` for any
channel other than 0, plus `_filt<cutoff>Hz<rate>Hz` when `--filter` was used.
Anything that makes two runs describe *different data* — a different channel,
a different filter setting — puts them in different files, so a second run
can't quietly overwrite the first. `check` rebuilds the same stem from the
same flags to find them again.

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

<a id="deconvolve"></a>

### `deconvolve` — detect without thresholds

```bash
python detect_deconv.py recording.abf --plot
python -m minianalysis deconvolve recording.abf --no-filter   # raw trace
```

Three stages, none of which need a parameter tuned by hand:

1. **Estimate the kernel.** Average the well-isolated events in *this*
   recording, normalise to unit peak, and truncate at 8× the 1/e decay. No
   time constant is assumed — this matters more than anything else here.
   Assuming 6 ms on 25 ms data drops precision to 0.51.
2. **Deconvolve.** Wiener deconvolution by that kernel, so each event becomes
   a sharp impulse. Peaks are taken at 6 SD of the *deconvolved* trace.
3. **Clean up.** A matching-pursuit pass accepts candidates largest-first and
   subtracts each accepted kernel from the residual, so a noise bump riding on
   an accepted event's decay tail has no amplitude left and is dropped. Each
   event is then measured on a trace with all *other* events' kernels removed,
   so a neighbour's decay cannot contaminate its baseline.

Every threshold is in units of the recording's own measured noise, so there is
nothing to re-tune per cell.

**Output:** `<stem>_deconv_events.csv` (same 8 columns as the classical
detector, so anything reading one reads the other), `<stem>_deconv_params.json`,
`<stem>_deconv_kernel.csv`, and `<stem>_deconv_trace.png` with `--plot`.

**Preprocessing differs from the other steps.** This one filters at 3 kHz and
resamples to 10 kHz **by default**, matching `preprocess.py`; pass
`--no-filter` to work on the raw trace. The other steps are opt-*in* with
`--filter`.

#### The benchmark, and what it does and doesn't show

```bash
python detect_deconv.py --benchmark        # through the default 3 kHz / 10 kHz
python detect_deconv.py --benchmark-raw    # unfiltered
```

22 planted-event conditions spanning decay τ 3–60 ms, event rate 1–40 Hz,
white and 1/f noise, baseline drift, and amplitudes from SNR 15 down to 3:

| | deconvolution | classical |
|---|---|---|
| mean F1 | **0.947** | 0.734 |
| worst F1 | **0.694** | 0.498 |
| conditions won | **22 / 22** | 0 / 22 |

The classical detector is given its *best* amplitude threshold for each
condition, chosen against the ground truth it would not have in practice, and
still loses every one. Precision is 1.000 in 15 of the 22.

Three things that table does **not** show, all of which matter:

- **The events are biexponential and this detector estimates its kernel from
  the data**, so the benchmark is kinder to it than to a fixed-template
  method. Treat the margin as indicative, not as a measured advantage on your
  recordings.
- **On real recordings it is markedly more conservative.** On one test file it
  found 722 events where the classical detector found 1073–2116 depending on
  threshold. There is no ground truth to adjudicate that — which is what
  `check --source deconv` is for. Review before trusting a count.
- **Slow events are its weak spot.** Kernel estimation averages events that
  nothing else comes within `--kernel-isolation-ms` (60 ms) of. Once decay τ
  exceeds ~25 ms at a normal event rate nothing is genuinely isolated, and the
  estimated kernel length becomes unstable — measured 28–116 ms on the same
  τ = 60 ms data depending only on recording duration, with F1 swinging
  0.80–0.93. It still beats the classical detector throughout that range, but
  raise `--kernel-isolation-ms` (and `--detrend-win-ms` with it) if your events
  are slow. The CLI warns, with a suggested value, when the estimate looks
  unreliable.

The recall floor at SNR ≈ 3 is not a defect: it is the noise-limited detection
limit described by [Greger & Watson
2025](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP288183), who
show that pushing sensitivity below it makes amplitude changes read as
frequency changes. `--min-amp-pa` enforces that limit explicitly if you want it
hard rather than implicit; the run reports the measured σ and the 4σ limit
either way.

---

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

#### Reviewing either detector — `--source`

```bash
python check_events.py rec.abf                              # classical (default)
python check_events.py rec.abf --source deconv --filter     # deconvolution
```

Each detector owns four filenames — events, params, reviewed, progress — and
**no suffix is shared**, so accepting or rejecting one detector's events never
touches the decisions you made on the other. You can run both on one recording
and review each independently.

The window is the same either way, because it rebuilds each event's windows
from the CSV's own `baseline` + `amplitude` columns. What changes is what it
*claims*: the classical detector's amplitude (a), area (b) and local-max (c)
parameters are accept/reject criteria that the deconvolution detector never
applies, so for `--source deconv` they are dropped from the sidebar rather than
shown at their defaults, the `(a)` bracket and `(c)` span aren't drawn, and the
event title reports amplitude/area/rise/decay with **no PASS/FAIL verdict**
against a threshold that was never checked. In their place the sidebar lists
that detector's real criteria, read from its own sidecar — deconvolved-trace
threshold, minimum amplitude in noise SDs, the measured σ and 4σ detection
limit, and the kernel's length and provenance.

**Controls:** click a red X to inspect it · `←`/`→` or `P`/`N` to step ·
`A`/`↑` accept · `R`/`↓` reject · scroll to zoom the overview (Ctrl = time only,
Shift = current only) · drag to pan · toolbar Home resets the view.

---

## Filtering

Every step takes the same preprocessing: a zero-phase Bessel low-pass (Bessel
for its flat group delay, so event shape and timing survive) applied at the
native sampling rate, *then* decimation to a target rate.

One asymmetry to know about. `optimize`, `run` and `check` are opt-**in** —
nothing happens without `--filter`. `deconvolve` filters **by default** at
3 kHz / 10 kHz and takes `--no-filter` to opt out, matching the defaults in
`preprocess.py`. In `launch.py` a single checkbox covers both conventions and
the right flag is emitted for whichever step you picked; picking `deconvolve`
pre-ticks it, so launching it from the window matches running it by hand.

```bash
python -m minianalysis run rec.abf --filter --cutoff-hz 3000 --target-rate-hz 10000
```

Before filtering, the recording's own header is checked for the amplifier's
telegraphed hardware low-pass setting, so a redundant re-filter below what was
already applied during acquisition is never silently done without saying so.
Filtered runs also warn if your amplitude threshold is small relative to the
filtered trace's noise floor — thresholds tuned on a raw trace don't transfer,
because filtering correlates adjacent noise samples.

> **The filter settings (and `--channel`) must match between detecting and
> checking.** Event positions are sample indices into whatever trace the
> detector saw. Run with `--filter` and you must check with the same
> `--filter` settings, or every event lands in the wrong place. In practice
> this is self-enforcing: the settings are part of the output filename, so a
> mismatched `check` reports that it can't find an events CSV rather than
> showing you something wrong. `launch.py` passes them through for you.

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
| Time before peak for baseline (d) | `--baseline-before-ms` | 5 ms | gap between the baseline window and the peak |
| Period to average baseline (e) | `--baseline-avg-ms` | 3 ms | length of the baseline window |
| Period to search decay (f) | `--decay-search-ms` | 20 ms | how far past the peak to look for the decay point |
| Fraction to find decay (g) | `--decay-fraction` | 0.5 | fraction of amplitude defining the decay point |
| Onset fraction | `--onset-fraction` | 0.1 | fraction of amplitude defining rise onset |
| Onset search period | `--onset-search-ms` | 5 ms | how far before the peak to look for that crossing |

Three things worth knowing:

- **(d) has to clear your events' rising phase.** If the baseline window ends
  too close to the peak it measures part of the event itself, which drags the
  baseline toward the peak, under-reads amplitude, and can push a real event
  below (a) entirely. At the old 2 ms default, 18% of events on a test
  recording had a baseline window already >5% of the way into their own rise;
  5 ms recovered 29% more events on that cell. Whether 5 ms is right for yours
  is a question for `optimize` — click a few peaks and look at where the green
  baseline window actually sits.
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

This is the **classical** detector (`run`); for the deconvolution one see
[its own section](#deconvolve).

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

`minianalysis.deconvolve` is the same — numpy/scipy only, no Qt — and returns
the same `Event` objects, so `events_frame` and everything downstream of it
work unchanged:

```python
from minianalysis.deconvolve import (DeconvParams, detect_events_deconv,
                                     noise_sd)

dt = 1 / abf.dataRate
sigma = noise_sd(abf.sweepY, dt)              # from the event-free side
events, diag = detect_events_deconv(
    None, abf.sweepY, dt,
    DeconvParams(min_amp_pa=4 * sigma),        # 4-sigma detection limit, hard
    return_diagnostics=True)

print(f"kernel {diag['kernel'].size * dt * 1e3:.1f} ms "
      f"({diag['kernel_source']}, {diag['n_isolated']} isolated events)")
print(f"{diag['n_candidates']} candidates -> {len(events)} after cleanup")
```

`diag` also carries the deconvolved trace (`deconv`), its noise SD, and the
tracked baseline, which is what you want for plotting or for deciding whether
a threshold is sensible on a given cell. The detector's own scoring helpers
(`_synth`, `score`) are importable too, if you want to benchmark a change
against planted events.

The analysis functions are importable the same way: `column_statistics`,
`frequency_histogram`, `cumulative_histogram`, `running_average`,
`autocorrelation_histogram`, `cross_correlation_histogram`,
`group_by_criteria`, `group_random`, `combine_data_arrays`,
`extract_event_traces`, `scale_traces`, `fit_exponential_decay`.

---

## Layout

```
optimize_params.py   run_detection.py   detect_deconv.py   check_events.py   launch.py
   └─ thin wrappers, so each step is runnable as a plain script from the repo root

minianalysis/
    core.py         the classical detector + the analysis functions (no matplotlib, no Qt)
    deconvolve.py   the deconvolution detector + its benchmark (no Qt)
    run.py          the batch run: parameter dialog, detection, CSV/sidecar/plot output
    optimize.py     the live three-panel parameter optimizer
    check.py        the click-to-verify + accept/reject window, for either detector
    preprocess.py   Bessel low-pass + downsample, shared by every step
    launch.py       the front door: pick a step, run it, then review what it found
    gui_utils.py    dialog-field helpers, GUI exception guard
    style.py        plot/GUI color tokens
    cli.py          `python -m minianalysis <command>` dispatch

tests/
    test_core.py            the classical detector, on planted synthetic events
    test_tools.py           optimize/check math agrees with detect_events exactly
    test_deconvolve.py      the deconvolution detector, incl. overlap and kernel estimation
    test_review_sources.py  reviewing either detector without their files colliding
    test_launch_steps.py    the launcher's argv building and auto-check follow-up
```

## Tests

```bash
pip install pytest
pytest
```

82 tests. The suite plants events of known amplitude, tau and timing in a
synthetic trace and checks each detector recovers them, that every threshold
and search window rejects what it should, that the two GUI tools' duplicated
per-candidate math still agrees with `detect_events` exactly, and that an event
survives the round trip out to CSV and back without moving.

For the deconvolution detector it also checks the cases the classical suite
doesn't cover: two events 2 ms apart (inside the classical detector's own
search window, so it merges them and this one shouldn't), that performance
holds across decay τ without being told τ, that drift doesn't change the
result, and that `noise_sd` doesn't collapse on a filtered trace the way a
`diff`-based estimate does.

For the reviewer and launcher it checks that no filename is shared between the
detectors, that a foreign params sidecar loads without raising, that the
checker opens after a detection run **only** on exit code 0 with an events CSV
actually on disk, and that `deconvolve` gets `--no-filter` where the other
steps get `--filter`. One test opens the real launcher dialog, because the
argv-level tests all passed while a stale tuple unpack in the dialog itself was
broken.

The GUI tests need no display: they drive Qt through its offscreen platform and
skip entirely if PyQt5 isn't installed.

## Credit

The detection method is from the Synaptosoft **Mini Analysis Program** tutorial
by C. Justin Lee, PhD (recovered from archive.org). This is an independent
reimplementation of the published method, not affiliated with Synaptosoft.

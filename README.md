# Sensor Reading Analysis

Desktop tool for extracting propulsion force from load-cell recordings. It
accepts Excel or CSV files, removes slow sensor drift, preserves signed
recovery resistance, detects actuation cycles automatically, and exports
publication-ready SVG plots plus summary spreadsheets.

## Highlights

- FFT-based automatic cycle-frequency detection.
- Explicit zero reference: a pre-actuation rest (tare) is required, and an
  optional post-run rest enables two-point drift correction.
- Cycle-template dynamic baseline that does not use valleys or a fixed
  percentile as the baseline, with a tare-only result reported next to it so
  the influence of the drift model stays visible.
- Signed force preservation: positive values represent propulsion and negative
  values represent recovery resistance.
- Stroke-centred cycles: every cycle window holds exactly one stroke, so no
  peak or recovery valley is counted twice.
- Zero-phase Butterworth low-pass filtering that switches itself off when the
  requested cutoff is at or above the Nyquist frequency.
- Batch analysis for files and folders, with one numeric summary workbook per
  source folder.
- Tkinter desktop UI with a scrollable control panel, status feedback,
  processing log, quick diagnostics, keyboard shortcuts, and an
  **Open Results Folder** action. The analysis runs on a worker thread that
  hands every UI update to the Tk thread.

## Processing pipeline

```text
Raw sensor reading
        |
Pre-actuation tare (and optional post-run rest)
        |
Uniform resampling and FFT frequency detection
        |
Stroke markers and stroke-centred cycle windows
        |
Sensor baseline: two-point tare line or cycle-template drift estimate
        |
Drift removal (signed force retained)
        |
Optional zero-phase low-pass filtering
        |
Cycle peaks, recovery valleys, impulse, and net-force summary
```

The baseline represents the slowly varying additive sensor offset. It is not
estimated from the minimum force, recovery valley, or a 5% percentile, because
those values may be real physical resistance.

### Zero reference and drift models

A periodic recording alone cannot separate a constant sensor offset from a
non-zero mean thrust, so every analysis needs a tare: samples recorded with the
sensor at rest before actuation starts.

- **two-point tare** (used when *Rest after* is set): the baseline is the
  straight line between the pre-actuation tare and the post-run rest. This is
  the recommended protocol: keep the robot still for a few seconds at the
  start and at the end of every recording.
- **cycle template** (used otherwise): every cycle is fitted as
  `offset + gain x template`, and the slowly varying offsets form the
  baseline. This assumes that the sensor offset during the first five cycles
  still equals the tare, and it treats a change of mean force that does not
  scale with the cycle amplitude as drift. `Mean_Cycle_Force_TareOnly(N)`
  reports the same cycles with only the tare subtracted, so the difference
  between the two columns shows how much the result depends on the drift
  model. Neither column can detect drift that happens between the tare and
  the first cycles; only a post-run rest measures it.

The analysis stops with an error instead of reporting numbers when the cycles
do not repeat (for example a recording of noise), or when a rest segment moves
by more than 25% of the cycle amplitude (the rest contains actuation).

## Installation

Python 3.10 or newer is recommended.

```bash
pip install pandas numpy matplotlib scipy openpyxl
```

Legacy `.xls` workbooks additionally need `pip install xlrd`.

## Run the desktop application

```bash
python thrust_analysis.py
```

Typical workflow:

1. Add one or more `.xlsx`, `.xls`, or `.csv` files, or scan folders.
2. Confirm the automatically detected sampling rate and the rest durations
   (*Rest before* `auto`; *Rest after* `0` when the recording has no final
   rest).
3. Keep **Processed Signal (Recommended)** selected for routine reporting.
4. Click **Start Analysis** (scroll the control panel if it does not fit the
   window).
5. Open the generated `thrust_analysis_results` folder from the UI.

Keyboard shortcuts:

- `Ctrl+O`: add a single input file
- `Ctrl+Enter`: start analysis
- `Esc`: stop an active analysis after the current file

## Input format

The input must contain one time column and one force/thrust column. Column
selection is automatic:

- the first column containing `time` is used as time;
- the first column containing `force` or `thrust` is used as force;
- otherwise the first and second columns are used.

`.csv` files are read as CSV; every other extension is read as an Excel
workbook.

Rest segments:

- **Rest before (s) = `auto`**: the block of samples that share the first
  timestamp, as written by the logger before actuation, is the tare. At least
  5 such samples are required; otherwise the analysis stops with an error
  instead of silently using a single sample.
- **Rest before (s) = a number**: every sample recorded within that many
  seconds of the first timestamp is the tare; it must also hold at least 5
  samples. `0` keeps only the first-timestamp samples, which is valid only
  when the recording starts at rest.
- **Rest after (s)**: the last seconds of the recording, at least 5 samples,
  are a post-run rest; `0` means there is none.

## Outputs

For each source folder, the application creates `thrust_analysis_results/`:

- `<file_name>_analysis.svg`: six-panel analysis figure;
- `batch_summary.xlsx`: one numeric row per analyzed file.

The SVG panels show the raw signal and sensor baseline, drift-removed signed
force, low-pass result, removed noise, detected cycles with the mean cycle
force, and peak/recovery-force values.

Key summary columns:

| Column | Meaning |
|---|---|
| `Mean_Cycle_Force(N)` | Time-averaged force over whole cycles: the net thrust. |
| `Std_Cycle_Force(N)` | Sample standard deviation of the per-cycle mean force. |
| `Mean_Cycle_Force_TareOnly(N)` | The same cycles with only the tare subtracted (no drift correction). |
| `Mean_Peak_Thrust(N)` | Mean of the per-cycle stroke peaks. |
| `Mean_Recovery_Force(N)` | Mean of the signed minimum after each stroke. |
| `Mean_Peak_to_Valley(N)` | Mean stroke amplitude (peak minus recovery); not a net force. |
| `Mean_Impulse(N·s)`, `Total_Impulse(N·s)` | Impulse per cycle and over all counted cycles. |
| `Drift_Model` | `two-point tare` or `cycle template`. |
| `Tare(N)`, `Tare_Samples` | The zero reference and the number of samples behind it. |
| `Signal` | Measurement signal and low-pass setting used for peaks and valleys. |

Strokes are located to sub-sample precision, and each cycle runs exactly from
half-way between two strokes to half-way between the next two, so it holds
exactly one stroke peak. Impulse and mean force integrate the linearly
interpolated force between these exact times, so a cycle lasts one stroke
interval even when it spans a non-integer number of samples. Each recovery
valley is the minimum between its stroke and the next one. Only complete
cycles inside the longest run of regular strokes are counted; the first
detected stroke and any start-up transient before it are excluded.

## Measurement modes

**Processed Signal (Recommended)** takes peaks and recovery valleys from the
drift-corrected low-pass signal. It is more stable for noisy recordings.

**Original Signal** takes peaks and recovery valleys from the unfiltered
drift-corrected signal. Use it when preserving small amplitude details is more
important than suppressing noise.

The low-pass cutoff defaults to 5 Hz. Lower values produce smoother curves;
higher values retain more high-frequency detail. A cutoff at or above the
Nyquist frequency (half the sampling rate) switches the filter off, because no
recorded frequency lies above it; at a 5.3 Hz sampling rate, for example, the
default 5 Hz cutoff leaves the signal unfiltered, so both modes give the same
values. A cutoff below five times the cycle frequency trims stroke peaks, and
the processing log warns about it.

Peaks and recovery valleys are sample values. With only about ten samples per
cycle they can miss the true extremes by several percent, so a higher sampling
rate is the reliable fix. Impulse and mean cycle force always use the
unfiltered drift-corrected force and are much less sensitive to sparse
sampling.

## Development checks

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover irregular sampling, repeated timestamps, tare and rest
validation, two-point drift correction, nonlinear drift, changing cycle
amplitude, signed recovery resistance, sparse (about 5 Hz) sampling without
fake drift, non-overlapping cycle impulses, rejection of non-repeating input,
headless SVG generation, the summary table, and the GUI worker-thread flow
including stop and failure reporting. The GUI tests are skipped when no
display is available.

## Repository layout

```text
thrust_analysis.py       Tkinter UI and analysis orchestration
dynamic_baseline.py      FFT, tare, drift models, filtering, cycle analysis
tests/                   Unit and integration contracts
```

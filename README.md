# Sensor Reading Analysis

Desktop tool for extracting propulsion force from load-cell recordings. It
accepts Excel or CSV files, removes slow sensor drift, preserves signed
recovery resistance, detects actuation cycles automatically, and exports
publication-ready SVG plots plus summary spreadsheets.

## Highlights

- FFT-based automatic cycle-frequency detection.
- Cycle-template dynamic baseline that does not use valleys or a fixed
  percentile as the baseline.
- Signed force preservation: positive values represent propulsion and negative
  values represent recovery resistance.
- Zero-phase Butterworth low-pass filtering for stable peak timing.
- Batch analysis for files and folders, with one summary workbook per source
  folder.
- Tkinter desktop UI with status feedback, processing log, quick diagnostics,
  keyboard shortcuts, and an **Open Results Folder** action.

## Processing pipeline

```text
Raw sensor reading
        |
Uniform resampling and FFT frequency detection
        |
Cycle-template dynamic baseline estimation
        |
Drift removal (signed force retained)
        |
Zero-phase low-pass filtering
        |
Cycle peaks, recovery valleys, impulse, and net-force summary
```

The baseline represents the slowly varying additive sensor offset. It is not
estimated from the minimum force, recovery valley, or a 5% percentile, because
those values may be real physical resistance.

## Installation

Python 3.10 or newer is recommended.

```bash
pip install pandas numpy matplotlib scipy openpyxl
```

## Run the desktop application

```bash
python thrust_analysis.py
```

Typical workflow:

1. Add one or more `.xlsx`, `.xls`, or `.csv` files, or scan folders.
2. Confirm the automatically detected sampling rate.
3. Keep **Processed Signal (Recommended)** selected for routine reporting.
4. Click **Start Analysis**.
5. Open the generated `thrust_analysis_results` folder from the UI.

Keyboard shortcuts:

- `Ctrl+O`: add a single input file
- `Ctrl+Enter`: start analysis
- `Esc`: stop an active analysis

## Input format

The input must contain one time column and one force/thrust column. Column
selection is automatic:

- the first column containing `time` is used as time;
- the first column containing `force` or `thrust` is used as force;
- otherwise the first and second columns are used.

The initial repeated-timestamp sample, when present, is used as a pre-actuation
tare reference. This is required because a periodic recording alone cannot
identify the absolute sensor offset.

## Outputs

For each source folder, the application creates `thrust_analysis_results/`:

- `<file_name>_analysis.svg`: six-panel analysis figure;
- `batch_summary.xlsx`: one row per analyzed file.

The SVG panels show the raw signal and dynamic baseline, drift-removed signed
force, low-pass result, removed noise, detected cycles, and peak/recovery-force
values.

The summary workbook includes duration, cycle count, forward/backward cycle
counts, mean thrust, signed recovery force, mean cycle force, net force,
impulse, detected period/frequency, and baseline drift.

## Measurement modes

**Processed Signal (Recommended)** measures peaks and recovery force on the
drift-corrected low-pass signal. It is more stable for noisy recordings.

**Original Signal** measures the unfiltered drift-corrected values at detected
cycle locations. Use it when preserving small amplitude details is more
important than suppressing noise.

The low-pass cutoff defaults to 5 Hz. Lower values produce smoother curves;
higher values retain more high-frequency detail. The cutoff must remain below
the Nyquist frequency.

## Development checks

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover irregular sampling, repeated timestamps, nonlinear drift,
changing cycle amplitude, signed recovery resistance, non-overlapping cycle
impulses, invalid input handling, and headless SVG generation.

## Repository layout

```text
thrust_analysis.py       Tkinter UI and analysis orchestration
dynamic_baseline.py      FFT, dynamic baseline, filtering, cycle analysis
tests/                   Unit and integration contracts
```

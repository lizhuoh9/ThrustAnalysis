# Thrust Analysis

This repository contains a single desktop GUI tool for cyclic thrust-data analysis:

- `thrust_analysis.py`

It is designed for thrust or force signals stored in `.xlsx`, `.xls`, or `.csv` files. The tool automatically estimates sampling rate, removes baseline drift, filters the signal, detects cycles, and exports both plots and summary tables.

## What the Script Does

The script processes each selected file in this order:

1. Load time and force/thrust data.
2. Detect the sampling rate from the time column.
3. Use FFT to estimate the dominant cycle period.
4. Fold complete cycles onto a common phase grid.
5. Separate each cycle into a repeating waveform, changing amplitude, and a
   slowly varying additive sensor offset.
6. Subtract only the additive sensor drift; keep both positive thrust and
   negative recovery resistance.
7. Apply a low-pass filter for peak/valley display.
8. Integrate adjacent peak-to-peak cycles without overlapping intervals.
9. Export plots and a batch summary table.

The dynamic baseline is not calculated from minima, valleys, or a 5th
percentile. The initial repeated-timestamp samples (or the first sample) are
used as a pre-actuation tare reference, and the first stable cycles calibrate
the waveform template. This is important when a recovery stroke creates real
negative force.

## Run

```bash
python thrust_analysis.py
```

## Requirements

Install the required Python packages:

```bash
pip install pandas numpy matplotlib scipy openpyxl
```

## Supported Input Files

Supported formats:

- `.xlsx`
- `.xls`
- `.csv`

The script expects:

- one time column
- one thrust or force column

Automatic column detection rules:

- time column:
  - the first column whose name contains `time`
  - otherwise the first column in the file
- force column:
  - the first column whose name contains `force` or `thrust`
  - otherwise the second column, if present

## Main GUI Parameters

### Input Files

- `Add Single File`
  - add one file to the analysis queue
- `Add Multiple Files`
  - add multiple files at once
- `Add Folders`
  - scan one or more folders and collect all supported data files
- `Clear List`
  - clear the queued files

### Sampling

- `Detected Rate`
  - the detected sampling frequency in Hz
  - calculated from the median time-step in the input file
  - if detection fails, the tool falls back to `100.0 Hz`

Meaning:

- this value controls filtering and FFT period estimation
- if the sampling rate is wrong, the detected period and cycle spacing may also be wrong

### Cycle Detection

- `Detected Period`
  - the dominant cycle period estimated from FFT
  - displayed in seconds

Meaning:

- if the detected period is `1.20 s`, the algorithm assumes one full thrust cycle is roughly 1.20 seconds

### Analysis Options

- `FFT cycle-template detection: enabled`
  - cycle frequency and spacing are determined automatically

- `Processed Signal (Recommended)`
  - uses the filtered, drift-corrected signal to measure cycle peaks and recovery valleys
  - better for noisy data and routine reporting

- `Original Signal`
  - uses unfiltered but drift-corrected values at detected cycle locations
  - useful when you want measured amplitude detail without low-pass smoothing

- `Low-pass Cutoff (Hz)`
  - the cutoff frequency of the low-pass filter applied after baseline removal

Practical meaning:

- lower cutoff:
  - smoother signal
  - stronger noise suppression
  - may remove fast detail
- higher cutoff:
  - keeps more detail
  - may retain more noise

Recommended default:

- `5.0 Hz`

## Exported Results

For each source folder, the tool creates a `thrust_analysis_results` directory containing:

- one SVG figure per analyzed file
- one `batch_summary.xlsx` file per source folder

### Exported Plot

Each file generates:

- `<file_name>_analysis.svg`

The SVG contains six panels:

1. `Raw Data & Dynamic Sensor Baseline`
   - raw force signal and the slowly varying load-cell offset
2. `Drift Removed (Signed Force Preserved)`
   - corrected signal, computed as `force_raw - sensor_baseline`
3. `Low-Pass Filter`
   - debiased signal plus filtered signal
4. `Removed Noise`
   - the difference between debiased and filtered signals
5. `Cycle Detection`
   - filtered signal with thrust peaks and signed recovery valleys marked
6. `Peak Thrust & Signed Recovery Force`
   - bar chart of cycle peak values and recovery values; recovery may be negative

### Exported Summary Table

The batch summary includes these columns:

- `Folder`
- `Folder_Path`
- `File`
- `Duration(s)`
- `Cycles`
- `Forward`
- `Backward`
- `Mean_Thrust(N)`
- `Mean_Recovery_Force(N)`
- `Mean_Cycle_Force(N)`
- `Mean_Net(N)`
- `Std_Net(N)`
- `Mean_Impulse(N*s)`
- `Total_Impulse(N*s)`
- `Detected_Period(s)`
- `Detected_Frequency(Hz)`
- `Baseline_Start(N)`
- `Baseline_End(N)`
- `Baseline_Drift(N)`

## What Each Result Means

Below is the full meaning of the exported results and how each quantity is computed.

### `Folder`

The immediate parent folder name of the source file.

Meaning:

- useful when many datasets from different experiment folders are analyzed together

### `Folder_Path`

The full source folder path of the analyzed file.

Meaning:

- useful when two folders share the same short name
- helps trace every row back to the original dataset location

### `File`

The source filename without extension.

Meaning:

- the dataset identifier used in the plot name and summary row

### `Duration(s)`

Computed from the time axis:

```text
Duration = t[-1]
```

In this script, the exported duration is the last time value after loading and cleaning the file.

Meaning:

- the analyzed recording length in seconds

### `Cycles`

Computed as:

```text
Cycles = len(cycle_results)
```

Meaning:

- the number of complete peak-to-peak cycles detected in the file
- cycles are retained regardless of impulse sign

### `Forward`

Computed as:

```text
Forward = number of cycles whose impulse > 0
```

Meaning:

- the number of cycles whose integrated filtered-signal area is positive
- in most datasets this is a count of cycles producing positive net impulse contribution

### `Backward`

Computed as:

```text
Backward = Cycles - Forward
```

Meaning:

- the number of detected cycles whose integrated impulse is not positive

### `Mean_Thrust(N)`

Computed as:

```text
Mean_Thrust = mean(peak_value for each detected cycle)
```

Where:

- `peak_value` is the cycle peak force
- it comes from the processed signal or original signal depending on the selected measurement mode

Meaning:

- the average peak force level across all detected cycles

### `Mean_Recovery_Force(N)`

Computed as:

```text
Mean_Recovery_Force = mean(recovery_value for each detected cycle)
```

Where:

- `recovery_value` is the signed minimum after sensor-drift removal

Meaning:

- a negative value represents measured recovery resistance
- a value close to zero represents little recovery resistance
- it is never used to define the sensor baseline

### `Mean_Cycle_Force(N)`

For every complete cycle:

```text
cycle_mean_force = cycle_impulse / cycle_duration
```

Meaning:

- the signed time-average force produced over a full cycle
- unlike peak-to-valley amplitude, it includes both thrust and recovery phases

### `Mean_Net(N)`

Computed in two steps.

First, for each cycle:

```text
net_thrust = peak_value - recovery_value
```

Then:

```text
Mean_Net = mean(net_thrust for all detected cycles)
```

Meaning:

- the average net thrust amplitude per cycle
- this is often the most intuitive measure of how strong the cycle is after subtracting the baseline side

### `Std_Net(N)`

Computed as:

```text
Std_Net = std(net_thrust for all detected cycles)
```

Meaning:

- cycle-to-cycle variation of net thrust
- lower values mean the motion is more repeatable
- higher values suggest inconsistent cycle strength or unstable measurements

### `Mean_Impulse(N*s)`

For each cycle, the script computes impulse from the unfiltered,
drift-corrected signal:

```text
impulse = trapz(corrected_force[peak_i:peak_i+1], t[peak_i:peak_i+1])
```

Then:

```text
Mean_Impulse = mean(impulse for all detected cycles)
```

Meaning:

- average time-integrated thrust contribution per cycle
- unlike `Mean_Net`, this depends on both amplitude and time duration

### `Total_Impulse(N*s)`

Computed as:

```text
Total_Impulse = sum(impulse for all detected cycles)
```

Meaning:

- the total integrated force across all complete, non-overlapping cycles

### `Detected_Period(s)`

Computed with FFT after resampling the raw signal onto a uniform time grid and
removing its linear trend:

```text
Detected_Period = 1 / dominant_frequency
```

Meaning:

- the dominant periodic motion time scale in the file
- used internally to estimate cycle spacing and support peak detection

## Important Internal Definitions

### `peak_value`

For each detected cycle, this is the signal value at the cycle peak index.

Depends on measurement mode:

- processed mode:
  - peak value comes from the filtered/debiased signal
- original mode:
  - peak value comes from the unfiltered, drift-corrected signal

### `recovery_value`

For each complete peak-to-peak cycle, this is the minimum signed force after
dynamic sensor-drift removal.

Meaning:

- this is a physical recovery-force measurement, not a baseline estimate

### `net_thrust`

Defined as:

```text
net_thrust = peak_value - recovery_value
```

Meaning:

- the peak-to-valley cycle amplitude

### `impulse`

Defined as the numerical integral of the unfiltered, drift-corrected signal
over the cycle interval:

```text
impulse = trapz(corrected_force over cycle)
```

Meaning:

- area under the drift-corrected force-time curve for that cycle
- a combined measure of force magnitude and duration
- adjacent cycle intervals share an endpoint but no time interval, so their
  impulses sum exactly to the integral over the covered span

## Output Interpretation Tips

If you want to compare datasets:

- use `Mean_Net(N)` to compare typical cycle strength
- use `Std_Net(N)` to compare repeatability
- use `Mean_Impulse(N*s)` to compare average cycle output over time
- use `Total_Impulse(N*s)` to compare whole-recording output
- use `Detected_Period(s)` to compare motion speed

If you want to judge measurement quality:

- inspect the SVG panels for baseline drift and noise
- compare `Mean_Thrust`, `Mean_Recovery_Force`, `Mean_Cycle_Force`, and `Mean_Net`
- watch for large `Std_Net` if the cycles are inconsistent

## Notes

- SVG files are saved in the same source-folder tree under `thrust_analysis_results`
- when multiple folders are selected, each folder receives its own summary file
- absolute mean force and absolute sensor offset are not identifiable from a
  periodic signal alone. The method therefore requires the recording to start
  with a valid pre-actuation tare sample (the file format may repeat its initial
  timestamp).

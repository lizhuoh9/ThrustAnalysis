"""Cycle-aware sensor-drift removal for periodic thrust measurements.

The estimator separates a repeating waveform from a slowly varying additive
load-cell offset.  It never defines the baseline from force minima or a lower
percentile, so a negative recovery stroke remains part of the measured force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt


@dataclass(frozen=True)
class DynamicBaselineResult:
    """Immutable result of cycle-aware drift correction."""

    sampling_rate_hz: float
    frequency_hz: float
    period_s: float
    baseline: np.ndarray
    corrected_force: np.ndarray
    cycle_boundaries: np.ndarray
    cycle_impulses: np.ndarray
    cycle_mean_forces: np.ndarray
    peaks: np.ndarray
    recovery_valleys: np.ndarray
    cycle_gains: np.ndarray
    baseline_anchor_n: float


def _read_only(values: np.ndarray, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _validated_inputs(
    time: np.ndarray | list[float], force: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray, int, float]:
    time_values = np.asarray(time, dtype=float)
    force_values = np.asarray(force, dtype=float)
    if time_values.ndim != 1 or force_values.ndim != 1:
        raise ValueError("time and force must be one-dimensional")
    if time_values.size != force_values.size or time_values.size < 64:
        raise ValueError("time and force must contain at least 64 matching samples")
    if not np.all(np.isfinite(time_values)) or not np.all(np.isfinite(force_values)):
        raise ValueError("time and force must contain only finite values")

    prefix_count = 1
    while (
        prefix_count < time_values.size
        and time_values[prefix_count] == time_values[0]
    ):
        prefix_count += 1

    differences = np.diff(time_values)
    first_active_difference = max(0, prefix_count - 1)
    if np.any(differences[first_active_difference:] <= 0.0):
        raise ValueError(
            "time must increase strictly after an optional repeated initial timestamp"
        )
    positive_differences = differences[differences > 0.0]
    if positive_differences.size < 32:
        raise ValueError("too few positive time intervals")
    sampling_rate_hz = 1.0 / float(np.median(positive_differences))
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
        raise ValueError("unable to determine a valid sampling rate")
    return time_values, force_values, prefix_count, sampling_rate_hz


def _fft_frequency(
    time: np.ndarray,
    force: np.ndarray,
    sampling_rate_hz: float,
    active_start: int,
) -> float:
    active_force = force[active_start:]
    if active_force.size < 64 or np.ptp(active_force) <= 1e-10:
        raise ValueError("signal is too short or has no periodic variation")

    source_time = time[active_start:] - time[active_start]
    uniform_time = np.arange(
        0.0, source_time[-1] + 0.5 / sampling_rate_hz, 1.0 / sampling_rate_hz
    )
    uniform_force = np.interp(uniform_time, source_time, active_force)
    working = detrend(uniform_force, type="linear") * np.hanning(uniform_force.size)
    transform_size = 1 << int(np.ceil(np.log2(max(256, 8 * working.size))))
    spectrum = np.abs(np.fft.rfft(working, n=transform_size)) ** 2
    frequencies = np.fft.rfftfreq(transform_size, d=1.0 / sampling_rate_hz)
    lower_hz = max(0.1, 2.5 / (source_time[-1] - source_time[0]))
    upper_hz = min(10.0, 0.45 * sampling_rate_hz)
    allowed = (frequencies >= lower_hz) & (frequencies <= upper_hz)
    if not np.any(allowed):
        raise ValueError("recording duration does not support FFT cycle detection")
    allowed_indices = np.flatnonzero(allowed)
    peak_index = int(allowed_indices[np.argmax(spectrum[allowed])])

    # Quadratic interpolation reduces FFT-bin quantisation without changing the
    # frequency-domain detection rule.
    fractional_index = float(peak_index)
    if 0 < peak_index < spectrum.size - 1:
        left, centre, right = np.log(spectrum[peak_index - 1 : peak_index + 2] + 1e-30)
        denominator = left - 2.0 * centre + right
        if abs(denominator) > 1e-15:
            fractional_index += float(0.5 * (left - right) / denominator)
    frequency_hz = fractional_index * sampling_rate_hz / transform_size
    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("unable to detect a periodic frequency")
    return frequency_hz


def _cycle_boundaries(
    time: np.ndarray,
    force: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    active_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    nyquist_hz = 0.5 * sampling_rate_hz
    cutoff_hz = min(3.5 * frequency_hz, 0.8 * nyquist_hz)
    normalized_cutoff = max(1e-4, min(0.99, cutoff_hz / nyquist_hz))
    active_time = time[active_start:]
    uniform_time = np.arange(
        active_time[0],
        active_time[-1] + 0.5 / sampling_rate_hz,
        1.0 / sampling_rate_hz,
    )
    uniform_force = np.interp(uniform_time, active_time, force[active_start:])
    smooth_uniform = sosfiltfilt(
        butter(4, normalized_cutoff, btype="low", output="sos"), uniform_force
    )
    expected_samples = sampling_rate_hz / frequency_hz
    prominence = max(1e-10, 0.12 * float(np.std(smooth_uniform)))
    relative_peaks, _ = find_peaks(
        smooth_uniform,
        distance=max(2, int(round(0.65 * expected_samples))),
        prominence=prominence,
    )
    peak_times = uniform_time[relative_peaks]
    insertion_points = np.searchsorted(active_time, peak_times)
    insertion_points = np.clip(insertion_points, 1, active_time.size - 1)
    left = insertion_points - 1
    choose_left = (
        np.abs(peak_times - active_time[left])
        <= np.abs(active_time[insertion_points] - peak_times)
    )
    relative_original = np.where(choose_left, left, insertion_points)
    peaks = np.unique(relative_original + active_start)
    if peaks.size < 4:
        raise ValueError("fewer than three complete cycles were detected")

    periods_s = np.diff(time[peaks])
    expected_period_s = 1.0 / frequency_hz
    valid = (
        (periods_s >= 0.55 * expected_period_s)
        & (periods_s <= 1.55 * expected_period_s)
    )
    best_start = 0
    best_stop = 0
    run_start = 0
    for index, is_valid in enumerate(np.append(valid, False)):
        if not is_valid:
            if index - run_start > best_stop - best_start:
                best_start, best_stop = run_start, index
            run_start = index + 1
    boundaries = peaks[best_start : best_stop + 1]
    if boundaries.size < 4:
        raise ValueError("fewer than three consecutive complete cycles were detected")
    smooth_force = np.array(force, dtype=float, copy=True)
    smooth_force[active_start:] = np.interp(
        active_time, uniform_time, smooth_uniform
    )
    return boundaries.astype(int, copy=False), smooth_force


def _robust_line_fit(template: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    design = np.column_stack((np.ones(template.size), template))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(6):
        residual = values - design @ coefficients
        centred = residual - np.median(residual)
        scale = 1.4826 * np.median(np.abs(centred))
        if scale <= 1e-10 * max(1.0, float(np.ptp(values))):
            break
        ratio = np.abs(residual) / (1.345 * scale)
        weights = np.ones_like(ratio)
        large = ratio > 1.0
        weights[large] = 1.0 / ratio[large]
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_values = values * np.sqrt(weights)
        coefficients = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _fit_slow_baseline(
    time: np.ndarray,
    cycle_times: np.ndarray,
    offsets: np.ndarray,
    anchor_time: float,
    anchor_force: float,
) -> np.ndarray:
    if float(time[-1] - anchor_time) <= 0.0:
        raise ValueError("time duration must be positive")
    if offsets.size < 3:
        raise ValueError("at least three cycle offsets are required")

    window_size = min(5, offsets.size if offsets.size % 2 else offsets.size - 1)
    robust_offsets = median_filter(offsets, size=max(3, window_size), mode="nearest")
    residual = offsets - robust_offsets
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-12
    clipped_offsets = np.clip(offsets, robust_offsets - 3.0 * scale, robust_offsets + 3.0 * scale)
    sigma_cycles = min(2.5, max(1.0, offsets.size / 20.0))
    smooth_offsets = gaussian_filter1d(
        clipped_offsets, sigma=sigma_cycles, mode="nearest"
    )

    node_times = np.concatenate(([anchor_time], cycle_times))
    node_values = np.concatenate(([anchor_force], smooth_offsets))
    return np.interp(time, node_times, node_values)


def lowpass_irregular_signal(
    time: np.ndarray | list[float],
    values: np.ndarray | list[float],
    sampling_rate_hz: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Low-pass an irregularly sampled signal on a uniform time grid."""

    time_values = np.asarray(time, dtype=float)
    signal_values = np.asarray(values, dtype=float)
    if time_values.shape != signal_values.shape or time_values.ndim != 1:
        raise ValueError("time and values must be matching one-dimensional arrays")
    prefix_count = 1
    while prefix_count < time_values.size and time_values[prefix_count] == time_values[0]:
        prefix_count += 1
    source_time = np.concatenate(([time_values[0]], time_values[prefix_count:]))
    source_values = np.concatenate(
        ([float(np.median(signal_values[:prefix_count]))], signal_values[prefix_count:])
    )
    uniform_time = np.arange(
        source_time[0],
        source_time[-1] + 0.5 / sampling_rate_hz,
        1.0 / sampling_rate_hz,
    )
    uniform_values = np.interp(uniform_time, source_time, source_values)
    nyquist_hz = 0.5 * sampling_rate_hz
    normalized_cutoff = min(0.99, max(1e-4, cutoff_hz / nyquist_hz))
    filtered_uniform = sosfiltfilt(
        butter(4, normalized_cutoff, btype="low", output="sos"), uniform_values
    )
    return np.interp(time_values, uniform_time, filtered_uniform)


def analyze_dynamic_baseline(
    time: np.ndarray | list[float], force: np.ndarray | list[float]
) -> DynamicBaselineResult:
    """Remove slow additive sensor drift while retaining the signed force cycle.

    The first sample, or the median of an initial repeated-timestamp block, is
    treated as the pre-actuation load-cell reference.  This explicit reference
    resolves the otherwise unidentifiable split between static sensor offset
    and non-zero mean thrust.
    """

    time_values, force_values, prefix_count, sampling_rate_hz = _validated_inputs(
        time, force
    )
    active_start = prefix_count if prefix_count > 1 else 0
    anchor_force = float(np.median(force_values[:prefix_count]))
    frequency_hz = _fft_frequency(
        time_values, force_values, sampling_rate_hz, active_start
    )
    boundaries, _ = _cycle_boundaries(
        time_values, force_values, sampling_rate_hz, frequency_hz, active_start
    )
    cycle_numbers = np.arange(boundaries.size, dtype=float)
    period_s = float(
        np.polyfit(cycle_numbers, time_values[boundaries], deg=1)[0]
    )
    frequency_hz = 1.0 / period_s

    phase_grid = np.arange(128, dtype=float) / 128.0
    cycle_rows: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        segment_time = time_values[start : stop + 1]
        phase = (segment_time - segment_time[0]) / (segment_time[-1] - segment_time[0])
        cycle_rows.append(np.interp(phase_grid, phase, force_values[start : stop + 1]))
    rows = np.asarray(cycle_rows)
    # Build the waveform from the first stable cycles.  Referencing this
    # waveform to the pre-actuation tare resolves the additive offset without
    # assuming that any cycle minimum is zero force.
    stable_stop = min(6, rows.shape[0])
    stable_start = 1 if stable_stop >= 3 else 0
    stable_rows = rows[stable_start:stable_stop]
    template = np.median(stable_rows, axis=0) - anchor_force
    offsets = np.empty(rows.shape[0], dtype=float)
    gains = np.empty(rows.shape[0], dtype=float)
    for index, row in enumerate(rows):
        offsets[index], gains[index] = _robust_line_fit(template, row)

    stable_gains = gains[stable_start:stable_stop]
    stable_offsets = offsets[stable_start:stable_stop]
    usable = np.abs(stable_gains) > 1e-8
    if not np.any(usable):
        raise ValueError("cycle waveform amplitude is too small for drift separation")
    template_shift = float(
        np.median((stable_offsets[usable] - anchor_force) / stable_gains[usable])
    )
    template = template + template_shift
    for index, row in enumerate(rows):
        offsets[index], gains[index] = _robust_line_fit(template, row)

    cycle_times = 0.5 * (
        time_values[boundaries[:-1]] + time_values[boundaries[1:]]
    )
    fit_start = 1 if offsets.size >= 6 else 0
    baseline = _fit_slow_baseline(
        time_values,
        cycle_times[fit_start:],
        offsets[fit_start:],
        float(time_values[0]),
        anchor_force,
    )
    corrected_force = force_values - baseline

    impulses: list[float] = []
    mean_forces: list[float] = []
    peaks: list[int] = []
    valleys: list[int] = []
    trapezoid = getattr(np, "trapezoid", np.trapz)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        cycle_slice = slice(start, stop + 1)
        cycle_force = corrected_force[cycle_slice]
        extrema_force = corrected_force[start:stop]
        impulse = float(trapezoid(cycle_force, time_values[cycle_slice]))
        duration = float(time_values[stop] - time_values[start])
        impulses.append(impulse)
        mean_forces.append(impulse / duration)
        peaks.append(int(start + np.argmax(extrema_force)))
        valleys.append(int(start + np.argmin(extrema_force)))

    return DynamicBaselineResult(
        sampling_rate_hz=float(sampling_rate_hz),
        frequency_hz=float(frequency_hz),
        period_s=period_s,
        baseline=_read_only(baseline, float),
        corrected_force=_read_only(corrected_force, float),
        cycle_boundaries=_read_only(boundaries, int),
        cycle_impulses=_read_only(np.asarray(impulses), float),
        cycle_mean_forces=_read_only(np.asarray(mean_forces), float),
        peaks=_read_only(np.asarray(peaks), int),
        recovery_valleys=_read_only(np.asarray(valleys), int),
        cycle_gains=_read_only(gains, float),
        baseline_anchor_n=anchor_force,
    )

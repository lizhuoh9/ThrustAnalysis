"""Cycle-aware sensor-drift removal for periodic thrust measurements.

The estimator separates a repeating waveform from a slowly varying additive
load-cell offset.  It never defines the baseline from force minima or a lower
percentile, so a negative recovery stroke remains part of the measured force.

Absolute force needs a zero reference: a pre-actuation tare recorded with the
sensor at rest.  When a post-run rest is also recorded, the drift is the
straight line between the two tares.  Otherwise the drift is estimated from the
cycle template, which assumes that the sensor offset of the first stable cycles
equals the tare; ``tare_only_cycle_mean_forces`` reports the drift-free
alternative so the sensitivity to that assumption stays visible.

Strokes are located to sub-sample precision, and every cycle runs exactly from
half-way between two strokes to half-way between the next two.  Impulses and
the cycle template integrate the linearly interpolated force between these
edges, so a cycle lasts one stroke interval even when it spans a non-integer
number of samples, and every cycle holds exactly one stroke peak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

MIN_TARE_SAMPLES = 5
# A rest segment may not vary by more than this fraction of the cycle amplitude.
MAX_REST_FRACTION_OF_AMPLITUDE = 0.25
# Cycles must correlate at least this well with their median waveform.
MIN_CYCLE_CORRELATION = 0.5
DRIFT_MODEL_TWO_POINT = "two-point tare"
DRIFT_MODEL_CYCLE_TEMPLATE = "cycle template"


@dataclass(frozen=True)
class DynamicBaselineResult:
    """Immutable result of cycle-aware drift correction.

    Cycle ``i`` spans ``cycle_boundary_times[i]`` through
    ``cycle_boundary_times[i + 1]`` seconds and holds the stroke at sample
    ``stroke_markers[i + 1]``; ``cycle_boundaries`` are the samples nearest to
    the cycle edges.
    """

    sampling_rate_hz: float
    frequency_hz: float
    period_s: float
    baseline: np.ndarray
    corrected_force: np.ndarray
    cycle_boundaries: np.ndarray
    cycle_boundary_times: np.ndarray
    stroke_markers: np.ndarray
    cycle_impulses: np.ndarray
    cycle_mean_forces: np.ndarray
    tare_only_cycle_mean_forces: np.ndarray
    peaks: np.ndarray
    recovery_valleys: np.ndarray
    cycle_gains: np.ndarray
    baseline_anchor_n: float
    tare_sample_count: int
    drift_model: str
    post_rest_force_n: float | None = None


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


def _duration_option(value: float, name: str) -> float:
    duration_s = float(value)
    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError(f"{name} must be a finite, non-negative number of seconds")
    return duration_s


def _tare_stop(time: np.ndarray, prefix_count: int, tare_duration_s: float | None) -> int:
    """Return the exclusive end index of the pre-actuation tare segment."""

    if tare_duration_s is None:
        stop = prefix_count
    else:
        duration_s = _duration_option(tare_duration_s, "tare_duration_s")
        if duration_s == 0.0:
            # Explicit opt-in: only the first-timestamp samples, which is valid
            # only when the recording starts at rest.
            return prefix_count
        elapsed = time - time[0]
        stop = max(prefix_count, int(np.searchsorted(elapsed, duration_s, side="left")))
    if stop < MIN_TARE_SAMPLES:
        raise ValueError(
            f"no usable pre-actuation tare: it needs at least {MIN_TARE_SAMPLES} rest "
            f"samples, found {stop}; record the sensor at rest before actuation or "
            "set the tare duration"
        )
    return stop


def _post_rest_start(
    time: np.ndarray, tare_stop: int, post_rest_duration_s: float | None
) -> int:
    """Return the first index of the post-run rest segment, or ``time.size``."""

    if post_rest_duration_s is None:
        return time.size
    duration_s = _duration_option(post_rest_duration_s, "post_rest_duration_s")
    if duration_s == 0.0:
        return time.size
    start = int(np.searchsorted(time, time[-1] - duration_s, side="right"))
    if time.size - start < MIN_TARE_SAMPLES:
        raise ValueError(
            f"the post-run rest needs at least {MIN_TARE_SAMPLES} samples, "
            f"found {time.size - start}"
        )
    if start <= tare_stop:
        raise ValueError("the post-run rest overlaps the pre-actuation tare")
    return start


def _fft_frequency(
    time: np.ndarray,
    force: np.ndarray,
    sampling_rate_hz: float,
    active_start: int,
    active_stop: int,
) -> float:
    active_force = force[active_start:active_stop]
    if active_force.size < 64 or np.ptp(active_force) <= 1e-10:
        raise ValueError("signal is too short or has no periodic variation")

    source_time = time[active_start:active_stop] - time[active_start]
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


def _nearest_samples(sample_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    insertion = np.clip(np.searchsorted(sample_times, query_times), 1, sample_times.size - 1)
    left = insertion - 1
    choose_left = (
        np.abs(query_times - sample_times[left])
        <= np.abs(sample_times[insertion] - query_times)
    )
    return np.where(choose_left, left, insertion)


def _longest_regular_run(marker_times: np.ndarray, expected_period_s: float) -> slice:
    """Return the longest run of markers whose spacing stays near one period."""

    periods_s = np.diff(marker_times)
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
    return slice(best_start, best_stop + 1)


class _CycleGeometry(NamedTuple):
    edge_times: np.ndarray
    boundaries: np.ndarray
    stroke_times: np.ndarray
    markers: np.ndarray


def _refined_peak_times(
    uniform_time: np.ndarray, smooth: np.ndarray, peaks: np.ndarray
) -> np.ndarray:
    """Place each peak at the vertex of the parabola through it and its neighbours."""

    step = float(uniform_time[1] - uniform_time[0])
    shifts = np.zeros(peaks.size)
    inner = (peaks > 0) & (peaks < smooth.size - 1)
    left = smooth[peaks[inner] - 1]
    centre = smooth[peaks[inner]]
    right = smooth[peaks[inner] + 1]
    curvature = left - 2.0 * centre + right
    vertex = np.divide(
        0.5 * (left - right), curvature, out=np.zeros(curvature.size), where=curvature < 0.0
    )
    shifts[inner] = np.clip(vertex, -0.5, 0.5)
    return uniform_time[peaks] + shifts * step


def _stroke_cycles(
    time: np.ndarray,
    force: np.ndarray,
    sampling_rate_hz: float,
    frequency_hz: float,
    active_start: int,
    active_stop: int,
) -> _CycleGeometry:
    """Locate the strokes to sub-sample precision and the cycle edges between them."""

    nyquist_hz = 0.5 * sampling_rate_hz
    cutoff_hz = min(3.5 * frequency_hz, 0.8 * nyquist_hz)
    normalized_cutoff = max(1e-4, min(0.99, cutoff_hz / nyquist_hz))
    active_time = time[active_start:active_stop]
    uniform_time = np.arange(
        active_time[0],
        active_time[-1] + 0.5 / sampling_rate_hz,
        1.0 / sampling_rate_hz,
    )
    uniform_force = np.interp(uniform_time, active_time, force[active_start:active_stop])
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
    stroke_times = _refined_peak_times(uniform_time, smooth_uniform, relative_peaks)
    if stroke_times.size < 5:
        raise ValueError("fewer than three complete cycles were detected")
    stroke_times = stroke_times[_longest_regular_run(stroke_times, 1.0 / frequency_hz)]
    if stroke_times.size < 5:
        raise ValueError("fewer than three consecutive complete cycles were detected")

    # An edge half-way between two strokes has a well-defined phase even when
    # the force is flat there.  Edges are not snapped to samples: at about ten
    # samples per cycle, snapping changes cycle lengths by up to 10 %.
    edge_times = 0.5 * (stroke_times[:-1] + stroke_times[1:])
    markers = _nearest_samples(active_time, stroke_times) + active_start
    boundaries = _nearest_samples(active_time, edge_times) + active_start
    if not (np.all(markers[:-1] < boundaries) and np.all(boundaries < markers[1:])):
        raise ValueError("cycles are too short for the sampling rate")
    return _CycleGeometry(
        edge_times, boundaries.astype(int), stroke_times, markers.astype(int)
    )


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


def _phase_rows(time: np.ndarray, force: np.ndarray, edge_times: np.ndarray) -> np.ndarray:
    """Resample every cycle onto a common 128-point phase grid between its exact edges."""

    phase_grid = np.arange(128, dtype=float) / 128.0
    sample_times = edge_times[:-1, None] + phase_grid[None, :] * np.diff(edge_times)[:, None]
    return np.interp(sample_times, time, force)


def _require_repeatable_cycles(rows: np.ndarray) -> None:
    """Reject recordings whose cycles do not share one waveform, such as pure noise."""

    centred_rows = rows - rows.mean(axis=1, keepdims=True)
    reference = np.median(rows, axis=0)
    centred_reference = reference - reference.mean()
    norms = np.linalg.norm(centred_rows, axis=1) * np.linalg.norm(centred_reference)
    correlations = np.divide(
        centred_rows @ centred_reference, norms, out=np.zeros(rows.shape[0]), where=norms > 0.0
    )
    median_correlation = float(np.median(correlations))
    if median_correlation < MIN_CYCLE_CORRELATION:
        raise ValueError(
            "no repeatable actuation cycle found: the cycles correlate with their median "
            f"waveform at {median_correlation:.2f}, below {MIN_CYCLE_CORRELATION}"
        )


def _require_quiet_rest(
    force: np.ndarray, tare_stop: int, post_start: int, boundaries: np.ndarray
) -> None:
    """Reject rest segments that move like actuation instead of sensor noise."""

    amplitude = float(np.median(
        [np.ptp(force[start : stop + 1]) for start, stop in zip(boundaries[:-1], boundaries[1:])]
    ))
    limit_n = MAX_REST_FRACTION_OF_AMPLITUDE * amplitude
    segments = (("pre-actuation tare", force[:tare_stop], "before"),
                ("post-run rest", force[post_start:], "after"))
    for name, values, side in segments:
        if values.size and np.ptp(values) > limit_n:
            raise ValueError(
                f"the {name} is not at rest: it varies by {np.ptp(values):.4g} N, more than "
                f"{MAX_REST_FRACTION_OF_AMPLITUDE:.0%} of the {amplitude:.4g} N cycle amplitude; "
                f"shorten the rest {side} actuation"
            )


def _fit_cycle_template(rows: np.ndarray, tare_force: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit each cycle as ``offset + gain * template``; return offsets and gains.

    The template is the median of the first five cycles referenced to the
    tare, then shifted so that their median offset equals the tare.  The first
    detected stroke lies outside every cycle window, so start-up transients do
    not enter the template.  This is the cycle-template model's assumption:
    during these first cycles the sensor offset still equals the tare.
    """

    stable_rows = slice(0, min(5, rows.shape[0]))
    template = np.median(rows[stable_rows], axis=0) - tare_force
    first_pass = np.array([_robust_line_fit(template, row) for row in rows])
    stable_offsets = first_pass[stable_rows, 0]
    stable_gains = first_pass[stable_rows, 1]
    usable = np.abs(stable_gains) > 1e-8
    if not np.any(usable):
        raise ValueError("cycle waveform amplitude is too small for drift separation")
    template = template + float(
        np.median((stable_offsets[usable] - tare_force) / stable_gains[usable])
    )
    fits = np.array([_robust_line_fit(template, row) for row in rows])
    return fits[:, 0], fits[:, 1]


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


def _drift_baseline(
    time: np.ndarray,
    force: np.ndarray,
    edge_times: np.ndarray,
    offsets: np.ndarray,
    tare_stop: int,
    post_start: int,
) -> tuple[np.ndarray, str, float | None]:
    """Return the sensor baseline, the drift model used, and the post-run tare."""

    tare_force = float(np.median(force[:tare_stop]))
    tare_time = float(np.mean(time[:tare_stop]))
    if post_start < time.size:
        post_force = float(np.median(force[post_start:]))
        post_time = float(np.mean(time[post_start:]))
        baseline = np.interp(time, [tare_time, post_time], [tare_force, post_force])
        return baseline, DRIFT_MODEL_TWO_POINT, post_force

    cycle_times = 0.5 * (edge_times[:-1] + edge_times[1:])
    baseline = _fit_slow_baseline(time, cycle_times, offsets, tare_time, tare_force)
    return baseline, DRIFT_MODEL_CYCLE_TEMPLATE, None


def _cycle_integrals(
    time: np.ndarray, values: np.ndarray, edge_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return impulse and mean of the linearly interpolated signal over every cycle.

    ``time`` must increase strictly.  Cycle ``i`` runs from ``edge_times[i]`` to
    ``edge_times[i + 1]``, so consecutive cycles share their edge exactly.
    """

    cumulative = np.concatenate(
        ([0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(time)))
    )
    segment = np.clip(np.searchsorted(time, edge_times, side="right") - 1, 0, time.size - 2)
    edge_values = np.interp(edge_times, time, values)
    running = cumulative[segment] + 0.5 * (values[segment] + edge_values) * (
        edge_times - time[segment]
    )
    impulses = np.diff(running)
    return impulses, impulses / np.diff(edge_times)


def _cycle_extrema(
    boundaries: np.ndarray, markers: np.ndarray, signal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    peaks: list[int] = []
    valleys: list[int] = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        peak = int(start + np.argmax(signal[start : stop + 1]))
        next_stroke = int(markers[index + 2])
        peaks.append(peak)
        valleys.append(int(peak + np.argmin(signal[peak:next_stroke])))
    return np.asarray(peaks, dtype=int), np.asarray(valleys, dtype=int)


def cycle_extrema(
    result: DynamicBaselineResult, signal: np.ndarray | list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the stroke-peak and recovery-valley sample indices of every cycle.

    The peak is the maximum of ``signal`` inside the cycle window, which holds
    exactly one stroke.  The recovery valley is the minimum after that peak and
    before the next stroke, so each valley belongs to the stroke before it.
    """

    values = np.asarray(signal, dtype=float)
    if values.shape != result.corrected_force.shape:
        raise ValueError("signal must contain one value per recorded sample")
    return _cycle_extrema(result.cycle_boundaries, result.stroke_markers, values)


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
    time: np.ndarray | list[float],
    force: np.ndarray | list[float],
    *,
    tare_duration_s: float | None = None,
    post_rest_duration_s: float | None = None,
) -> DynamicBaselineResult:
    """Remove slow additive sensor drift while retaining the signed force cycle.

    ``tare_duration_s=None`` uses the logger's block of samples that share the
    first timestamp as the pre-actuation tare.  A number marks every sample
    recorded within that many seconds of the first timestamp as rest.  Both
    need at least ``MIN_TARE_SAMPLES`` samples, except the explicit opt-in
    ``0``, which keeps only the first-timestamp samples.  A positive
    ``post_rest_duration_s`` marks the end of the recording as rest and
    switches the drift model to a straight line between the two tares.
    """

    time_values, force_values, prefix_count, sampling_rate_hz = _validated_inputs(
        time, force
    )
    tare_stop = _tare_stop(time_values, prefix_count, tare_duration_s)
    post_start = _post_rest_start(time_values, tare_stop, post_rest_duration_s)
    tare_force = float(np.median(force_values[:tare_stop]))
    frequency_hz = _fft_frequency(
        time_values, force_values, sampling_rate_hz, tare_stop, post_start
    )
    cycles = _stroke_cycles(
        time_values, force_values, sampling_rate_hz, frequency_hz, tare_stop, post_start
    )
    # After the tare the timestamps increase strictly, as interpolation requires.
    active_time = time_values[tare_stop:]
    rows = _phase_rows(active_time, force_values[tare_stop:], cycles.edge_times)
    _require_repeatable_cycles(rows)
    _require_quiet_rest(force_values, tare_stop, post_start, cycles.boundaries)
    stroke_numbers = np.arange(cycles.stroke_times.size, dtype=float)
    period_s = float(np.polyfit(stroke_numbers, cycles.stroke_times, deg=1)[0])
    offsets, gains = _fit_cycle_template(rows, tare_force)
    baseline, drift_model, post_rest_force = _drift_baseline(
        time_values, force_values, cycles.edge_times, offsets, tare_stop, post_start
    )
    corrected_force = force_values - baseline
    impulses, mean_forces = _cycle_integrals(
        active_time, corrected_force[tare_stop:], cycles.edge_times
    )
    _, tare_only_mean_forces = _cycle_integrals(
        active_time, force_values[tare_stop:] - tare_force, cycles.edge_times
    )
    peaks, valleys = _cycle_extrema(cycles.boundaries, cycles.markers, corrected_force)

    return DynamicBaselineResult(
        sampling_rate_hz=float(sampling_rate_hz),
        frequency_hz=1.0 / period_s,
        period_s=period_s,
        baseline=_read_only(baseline, float),
        corrected_force=_read_only(corrected_force, float),
        cycle_boundaries=_read_only(cycles.boundaries, int),
        cycle_boundary_times=_read_only(cycles.edge_times, float),
        stroke_markers=_read_only(cycles.markers, int),
        cycle_impulses=_read_only(impulses, float),
        cycle_mean_forces=_read_only(mean_forces, float),
        tare_only_cycle_mean_forces=_read_only(tare_only_mean_forces, float),
        peaks=_read_only(peaks, int),
        recovery_valleys=_read_only(valleys, int),
        cycle_gains=_read_only(gains, float),
        baseline_anchor_n=tare_force,
        tare_sample_count=int(tare_stop),
        drift_model=drift_model,
        post_rest_force_n=post_rest_force,
    )

"""Synthetic recordings shared by the analysis contract tests."""

from __future__ import annotations

import numpy as np

# The squid-robot logger samples at about 5.33 Hz while the robot actuates at
# about 0.5 Hz, i.e. roughly eleven samples per cycle.
SPARSE_SAMPLING_RATE_HZ = 5.333
SPARSE_FREQUENCY_HZ = 0.4992
REST_FORCE_N = 0.86


def _phase_offset(phase: np.ndarray, centre: float) -> np.ndarray:
    """Signed distance from ``centre`` on the unit phase circle."""

    return np.mod(phase - centre + 0.5, 1.0) - 0.5


def jet_cycle_force(time: np.ndarray, frequency_hz: float) -> np.ndarray:
    """One narrow jet stroke per cycle, a recovery dip after it, and a refill shoulder."""

    phase = np.mod(frequency_hz * np.asarray(time, dtype=float), 1.0)
    stroke = 0.45 * np.exp(-((_phase_offset(phase, 0.30) / 0.12) ** 2))
    recovery = -0.08 * np.exp(-((_phase_offset(phase, 0.62) / 0.08) ** 2))
    shoulder = 0.10 * np.exp(-((_phase_offset(phase, 0.05) / 0.10) ** 2))
    return stroke + recovery + shoulder


def jet_cycle_mean_force(frequency_hz: float) -> float:
    """Exact time-averaged force of :func:`jet_cycle_force` over one cycle."""

    phase_time = np.arange(20000, dtype=float) / (20000.0 * frequency_hz)
    return float(np.mean(jet_cycle_force(phase_time, frequency_hz)))


def sparse_logger_recording(
    duration_s: float = 60.0, tare_count: int = 24, noise_n: float = 0.004, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """Recording shaped like the logger output: a repeated-timestamp rest block, then strokes."""

    rng = np.random.default_rng(seed)
    active_time = np.arange(1, int(duration_s * SPARSE_SAMPLING_RATE_HZ) + 1) / SPARSE_SAMPLING_RATE_HZ
    time = np.concatenate((np.zeros(tare_count), active_time))
    rest = REST_FORCE_N + noise_n * rng.standard_normal(tare_count)
    active = REST_FORCE_N + jet_cycle_force(active_time, SPARSE_FREQUENCY_HZ)
    active = active + noise_n * rng.standard_normal(active_time.size)
    return time, np.concatenate((rest, active))

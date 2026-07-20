"""Behavioral contract for cycle-aware dynamic baseline correction.

The public API under test is::

    analyze_dynamic_baseline(time, force) -> DynamicBaselineResult

``cycle_boundaries``, ``peaks``, and ``recovery_valleys`` are sample indices.
Adjacent boundary indices describe closed integration endpoints: cycle ``i``
spans ``boundary[i]`` through ``boundary[i + 1]``.  This convention makes
each sample interval belong to exactly one cycle and prevents double-counting
impulse when per-cycle values are summed.
"""

from __future__ import annotations

import dataclasses
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _positive_pulse(time: np.ndarray, frequency_hz: float) -> np.ndarray:
    """Return a smooth, positive-only pulse with one pulse per cycle."""

    phase_signal = np.sin(2.0 * np.pi * frequency_hz * time)
    return np.maximum(phase_signal, 0.0) ** 2


def _trapz(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Use NumPy's current trapezoid spelling while supporting older NumPy."""

    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, coordinates))


class DynamicBaselineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dynamic_baseline = importlib.import_module("dynamic_baseline")
        cls.analyze = staticmethod(dynamic_baseline.analyze_dynamic_baseline)

    def assert_read_only(self, value: object) -> None:
        """Assert that a returned vector cannot be changed in place."""

        with self.assertRaises((AttributeError, TypeError, ValueError)):
            value[0] = value[0]  # type: ignore[index]

    def test_positive_periodic_thrust_separates_nonlinear_slow_drift(self) -> None:
        sampling_rate_hz = 200.0
        frequency_hz = 2.0
        time = np.arange(0.0, 20.0, 1.0 / sampling_rate_hz)
        true_baseline = 0.22 + 0.012 * time + 0.0009 * time**2
        true_thrust = 0.80 * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, true_baseline + true_thrust)

        self.assertTrue(dataclasses.is_dataclass(result))
        for field_name in (
            "sampling_rate_hz",
            "frequency_hz",
            "period_s",
            "baseline",
            "corrected_force",
            "cycle_boundaries",
            "cycle_impulses",
            "cycle_mean_forces",
            "peaks",
            "recovery_valleys",
        ):
            self.assertTrue(hasattr(result, field_name), field_name)

        self.assertAlmostEqual(result.sampling_rate_hz, sampling_rate_hz, delta=0.5)
        self.assertAlmostEqual(result.frequency_hz, frequency_hz, delta=0.04)
        self.assertAlmostEqual(result.period_s, 1.0 / frequency_hz, delta=0.01)

        baseline = np.asarray(result.baseline, dtype=float)
        corrected = np.asarray(result.corrected_force, dtype=float)
        self.assertEqual(baseline.shape, time.shape)
        self.assertEqual(corrected.shape, time.shape)

        interior = slice(int(0.10 * time.size), int(0.90 * time.size))
        baseline_rmse = float(
            np.sqrt(np.mean((baseline[interior] - true_baseline[interior]) ** 2))
        )
        corrected_rmse = float(
            np.sqrt(np.mean((corrected[interior] - true_thrust[interior]) ** 2))
        )
        self.assertLess(baseline_rmse, 0.04)
        self.assertLess(corrected_rmse, 0.05)

        with self.assertRaises((AttributeError, dataclasses.FrozenInstanceError)):
            result.frequency_hz = 99.0
        self.assert_read_only(result.baseline)
        self.assert_read_only(result.corrected_force)

    def test_bipolar_recovery_drag_is_preserved_as_signal(self) -> None:
        sampling_rate_hz = 240.0
        frequency_hz = 1.5
        time = np.arange(0.0, 16.0, 1.0 / sampling_rate_hz)
        sine = np.sin(2.0 * np.pi * frequency_hz * time)
        true_signal = 0.72 * np.maximum(sine, 0.0) ** 2
        true_signal -= 0.18 * np.maximum(-sine, 0.0) ** 2
        true_baseline = 0.30 + 0.025 * np.sin(2.0 * np.pi * 0.035 * time)

        result = self.analyze(time, true_baseline + true_signal)

        corrected = np.asarray(result.corrected_force, dtype=float)
        valley_indices = np.asarray(result.recovery_valleys, dtype=int)
        self.assertGreaterEqual(valley_indices.size, 10)
        self.assertTrue(np.all((0 <= valley_indices) & (valley_indices < time.size)))
        self.assertLess(float(np.median(corrected[valley_indices])), -0.12)

        recovery_region = sine < -0.8
        recovery_error = float(
            np.sqrt(np.mean((corrected[recovery_region] - true_signal[recovery_region]) ** 2))
        )
        self.assertLess(recovery_error, 0.05)

    def test_changing_cycle_amplitude_is_not_absorbed_into_additive_baseline(self) -> None:
        sampling_rate_hz = 200.0
        frequency_hz = 2.0
        time = np.arange(0.0, 15.0, 1.0 / sampling_rate_hz)
        amplitude = 0.35 + 0.75 * time / time[-1]
        true_baseline = np.full_like(time, 0.27)
        true_signal = amplitude * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, true_baseline + true_signal)

        baseline = np.asarray(result.baseline, dtype=float)
        corrected = np.asarray(result.corrected_force, dtype=float)
        peak_indices = np.asarray(result.peaks, dtype=int)
        self.assertGreaterEqual(peak_indices.size, 20)

        interior = slice(int(0.10 * time.size), int(0.90 * time.size))
        self.assertLess(float(np.max(np.abs(baseline[interior] - true_baseline[interior]))), 0.06)

        peak_values = corrected[peak_indices]
        third = max(2, peak_values.size // 3)
        early_peak = float(np.median(peak_values[:third]))
        late_peak = float(np.median(peak_values[-third:]))
        self.assertGreater(late_peak, 1.8 * early_peak)
        self.assertGreater(float(np.corrcoef(peak_values, amplitude[peak_indices])[0, 1]), 0.95)

    def test_repeated_initial_timestamps_are_pre_actuation_zero_samples(self) -> None:
        sampling_rate_hz = 120.0
        repeated_zero_count = 6
        active_time = np.arange(1, 1201, dtype=float) / sampling_rate_hz
        time = np.concatenate((np.zeros(repeated_zero_count), active_time))
        preload = 0.41
        force = np.concatenate(
            (
                np.full(repeated_zero_count, preload),
                preload + 0.55 * _positive_pulse(active_time, 2.0),
            )
        )

        result = self.analyze(time, force)

        corrected = np.asarray(result.corrected_force, dtype=float)
        self.assertAlmostEqual(result.sampling_rate_hz, sampling_rate_hz, delta=0.5)
        self.assertEqual(corrected.shape, force.shape)
        self.assertLess(float(np.max(np.abs(corrected[:repeated_zero_count]))), 0.03)
        self.assertGreater(float(np.max(corrected[repeated_zero_count:])), 0.48)

    def test_cycle_impulses_sum_without_overlapping_intervals(self) -> None:
        sampling_rate_hz = 250.0
        frequency_hz = 2.5
        time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
        force = 0.19 + 0.64 * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, force)

        boundaries = np.asarray(result.cycle_boundaries)
        impulses = np.asarray(result.cycle_impulses, dtype=float)
        corrected = np.asarray(result.corrected_force, dtype=float)
        self.assertTrue(np.issubdtype(boundaries.dtype, np.integer))
        boundaries = boundaries.astype(int, copy=False)
        self.assertGreaterEqual(boundaries.size, 20)
        self.assertTrue(np.all(np.diff(boundaries) > 0))
        self.assertGreaterEqual(int(boundaries[0]), 0)
        self.assertLess(int(boundaries[-1]), time.size)
        self.assertEqual(impulses.size, boundaries.size - 1)

        reconstructed = np.asarray(
            [
                _trapz(corrected[start : stop + 1], time[start : stop + 1])
                for start, stop in zip(boundaries[:-1], boundaries[1:])
            ]
        )
        np.testing.assert_allclose(impulses, reconstructed, rtol=1e-6, atol=1e-9)
        full_span_impulse = _trapz(
            corrected[boundaries[0] : boundaries[-1] + 1],
            time[boundaries[0] : boundaries[-1] + 1],
        )
        self.assertAlmostEqual(float(np.sum(impulses)), full_span_impulse, places=8)

        mean_forces = np.asarray(result.cycle_mean_forces, dtype=float)
        durations = time[boundaries[1:]] - time[boundaries[:-1]]
        np.testing.assert_allclose(mean_forces, impulses / durations, rtol=1e-6, atol=1e-9)

    def test_invalid_and_too_short_inputs_raise_value_error(self) -> None:
        valid_time = np.arange(0.0, 2.0, 0.01)
        valid_force = np.sin(2.0 * np.pi * 2.0 * valid_time)
        invalid_inputs = (
            ([], []),
            (valid_time, valid_force[:-1]),
            (valid_time, np.where(np.arange(valid_force.size) == 20, np.nan, valid_force)),
            ([0.0, 0.01, 0.01, 0.02, 0.03, 0.04], [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            (np.arange(12, dtype=float) / 100.0, np.sin(np.arange(12, dtype=float))),
        )

        for time, force in invalid_inputs:
            with self.subTest(sample_count=len(time)):
                with self.assertRaises(ValueError):
                    self.analyze(time, force)


class DynamicBaselineHeadlessFlowTests(unittest.TestCase):
    def test_analysis_result_can_be_rendered_to_headless_svg(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        analyze = importlib.import_module("dynamic_baseline").analyze_dynamic_baseline
        time = np.arange(0.0, 8.0, 1.0 / 160.0)
        force = 0.16 + 0.02 * np.cos(2.0 * np.pi * 0.08 * time)
        force += 0.58 * _positive_pulse(time, 2.0)

        result = analyze(time, force)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "dynamic-baseline.svg"
            figure, axis = plt.subplots(figsize=(7.0, 3.5))
            axis.plot(time, force, label="raw")
            axis.plot(time, result.baseline, label="dynamic baseline")
            axis.plot(time, result.corrected_force, label="corrected")
            axis.legend()
            figure.savefig(output_path, format="svg")
            plt.close(figure)

            self.assertTrue(output_path.is_file())
            svg_text = output_path.read_text(encoding="utf-8")
            self.assertIn("<svg", svg_text)
            self.assertIn("dynamic baseline", svg_text)


if __name__ == "__main__":
    unittest.main()

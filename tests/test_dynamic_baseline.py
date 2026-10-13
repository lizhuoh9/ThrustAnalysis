"""Behavioral contract for cycle-aware dynamic baseline correction.

The public API under test is::

    analyze_dynamic_baseline(time, force) -> DynamicBaselineResult

``cycle_boundary_times`` are the exact cycle edges in seconds; cycle ``i``
spans ``cycle_boundary_times[i]`` through ``cycle_boundary_times[i + 1]``.
Adjacent cycles share an edge, so every instant belongs to exactly one cycle
and summing per-cycle impulses never double-counts.  ``cycle_boundaries``,
``peaks``, and ``recovery_valleys`` are sample indices.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.synthetic_signals import (
    REST_FORCE_N,
    SPARSE_FREQUENCY_HZ,
    SPARSE_SAMPLING_RATE_HZ,
    jet_cycle_force,
    jet_cycle_mean_force,
    sparse_logger_recording,
)


def _positive_pulse(time: np.ndarray, frequency_hz: float) -> np.ndarray:
    """Return a smooth, positive-only pulse with one pulse per cycle."""

    phase_signal = np.sin(2.0 * np.pi * frequency_hz * time)
    return np.maximum(phase_signal, 0.0) ** 2


def _trapz(values: np.ndarray, coordinates: np.ndarray) -> float:
    """Use NumPy's current trapezoid spelling while supporting older NumPy."""

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(values, coordinates))


class DynamicBaselineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dynamic_baseline = importlib.import_module("dynamic_baseline")
        # These synthetic recordings start exactly at zero force, so their first
        # timestamp is a valid tare.  Real recordings must provide a rest block.
        cls.analyze = staticmethod(
            functools.partial(dynamic_baseline.analyze_dynamic_baseline, tare_duration_s=0.0)
        )

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

    def test_local_nonpolynomial_drift_is_followed_on_a_slow_cycle_scale(self) -> None:
        sampling_rate_hz = 80.0
        frequency_hz = 1.25
        time = np.arange(0.0, 40.0, 1.0 / sampling_rate_hz)
        bump = 0.030 * np.exp(-((time - 25.0) / 3.5) ** 2)
        true_baseline = 0.21 + 0.018 * np.sin(2.0 * np.pi * 0.018 * time) + bump
        true_signal = 0.42 * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, true_baseline + true_signal)

        interior = slice(int(0.10 * time.size), int(0.90 * time.size))
        error = np.asarray(result.baseline)[interior] - true_baseline[interior]
        self.assertLess(float(np.sqrt(np.mean(error**2))), 0.007)

    def test_irregular_sampling_is_resampled_before_cycle_detection(self) -> None:
        nominal_rate_hz = 20.0
        frequency_hz = 1.25
        steps = np.concatenate((np.full(400, 0.025), np.full(400, 0.075)))
        time = np.concatenate(([0.0], np.cumsum(steps)))
        waveform = 0.46 * _positive_pulse(time, frequency_hz)
        baseline = 0.18 + 0.0008 * time

        result = self.analyze(time, baseline + waveform)

        self.assertAlmostEqual(result.frequency_hz, frequency_hz, delta=0.03)
        periods = np.diff(time[np.asarray(result.cycle_boundaries, dtype=int)])
        self.assertLess(float(np.std(periods)), 0.08)
        covered_duration = time[result.cycle_boundaries[-1]] - time[result.cycle_boundaries[0]]
        self.assertGreater(float(covered_duration), 0.80 * float(time[-1] - time[0]))

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

    def test_noisy_repeated_timestamp_block_is_never_detected_as_cycles(self) -> None:
        sampling_rate_hz = 6.0
        repeated_zero_count = 30
        active_time = np.arange(1, 361, dtype=float) / sampling_rate_hz
        time = np.concatenate((np.zeros(repeated_zero_count), active_time))
        prefix_noise = 0.22 + 0.004 * np.sin(np.arange(repeated_zero_count))
        active_force = 0.22 + 0.08 * _positive_pulse(active_time, 5.0 / 6.0)
        force = np.concatenate((prefix_noise, active_force))

        result = self.analyze(time, force)

        boundaries = np.asarray(result.cycle_boundaries, dtype=int)
        self.assertGreaterEqual(int(boundaries[0]), repeated_zero_count)
        self.assertAlmostEqual(result.frequency_hz, 5.0 / 6.0, delta=0.03)

    def test_cycle_impulses_sum_without_overlapping_intervals(self) -> None:
        sampling_rate_hz = 250.0
        frequency_hz = 2.5
        time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
        force = 0.19 + 0.64 * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, force)

        boundaries = np.asarray(result.cycle_boundaries)
        edges = np.asarray(result.cycle_boundary_times, dtype=float)
        impulses = np.asarray(result.cycle_impulses, dtype=float)
        corrected = np.asarray(result.corrected_force, dtype=float)
        self.assertTrue(np.issubdtype(boundaries.dtype, np.integer))
        boundaries = boundaries.astype(int, copy=False)
        self.assertGreaterEqual(boundaries.size, 20)
        self.assertTrue(np.all(np.diff(boundaries) > 0))
        self.assertGreaterEqual(int(boundaries[0]), 0)
        self.assertLess(int(boundaries[-1]), time.size)
        self.assertEqual(edges.shape, boundaries.shape)
        self.assertTrue(np.all(np.diff(edges) > 0))
        self.assertTrue(np.all(np.abs(time[boundaries] - edges) <= 0.5 / sampling_rate_hz + 1e-12))
        self.assertEqual(impulses.size, edges.size - 1)

        # Integrate the linearly interpolated force on a dense grid per cycle.
        reconstructed = []
        for start, stop in zip(edges[:-1], edges[1:]):
            dense_time = np.linspace(start, stop, 4001)
            reconstructed.append(_trapz(np.interp(dense_time, time, corrected), dense_time))
        np.testing.assert_allclose(impulses, reconstructed, rtol=1e-5, atol=1e-8)
        dense_span = np.linspace(edges[0], edges[-1], 400001)
        full_span_impulse = _trapz(np.interp(dense_span, time, corrected), dense_span)
        self.assertAlmostEqual(float(np.sum(impulses)), full_span_impulse, places=6)

        mean_forces = np.asarray(result.cycle_mean_forces, dtype=float)
        np.testing.assert_allclose(mean_forces, impulses / np.diff(edges), rtol=1e-9, atol=1e-12)

    def test_each_boundary_peak_belongs_to_only_one_cycle(self) -> None:
        sampling_rate_hz = 200.0
        frequency_hz = 2.0
        time = np.arange(0.0, 12.0, 1.0 / sampling_rate_hz)
        cycle_number = np.floor(frequency_hz * time).astype(int)
        amplitude = np.where(cycle_number % 2 == 0, 0.30, 0.80)
        force = 0.20 + amplitude * _positive_pulse(time, frequency_hz)

        result = self.analyze(time, force)

        peak_indices = np.asarray(result.peaks, dtype=int)
        boundaries = np.asarray(result.cycle_boundaries, dtype=int)
        self.assertEqual(np.unique(peak_indices).size, peak_indices.size)
        self.assertTrue(np.all(boundaries[:-1] < peak_indices))
        self.assertTrue(np.all(peak_indices < boundaries[1:]))
        # Each peak sits on the apex of its own pulse, at t = (k + 1/4) / f.
        pulse_phase = np.mod(time[peak_indices] * frequency_hz - 0.25, 1.0)
        self.assertTrue(np.all(np.minimum(pulse_phase, 1.0 - pulse_phase) < 0.02))

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


class TareAndDriftModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analyze = staticmethod(
            importlib.import_module("dynamic_baseline").analyze_dynamic_baseline
        )

    def test_missing_pre_actuation_tare_is_an_error_in_auto_mode(self) -> None:
        time = np.arange(0.0, 20.0, 0.01)
        force = 0.30 + _positive_pulse(time, 1.0)

        with self.assertRaisesRegex(ValueError, "tare"):
            self.analyze(time, force)

    def test_too_short_repeated_block_is_not_accepted_as_tare(self) -> None:
        active_time = np.arange(1, 2001, dtype=float) / 100.0
        time = np.concatenate((np.zeros(3), active_time))
        force = np.concatenate((np.full(3, 0.30), 0.30 + _positive_pulse(active_time, 1.0)))

        with self.assertRaisesRegex(ValueError, "tare"):
            self.analyze(time, force)

    def test_explicit_rest_duration_defines_the_tare(self) -> None:
        sampling_rate_hz = 50.0
        rest_s = 3.0
        time = np.arange(0.0, 33.0, 1.0 / sampling_rate_hz)
        waveform = np.where(time >= rest_s, jet_cycle_force(time - rest_s, 1.0), 0.0)

        result = self.analyze(time, REST_FORCE_N + waveform, tare_duration_s=rest_s)

        self.assertAlmostEqual(result.baseline_anchor_n, REST_FORCE_N, places=9)
        self.assertEqual(result.tare_sample_count, int(rest_s * sampling_rate_hz))
        self.assertGreaterEqual(float(time[result.cycle_boundaries[0]]), rest_s)
        self.assertAlmostEqual(
            float(np.mean(result.cycle_mean_forces)), jet_cycle_mean_force(1.0), delta=0.004
        )

    def test_post_run_rest_enables_two_point_drift_correction(self) -> None:
        sampling_rate_hz = 50.0
        repeated_count = 10
        active_time = np.arange(1, 2001, dtype=float) / sampling_rate_hz
        true_drift = REST_FORCE_N + 0.0015 * active_time
        waveform = np.where(active_time < 35.0, jet_cycle_force(active_time, 1.0), 0.0)
        time = np.concatenate((np.zeros(repeated_count), active_time))
        force = np.concatenate((np.full(repeated_count, REST_FORCE_N), true_drift + waveform))

        result = self.analyze(time, force, post_rest_duration_s=4.0)

        self.assertEqual(result.drift_model, "two-point tare")
        self.assertAlmostEqual(
            result.post_rest_force_n, float(np.median(true_drift[active_time > 36.0])), delta=1e-6
        )
        interior = (active_time > 2.0) & (active_time < 33.0)
        baseline_error = np.asarray(result.baseline)[repeated_count:][interior] - true_drift[interior]
        self.assertLess(float(np.max(np.abs(baseline_error))), 0.004)
        self.assertLess(float(time[result.cycle_boundaries[-1]]), 35.5)
        self.assertAlmostEqual(
            float(np.mean(result.cycle_mean_forces)), jet_cycle_mean_force(1.0), delta=0.003
        )

    def test_tare_only_mean_force_keeps_the_uncorrected_drift(self) -> None:
        sampling_rate_hz = 50.0
        repeated_count = 10
        active_time = np.arange(1, 2001, dtype=float) / sampling_rate_hz
        true_drift = REST_FORCE_N + 0.0015 * active_time
        waveform = np.where(active_time < 35.0, jet_cycle_force(active_time, 1.0), 0.0)
        time = np.concatenate((np.zeros(repeated_count), active_time))
        force = np.concatenate((np.full(repeated_count, REST_FORCE_N), true_drift + waveform))

        result = self.analyze(time, force, post_rest_duration_s=4.0)

        edges = np.asarray(result.cycle_boundary_times)
        cycle_centres = 0.5 * (edges[:-1] + edges[1:])
        expected_drift_n = 0.0015 * cycle_centres
        self.assertEqual(result.tare_only_cycle_mean_forces.shape, result.cycle_mean_forces.shape)
        np.testing.assert_allclose(
            result.tare_only_cycle_mean_forces - result.cycle_mean_forces,
            expected_drift_n,
            atol=0.002,
        )

    def test_drift_free_recording_gives_matching_drift_models(self) -> None:
        sampling_rate_hz = 50.0
        repeated_count = 10
        active_time = np.arange(1, 1501, dtype=float) / sampling_rate_hz
        time = np.concatenate((np.zeros(repeated_count), active_time))
        force = np.concatenate(
            (np.full(repeated_count, REST_FORCE_N), REST_FORCE_N + jet_cycle_force(active_time, 1.0))
        )

        result = self.analyze(time, force)

        self.assertEqual(result.drift_model, "cycle template")
        np.testing.assert_allclose(
            result.tare_only_cycle_mean_forces, result.cycle_mean_forces, atol=0.003
        )

    def test_explicit_rest_durations_need_five_samples(self) -> None:
        sampling_rate_hz = 50.0
        time = np.arange(0.0, 33.0, 1.0 / sampling_rate_hz)
        waveform = np.where((time >= 3.0) & (time < 30.0), jet_cycle_force(time - 3.0, 1.0), 0.0)
        force = REST_FORCE_N + waveform

        with self.assertRaisesRegex(ValueError, "tare"):
            self.analyze(time, force, tare_duration_s=0.05)
        with self.assertRaisesRegex(ValueError, "post-run rest"):
            self.analyze(time, force, tare_duration_s=3.0, post_rest_duration_s=0.05)

    def test_rest_segments_must_be_at_rest(self) -> None:
        sampling_rate_hz = 50.0
        time = np.arange(0.0, 33.0, 1.0 / sampling_rate_hz)
        waveform = np.where((time >= 3.0) & (time < 30.0), jet_cycle_force(time - 3.0, 1.0), 0.0)
        force = REST_FORCE_N + waveform

        with self.assertRaisesRegex(ValueError, "tare is not at rest"):
            self.analyze(time, force, tare_duration_s=6.0)
        with self.assertRaisesRegex(ValueError, "post-run rest is not at rest"):
            self.analyze(time, force, tare_duration_s=3.0, post_rest_duration_s=10.0)

    def test_noise_without_repeatable_cycles_is_rejected(self) -> None:
        rng = np.random.default_rng(3)
        time, _ = sparse_logger_recording()
        force = REST_FORCE_N + 0.004 * rng.standard_normal(time.size)

        with self.assertRaisesRegex(ValueError, "cycle"):
            self.analyze(time, force)


class SparseSamplingCycleTests(unittest.TestCase):
    """The robot logger records about eleven samples per actuation cycle."""

    @classmethod
    def setUpClass(cls) -> None:
        analyze = importlib.import_module("dynamic_baseline").analyze_dynamic_baseline
        cls.time, cls.force = sparse_logger_recording()
        cls.result = analyze(cls.time, cls.force)

    def test_each_cycle_window_holds_its_stroke_peak_inside(self) -> None:
        boundaries = np.asarray(self.result.cycle_boundaries, dtype=int)
        peaks = np.asarray(self.result.peaks, dtype=int)
        samples_per_cycle = SPARSE_SAMPLING_RATE_HZ / SPARSE_FREQUENCY_HZ

        self.assertGreaterEqual(peaks.size, 25)
        self.assertTrue(np.all(boundaries[:-1] < peaks))
        self.assertTrue(np.all(peaks < boundaries[1:]))
        self.assertTrue(np.all(np.diff(peaks) > 0.5 * samples_per_cycle))

    def test_recovery_valley_follows_its_own_stroke(self) -> None:
        peaks = np.asarray(self.result.peaks, dtype=int)
        valleys = np.asarray(self.result.recovery_valleys, dtype=int)

        self.assertTrue(np.all(valleys > peaks))
        self.assertTrue(np.all(valleys[:-1] < peaks[1:]))
        self.assertEqual(np.unique(valleys).size, valleys.size)

    def test_mean_force_matches_the_waveform_mean(self) -> None:
        self.assertAlmostEqual(
            float(np.mean(self.result.cycle_mean_forces)),
            jet_cycle_mean_force(SPARSE_FREQUENCY_HZ),
            delta=0.004,
        )

    def test_noise_free_cycles_at_logger_rate_show_no_fake_drift(self) -> None:
        # About 10.7 samples per cycle: cycle edges must not be snapped to samples,
        # otherwise windows of 10 or 11 samples fake cycle-to-cycle force changes.
        analyze = importlib.import_module("dynamic_baseline").analyze_dynamic_baseline
        true_mean = jet_cycle_mean_force(SPARSE_FREQUENCY_HZ)
        time, _ = sparse_logger_recording(noise_n=0.0)
        actuated = time > 0.0
        for phase_s in np.linspace(0.0, 1.8, 7):
            with self.subTest(phase_s=float(phase_s)):
                force = np.full(time.size, REST_FORCE_N)
                force[actuated] += jet_cycle_force(time[actuated] + phase_s, SPARSE_FREQUENCY_HZ)

                result = analyze(time, force)

                cycle_forces = np.asarray(result.cycle_mean_forces)
                self.assertLess(float(np.std(cycle_forces, ddof=1)) / true_mean, 0.025)
                self.assertLess(float(np.ptp(result.baseline)), 0.005)
                self.assertAlmostEqual(float(np.mean(cycle_forces)), true_mean, delta=0.03 * true_mean)


class DynamicBaselineHeadlessFlowTests(unittest.TestCase):
    def test_analysis_result_can_be_rendered_to_headless_svg(self) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt

        analyze = importlib.import_module("dynamic_baseline").analyze_dynamic_baseline
        time = np.arange(0.0, 8.0, 1.0 / 160.0)
        force = 0.16 + 0.02 * np.cos(2.0 * np.pi * 0.08 * time)
        force += 0.58 * _positive_pulse(time, 2.0)

        result = analyze(time, force, tare_duration_s=0.0)
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

"""Integration coverage for the spreadsheet-to-SVG analysis flow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests.synthetic_signals import SPARSE_SAMPLING_RATE_HZ, sparse_logger_recording
from thrust_analysis import (
    ProfessionalThrustAnalyzer,
    build_summary_dataframe,
    parse_optional_seconds,
)


def _quiet_analyzer(sampling_rate_hz: float, **options: object) -> ProfessionalThrustAnalyzer:
    return ProfessionalThrustAnalyzer(sampling_rate_hz, log_func=lambda _message: None, **options)


def _bipolar_recording(start_time_s: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """120 Hz recording with a repeated-timestamp rest block and slow drift."""

    sampling_rate_hz = 120.0
    repeated_count = 10
    active_time = np.arange(1, 1201, dtype=float) / sampling_rate_hz
    phase_signal = np.sin(2.0 * np.pi * 1.5 * active_time)
    waveform = 0.50 * np.maximum(phase_signal, 0.0) ** 2
    waveform -= 0.12 * np.maximum(-phase_signal, 0.0) ** 2
    baseline = 0.24 + 0.018 * active_time / active_time[-1]
    time = start_time_s + np.concatenate((np.zeros(repeated_count), active_time))
    force = np.concatenate((np.full(repeated_count, 0.24), baseline + waveform))
    return time, force


class AnalyzerIntegrationTests(unittest.TestCase):
    def test_processed_mode_extrema_are_taken_from_filtered_signal(self) -> None:
        sampling_rate_hz = 160.0
        time = np.arange(0.0, 10.0, 1.0 / sampling_rate_hz)
        phase_signal = np.sin(2.0 * np.pi * 2.0 * time)
        physical = 0.45 * np.maximum(phase_signal, 0.0) ** 2
        physical -= 0.10 * np.maximum(-phase_signal, 0.0) ** 2
        noisy_force = 0.20 + physical + 0.035 * np.sin(2.0 * np.pi * 35.0 * time)
        # The recording starts at zero force, so the first sample is the tare.
        analyzer = _quiet_analyzer(
            sampling_rate_hz, use_processed_signal=True, tare_duration_s=0.0
        )

        corrected, filtered, _baseline = analyzer.process_signal(
            noisy_force, cutoff_freq=6.0, t=time
        )
        cycles, _period = analyzer.measure_cycles(time, corrected, filtered)

        self.assertGreaterEqual(len(cycles), 15)
        for cycle in cycles:
            start = cycle["cycle_start_idx"]
            stop = cycle["cycle_end_idx"]
            self.assertAlmostEqual(cycle["peak_value"], float(filtered[cycle["peak_idx"]]))
            self.assertAlmostEqual(cycle["peak_value"], float(np.max(filtered[start : stop + 1])))
            self.assertAlmostEqual(
                cycle["recovery_value"], float(filtered[cycle["valley_idx"]])
            )
            self.assertGreater(cycle["valley_idx"], cycle["peak_idx"])
            self.assertLess(cycle["recovery_value"], -0.05)

    def test_excel_analysis_preserves_recovery_drag_and_writes_svg(self) -> None:
        time, force = _bipolar_recording()

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "synthetic.xlsx"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(
                input_path, index=False
            )
            analyzer = _quiet_analyzer(120.0, use_processed_signal=True)

            result = analyzer.analyze_file(input_path, cutoff_freq=5.0)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertAlmostEqual(result["detected_frequency"], 1.5, delta=0.03)
            self.assertLess(result["mean_recovery_force"], -0.06)
            self.assertGreater(result["mean_cycle_force"], 0.05)
            self.assertGreater(result["baseline_drift"], 0.012)
            svg_path = (
                input_path.parent
                / "thrust_analysis_results"
                / "synthetic_analysis.svg"
            )
            self.assertTrue(svg_path.is_file())
            self.assertIn("<svg", svg_path.read_text(encoding="utf-8"))

            table = build_summary_dataframe([result])

        columns = list(table.columns)
        for expected in (
            "Mean_Cycle_Force(N)",
            "Mean_Cycle_Force_TareOnly(N)",
            "Mean_Peak_Thrust(N)",
            "Mean_Peak_to_Valley(N)",
            "Mean_Impulse(N\u00b7s)",
            "Total_Impulse(N\u00b7s)",
            "Drift_Model",
            "Signal",
        ):
            self.assertIn(expected, columns)
        self.assertNotIn("Mean_Net(N)", columns)
        self.assertFalse(any("\u8def" in column for column in columns))
        self.assertTrue(pd.api.types.is_float_dtype(table["Mean_Cycle_Force(N)"]))

    def test_duration_is_elapsed_time_and_csv_is_read_directly(self) -> None:
        time, force = _bipolar_recording(start_time_s=5.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "synthetic.csv"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_csv(
                input_path, index=False
            )
            not_excel = AssertionError("a .csv file must not be read as Excel")
            with mock.patch("thrust_analysis.pd.read_excel", side_effect=not_excel):
                result = _quiet_analyzer(120.0).analyze_file(input_path, cutoff_freq=5.0)

        assert result is not None
        self.assertAlmostEqual(result["data_duration"], float(time[-1] - time[0]))

    def test_low_pass_switches_off_exactly_at_nyquist(self) -> None:
        time, force = sparse_logger_recording()
        analyzer = _quiet_analyzer(SPARSE_SAMPLING_RATE_HZ, use_processed_signal=True)
        analyzer.process_signal(force, cutoff_freq=100.0, t=time)
        nyquist_hz = 0.5 * analyzer.fs

        corrected, filtered, _baseline = analyzer.process_signal(force, cutoff_freq=nyquist_hz, t=time)
        np.testing.assert_array_equal(filtered, corrected)
        self.assertIn("off", analyzer.lowpass_description)

        just_below = float(np.nextafter(nyquist_hz, 0.0))
        corrected, filtered, _baseline = analyzer.process_signal(force, cutoff_freq=just_below, t=time)
        self.assertGreater(float(np.max(np.abs(filtered - corrected))), 1e-6)
        self.assertNotIn("off", analyzer.lowpass_description)

    def test_cutoff_below_nyquist_filters_and_warns_when_it_trims_harmonics(self) -> None:
        time, force = sparse_logger_recording()
        messages: list[str] = []
        analyzer = ProfessionalThrustAnalyzer(
            SPARSE_SAMPLING_RATE_HZ, log_func=messages.append, use_processed_signal=True
        )

        corrected, filtered, _baseline = analyzer.process_signal(force, cutoff_freq=1.5, t=time)
        self.assertGreater(float(np.max(np.abs(filtered - corrected))), 1e-3)
        self.assertIn("1.50 Hz", analyzer.lowpass_description)
        self.assertTrue(any("Warning" in message for message in messages))

        messages.clear()
        # 2.6 Hz is above five times the 0.4992 Hz cycle frequency but below Nyquist.
        analyzer.process_signal(force, cutoff_freq=2.6, t=time)
        self.assertIn("2.60 Hz", analyzer.lowpass_description)
        self.assertFalse(any("Warning" in message for message in messages))

    def test_processed_signal_is_the_default_measurement(self) -> None:
        self.assertTrue(_quiet_analyzer(100.0).use_processed_signal)


class OptionParsingTests(unittest.TestCase):
    def test_parse_optional_seconds(self) -> None:
        self.assertIsNone(parse_optional_seconds("auto", allow_auto=True))
        self.assertIsNone(parse_optional_seconds("  ", allow_auto=True))
        self.assertEqual(parse_optional_seconds("2.5", allow_auto=True), 2.5)
        self.assertEqual(parse_optional_seconds("0", allow_auto=False), 0.0)
        for bad_text, allow_auto in (("-1", True), ("abc", True), ("nan", True), ("auto", False)):
            with self.subTest(text=bad_text):
                with self.assertRaises(ValueError):
                    parse_optional_seconds(bad_text, allow_auto=allow_auto)


if __name__ == "__main__":
    unittest.main()

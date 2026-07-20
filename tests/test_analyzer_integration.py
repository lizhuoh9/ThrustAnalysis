"""Integration coverage for the spreadsheet-to-SVG analysis flow."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from thrust_analysis import ProfessionalThrustAnalyzer


class AnalyzerIntegrationTests(unittest.TestCase):
    def test_processed_mode_extrema_are_taken_from_filtered_signal(self) -> None:
        sampling_rate_hz = 160.0
        time = np.arange(0.0, 10.0, 1.0 / sampling_rate_hz)
        phase_signal = np.sin(2.0 * np.pi * 2.0 * time)
        physical = 0.45 * np.maximum(phase_signal, 0.0) ** 2
        physical -= 0.10 * np.maximum(-phase_signal, 0.0) ** 2
        noisy_force = 0.20 + physical + 0.035 * np.sin(2.0 * np.pi * 35.0 * time)
        analyzer = ProfessionalThrustAnalyzer(
            sampling_rate_hz,
            log_func=lambda _message: None,
            use_processed_signal=True,
        )

        corrected, filtered, _baseline = analyzer.process_signal(
            noisy_force, cutoff_freq=6.0, t=time
        )
        cycles, _period = analyzer.detect_cycles_with_fallback(
            time, corrected, filtered
        )

        for cycle in cycles:
            start = cycle["cycle_start_idx"]
            stop = cycle["cycle_end_idx"]
            self.assertAlmostEqual(
                cycle["peak_value"], float(np.max(filtered[start:stop]))
            )
            self.assertAlmostEqual(
                cycle["recovery_value"], float(np.min(filtered[start:stop]))
            )

    def test_excel_analysis_preserves_recovery_drag_and_writes_svg(self) -> None:
        sampling_rate_hz = 120.0
        frequency_hz = 1.5
        repeated_count = 10
        active_time = np.arange(1, 1201, dtype=float) / sampling_rate_hz
        phase_signal = np.sin(2.0 * np.pi * frequency_hz * active_time)
        waveform = 0.50 * np.maximum(phase_signal, 0.0) ** 2
        waveform -= 0.12 * np.maximum(-phase_signal, 0.0) ** 2
        baseline = 0.24 + 0.018 * active_time / active_time[-1]
        time = np.concatenate((np.zeros(repeated_count), active_time))
        force = np.concatenate((np.full(repeated_count, 0.24), baseline + waveform))

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "synthetic.xlsx"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(
                input_path, index=False
            )
            analyzer = ProfessionalThrustAnalyzer(
                sampling_rate_hz,
                log_func=lambda _message: None,
                use_processed_signal=True,
            )

            result = analyzer.analyze_file(input_path, cutoff_freq=5.0)

            self.assertIsNotNone(result)
            assert result is not None
            self.assertAlmostEqual(result["detected_frequency"], frequency_hz, delta=0.03)
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


if __name__ == "__main__":
    unittest.main()

"""GUI wiring contracts; skipped when no Tk display is available."""

from __future__ import annotations

import tempfile
import threading
import time as wall_clock
import tkinter as tk
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import pandas as pd

from tests.synthetic_signals import sparse_logger_recording
from thrust_analysis import ProfessionalThrustAnalyzer, ThrustAnalyzerGUI

UI_TIMEOUT_S = 60.0


class GuiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}") from error
        cls.root.withdraw()
        cls.gui = ThrustAnalyzerGUI(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def _pump_until(self, condition: Callable[[], bool], timeout_s: float = UI_TIMEOUT_S) -> bool:
        """Process Tk events (as mainloop would) until ``condition`` holds."""

        deadline = wall_clock.monotonic() + timeout_s
        while wall_clock.monotonic() < deadline:
            self.root.update()
            if condition():
                return True
            wall_clock.sleep(0.01)
        return False

    def test_cutoff_entry_is_editable(self) -> None:
        self.assertEqual(str(self.gui.cutoff_entry.cget("state")), "normal")

    def test_log_from_worker_thread_reaches_the_log_panel(self) -> None:
        message = "message-from-worker-thread"
        worker = threading.Thread(target=self.gui.log, args=(message,))
        worker.start()
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertTrue(
            self._pump_until(lambda: message in self.gui.log_text.get("1.0", tk.END), 5.0)
        )

    def test_batch_run_writes_summary_through_the_worker_thread(self) -> None:
        time, force = sparse_logger_recording()
        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "logger.xlsx"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(input_path, index=False)
            self.gui.file_paths = [str(input_path)]
            self.gui.detect_sampling_rate(str(input_path))

            with mock.patch("thrust_analysis.messagebox") as dialogs:
                self.gui.start_analysis()
                finished = self._pump_until(lambda: not self.gui.is_processing)

            self.assertTrue(finished)
            dialogs.showerror.assert_not_called()
            dialogs.showinfo.assert_called_once()
            summary_path = input_path.parent / "thrust_analysis_results" / "batch_summary.xlsx"
            self.assertTrue(summary_path.is_file())
            table = pd.read_excel(summary_path)

        self.assertEqual(len(table), 1)
        self.assertGreaterEqual(int(table.loc[0, "Cycles"]), 25)

    def test_invalid_cutoff_is_rejected_before_starting(self) -> None:
        self.gui.file_paths = ["unused.xlsx"]
        self.gui.detected_sampling_rate = 5.333
        self.gui.cutoff_freq_var.set("not-a-number")
        try:
            with mock.patch("thrust_analysis.messagebox") as dialogs:
                self.gui.start_analysis()
        finally:
            self.gui.cutoff_freq_var.set("5.0")

        dialogs.showerror.assert_called_once()
        self.assertFalse(self.gui.is_processing)

    def test_stop_keeps_the_summary_of_finished_files(self) -> None:
        time, force = sparse_logger_recording()
        original_analyze_file = ProfessionalThrustAnalyzer.analyze_file
        stop_applied = threading.Event()

        def request_stop() -> None:
            self.gui.stop_analysis()
            stop_applied.set()

        def analyze_then_stop(analyzer, file_path, cutoff_freq):
            result = original_analyze_file(analyzer, file_path, cutoff_freq=cutoff_freq)
            self.gui._call_in_ui(request_stop)
            stop_applied.wait(timeout=10.0)
            return result

        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = []
            for name in ("first.xlsx", "second.xlsx"):
                path = Path(temporary_directory) / name
                pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(path, index=False)
                paths.append(str(path))
            self.gui.file_paths = paths
            self.gui.detect_sampling_rate(paths[0])

            with mock.patch("thrust_analysis.messagebox") as dialogs, mock.patch.object(
                ProfessionalThrustAnalyzer, "analyze_file", analyze_then_stop
            ):
                self.gui.start_analysis()
                finished = self._pump_until(lambda: not self.gui.is_processing)

            self.assertTrue(finished)
            dialogs.showinfo.assert_not_called()
            self.assertEqual(self.gui.status_label.cget("text"), "Stopped")
            summary_path = Path(temporary_directory) / "thrust_analysis_results" / "batch_summary.xlsx"
            self.assertEqual(len(pd.read_excel(summary_path)), 1)

    def test_partial_failure_is_reported_with_the_failed_file(self) -> None:
        time, force = sparse_logger_recording()
        with tempfile.TemporaryDirectory() as temporary_directory:
            good_path = Path(temporary_directory) / "good.xlsx"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(good_path, index=False)
            broken_path = Path(temporary_directory) / "broken.csv"
            pd.DataFrame({"Time (s)": [0.0, 0.2, 0.4], "Thrust (N)": [0.8, 0.9, 0.8]}).to_csv(
                broken_path, index=False
            )
            self.gui.file_paths = [str(good_path), str(broken_path)]
            self.gui.detect_sampling_rate(str(good_path))

            with mock.patch("thrust_analysis.messagebox") as dialogs:
                self.gui.start_analysis()
                finished = self._pump_until(lambda: not self.gui.is_processing)

        self.assertTrue(finished)
        self.assertEqual(self.gui.status_label.cget("text"), "Complete with errors")
        dialogs.showinfo.assert_not_called()
        dialogs.showwarning.assert_called_once()
        self.assertIn("broken.csv", dialogs.showwarning.call_args.args[1])

    def test_stop_during_the_last_file_still_completes(self) -> None:
        time, force = sparse_logger_recording()
        original_analyze_file = ProfessionalThrustAnalyzer.analyze_file
        stop_applied = threading.Event()

        def request_stop() -> None:
            self.gui.stop_analysis()
            stop_applied.set()

        def stop_while_analyzing(analyzer, file_path, cutoff_freq):
            self.gui._call_in_ui(request_stop)
            stop_applied.wait(timeout=10.0)
            return original_analyze_file(analyzer, file_path, cutoff_freq=cutoff_freq)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "only.xlsx"
            pd.DataFrame({"Time (s)": time, "Thrust (N)": force}).to_excel(path, index=False)
            self.gui.file_paths = [str(path)]
            self.gui.detect_sampling_rate(str(path))

            with mock.patch("thrust_analysis.messagebox") as dialogs, mock.patch.object(
                ProfessionalThrustAnalyzer, "analyze_file", stop_while_analyzing
            ):
                self.gui.start_analysis()
                finished = self._pump_until(lambda: not self.gui.is_processing)

        self.assertTrue(finished)
        self.assertEqual(self.gui.status_label.cget("text"), "Complete")
        dialogs.showinfo.assert_called_once()

    def test_run_where_every_file_fails_reports_no_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            broken_path = Path(temporary_directory) / "too_short.csv"
            pd.DataFrame({"Time (s)": [0.0, 0.2, 0.4], "Thrust (N)": [0.8, 0.9, 0.8]}).to_csv(
                broken_path, index=False
            )
            self.gui.file_paths = [str(broken_path)]
            self.gui.detected_sampling_rate = 5.0

            with mock.patch("thrust_analysis.messagebox") as dialogs:
                self.gui.start_analysis()
                finished = self._pump_until(lambda: not self.gui.is_processing)

            self.assertTrue(finished)
            dialogs.showinfo.assert_not_called()
            self.assertEqual(self.gui.status_label.cget("text"), "No results")
            self.assertFalse((Path(temporary_directory) / "thrust_analysis_results").exists())


if __name__ == "__main__":
    unittest.main()

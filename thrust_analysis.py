import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues with Tkinter
import matplotlib.pyplot as plt
from pathlib import Path
import math
import os
import queue
import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading

from dynamic_baseline import analyze_dynamic_baseline, cycle_extrema, lowpass_irregular_signal

# Set up matplotlib for English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR_NAME = "thrust_analysis_results"
SUMMARY_FILE_NAME = "batch_summary.xlsx"
UI_POLL_INTERVAL_MS = 50
# A low-pass cutoff below this multiple of the cycle frequency trims stroke peaks.
HARMONIC_SAFE_CUTOFF_MULTIPLE = 5.0


def read_recording_table(file_path):
    """Read a .csv file as CSV and any other file as an Excel workbook."""
    if Path(file_path).suffix.lower() == '.csv':
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def parse_optional_seconds(text, *, allow_auto):
    """Parse a duration field: blank or 'auto' gives None when allowed, else seconds >= 0."""
    cleaned = str(text).strip().lower()
    if cleaned in ('', 'auto'):
        if allow_auto:
            return None
        raise ValueError("enter a number of seconds (0 = none)")
    try:
        value = float(cleaned)
    except ValueError:
        raise ValueError(f"'{text}' is not a number of seconds") from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("durations must be finite and non-negative")
    return value


def build_summary_dataframe(results):
    """Create a numeric summary table with one row per analyzed file."""
    rows = []
    for r in results:
        rows.append({
            'Folder': r['folder_name'],
            'Folder_Path': r['source_folder'],
            'File': r['file_name'],
            'Duration(s)': round(r['data_duration'], 2),
            'Cycles': r['num_cycles'],
            'Forward': r['forward_cycles'],
            'Backward': r['backward_cycles'],
            'Mean_Cycle_Force(N)': round(r['mean_cycle_force'], 4),
            'Std_Cycle_Force(N)': round(r['std_cycle_force'], 4),
            'Mean_Cycle_Force_TareOnly(N)': round(r['mean_cycle_force_tare_only'], 4),
            'Mean_Peak_Thrust(N)': round(r['mean_peak_thrust'], 4),
            'Mean_Recovery_Force(N)': round(r['mean_recovery_force'], 4),
            'Mean_Peak_to_Valley(N)': round(r['mean_peak_to_valley'], 4),
            'Std_Peak_to_Valley(N)': round(r['std_peak_to_valley'], 4),
            'Mean_Impulse(N\u00b7s)': round(r['mean_impulse'], 6),
            'Total_Impulse(N\u00b7s)': round(r['total_impulse'], 6),
            'Detected_Period(s)': round(r['detected_period'], 4),
            'Detected_Frequency(Hz)': round(r['detected_frequency'], 4),
            'Drift_Model': r['drift_model'],
            'Tare(N)': round(r['tare_force'], 4),
            'Tare_Samples': r['tare_samples'],
            'Baseline_Start(N)': round(r['baseline_start'], 4),
            'Baseline_End(N)': round(r['baseline_end'], 4),
            'Baseline_Drift(N)': round(r['baseline_drift'], 4),
            'Signal': r['signal_description']
        })
    return pd.DataFrame(rows)


def save_summary_reports(results):
    """Save one batch summary workbook per source folder and return their paths."""
    grouped_results = {}
    for result in results:
        grouped_results.setdefault(result['source_folder'], []).append(result)

    summary_paths = []
    for source_folder in sorted(grouped_results):
        output_dir = Path(source_folder) / RESULTS_DIR_NAME
        output_dir.mkdir(exist_ok=True)
        summary_path = output_dir / SUMMARY_FILE_NAME
        build_summary_dataframe(grouped_results[source_folder]).to_excel(
            summary_path, index=False, engine='openpyxl'
        )
        summary_paths.append(summary_path)
    return summary_paths


class ThrustAnalyzerGUI:
    """Unified Thrust Analysis GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Reading Analysis")
        self.root.geometry("1240x820")
        self.root.minsize(1120, 760)
        self.colors = {
            'bg': '#eef3f8',
            'surface': '#ffffff',
            'surface_alt': '#f7fbff',
            'border': '#d7e1ea',
            'text': '#17324d',
            'muted': '#60758a',
            'accent': '#1d6fa5',
            'accent_dark': '#15557d',
            'accent_soft': '#dcecf8',
            'success': '#1f9d68',
            'warning': '#c57a16',
            'danger': '#c54f49',
            'console_bg': '#10212f',
            'console_fg': '#e8f0f7'
        }
        
        self.file_paths = []
        self.last_output_dirs = []
        self.is_processing = False
        self.stop_requested = False
        self.detected_sampling_rate = None
        # Tk is single-threaded: the analysis worker queues UI updates and the
        # Tk thread applies them in _drain_ui_queue.
        self._ui_queue = queue.Queue()
        self._ui_thread = threading.current_thread()
        
        self.use_processed_signal = tk.BooleanVar(value=True)
        self.tare_duration_var = tk.StringVar(value="auto")
        self.post_rest_var = tk.StringVar(value="0")
        self.file_summary_var = tk.StringVar(value="No files selected yet.")
        self.signal_mode_hint_var = tk.StringVar()
        
        self.configure_styles()
        self.create_widgets()
        self.root.after(UI_POLL_INTERVAL_MS, self._drain_ui_queue)

    def configure_styles(self):
        """Configure ttk styles for a cleaner interface."""
        self.root.configure(bg=self.colors['bg'])
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')

        style.configure('.', font=('Segoe UI', 10))
        style.configure('App.TFrame', background=self.colors['bg'])
        style.configure('Panel.TFrame', background=self.colors['surface'])
        style.configure(
            'Section.TLabelframe',
            background=self.colors['surface'],
            bordercolor=self.colors['border'],
            borderwidth=1,
            relief='solid'
        )
        style.configure(
            'Section.TLabelframe.Label',
            background=self.colors['surface'],
            foreground=self.colors['text'],
            font=('Segoe UI Semibold', 11)
        )
        style.configure('Panel.TLabel', background=self.colors['surface'], foreground=self.colors['text'])
        style.configure('Muted.TLabel', background=self.colors['surface'], foreground=self.colors['muted'])
        style.configure(
            'Primary.TButton',
            background=self.colors['accent'],
            foreground='white',
            borderwidth=0,
            padding=(14, 9),
            font=('Segoe UI Semibold', 10)
        )
        style.map(
            'Primary.TButton',
            background=[('active', self.colors['accent_dark']), ('disabled', '#aac0d2')],
            foreground=[('disabled', '#edf5fb')]
        )
        style.configure(
            'Secondary.TButton',
            background=self.colors['surface_alt'],
            foreground=self.colors['text'],
            borderwidth=1,
            relief='solid',
            padding=(14, 9)
        )
        style.map('Secondary.TButton', background=[('active', self.colors['accent_soft'])])
        style.configure(
            'Danger.TButton',
            background='#fbe9e8',
            foreground=self.colors['danger'],
            borderwidth=1,
            relief='solid',
            padding=(14, 9),
            font=('Segoe UI Semibold', 10)
        )
        style.map(
            'Danger.TButton',
            background=[('active', '#f6d5d3'), ('disabled', '#f5eded')],
            foreground=[('disabled', '#c7a3a0')]
        )
        style.configure(
            'Accent.Horizontal.TProgressbar',
            troughcolor='#dce6ef',
            background=self.colors['accent'],
            bordercolor='#dce6ef',
            lightcolor=self.colors['accent'],
            darkcolor=self.colors['accent']
        )
        style.configure('Custom.TNotebook', background=self.colors['surface'], borderwidth=0)
        style.configure(
            'Custom.TNotebook.Tab',
            background='#e5edf5',
            foreground=self.colors['muted'],
            padding=(16, 8),
            font=('Segoe UI Semibold', 10)
        )
        style.map(
            'Custom.TNotebook.Tab',
            background=[('selected', self.colors['surface']), ('active', self.colors['accent_soft'])],
            foreground=[('selected', self.colors['text']), ('active', self.colors['text'])]
        )
        
    def create_widgets(self):
        """Create GUI components"""
        
        self.root.columnconfigure(0, weight=0, minsize=410)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left Control Panel (scrollable: it is taller than most screens)
        left_frame = self.create_scrollable_panel()
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        
        hero_frame = tk.Frame(left_frame, bg=self.colors['accent'], padx=18, pady=14)
        hero_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 12))
        tk.Label(
            hero_frame,
            text="Sensor Reading Analysis",
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI Semibold', 16)
        ).grid(row=0, column=0, sticky=tk.W)
        tk.Label(
            hero_frame,
            text="Remove drift, preserve signed force, and measure repeatable propulsion cycles.",
            bg=self.colors['accent'],
            fg='#d8ecfb',
            font=('Segoe UI', 10)
        ).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        
        # ========== Section 1: File Selection ==========
        file_frame = ttk.LabelFrame(left_frame, text="1. Input Files", padding="12", style='Section.TLabelframe')
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=6)
        file_frame.columnconfigure(0, weight=1)
        file_frame.columnconfigure(1, weight=1)
        
        ttk.Button(
            file_frame, text="Add Single File", style='Secondary.TButton',
            command=self.select_single_file
        ).grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=(0, 8))
        ttk.Button(
            file_frame, text="Add Multiple Files", style='Secondary.TButton',
            command=self.select_multiple_files
        ).grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=(0, 8))
        ttk.Button(
            file_frame, text="Add Folders", style='Secondary.TButton',
            command=self.select_folders
        ).grid(row=1, column=0, sticky='ew', padx=(0, 6), pady=(0, 10))
        ttk.Button(
            file_frame, text="Clear List", style='Secondary.TButton',
            command=self.clear_files
        ).grid(row=1, column=1, sticky='ew', padx=(6, 0), pady=(0, 10))
        
        # File List
        self.file_listbox = tk.Listbox(
            file_frame, height=7, width=45, font=("Segoe UI", 10),
            bg=self.colors['surface_alt'], fg=self.colors['text'],
            selectbackground=self.colors['accent'], selectforeground='white',
            relief='flat', highlightthickness=1, highlightbackground=self.colors['border'],
            activestyle='none'
        )
        self.file_listbox.grid(row=2, column=0, columnspan=2, sticky='ew', pady=2, ipady=4)
        
        scrollbar = ttk.Scrollbar(file_frame, orient="vertical", command=self.file_listbox.yview)
        scrollbar.grid(row=2, column=2, sticky=(tk.N, tk.S), padx=(8, 0))
        self.file_listbox.config(yscrollcommand=scrollbar.set)
        ttk.Label(file_frame, textvariable=self.file_summary_var, style='Muted.TLabel').grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(8, 0)
        )
        
        # ========== Section 2: Sampling Rate ==========
        sr_frame = ttk.LabelFrame(left_frame, text="2. Sampling & Tare", padding="12", style='Section.TLabelframe')
        sr_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=6)
        sr_frame.columnconfigure(1, weight=1)
        
        ttk.Label(sr_frame, text="Detected Rate", style='Muted.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.sampling_rate_var = tk.StringVar(value="Auto-detect")
        self.sampling_rate_label = ttk.Label(sr_frame, textvariable=self.sampling_rate_var,
                                            font=("Segoe UI Semibold", 11), foreground=self.colors['accent'])
        self.sampling_rate_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(
            sr_frame,
            text="The first valid file is used to estimate the sampling interval automatically.",
            style='Muted.TLabel'
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        tare_frame = ttk.Frame(sr_frame, style='Panel.TFrame')
        tare_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        ttk.Label(tare_frame, text="Rest before (s)", style='Muted.TLabel').grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(tare_frame, textvariable=self.tare_duration_var, width=7).grid(
            row=0, column=1, sticky=tk.W, padx=(6, 16)
        )
        ttk.Label(tare_frame, text="Rest after (s)", style='Muted.TLabel').grid(row=0, column=2, sticky=tk.W)
        ttk.Entry(tare_frame, textvariable=self.post_rest_var, width=7).grid(
            row=0, column=3, sticky=tk.W, padx=(6, 0)
        )
        ttk.Label(
            sr_frame,
            text="'auto' tares on the rest samples logged at the first timestamp. "
                 "A rest after actuation enables two-point drift correction.",
            style='Muted.TLabel',
            wraplength=360
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        
        # ========== Section 3: Cycle Detection (Auto) ==========
        param_frame = ttk.LabelFrame(left_frame, text="3. Cycle Detection", padding="12", style='Section.TLabelframe')
        param_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=6)
        param_frame.columnconfigure(1, weight=1)
        
        ttk.Label(
            param_frame,
            text="FFT-based period detection is enabled automatically.",
            style='Muted.TLabel'
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        ttk.Label(param_frame, text="Detected Period", style='Muted.TLabel').grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        self.detected_period_var = tk.StringVar(value="Run analysis to detect")
        ttk.Label(param_frame, textvariable=self.detected_period_var,
                 font=("Segoe UI Semibold", 11), foreground=self.colors['accent']).grid(
                     row=1, column=1, sticky=tk.W, padx=5, pady=(8, 0)
                 )
        
        # ========== Section 4: Advanced Options ==========
        adv_frame = ttk.LabelFrame(left_frame, text="4. Analysis Options", padding="12", style='Section.TLabelframe')
        adv_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=6)
        adv_frame.columnconfigure(0, weight=1)
        adv_frame.columnconfigure(1, weight=1)
        
        ttk.Label(
            adv_frame,
            text="FFT cycle-template detection: enabled",
            style='Muted.TLabel'
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Signal choice
        ttk.Label(adv_frame, text="Measurement Signal", style='Muted.TLabel').grid(
            row=1, column=0, columnspan=2, sticky=tk.W, pady=(12, 4)
        )
        ttk.Radiobutton(adv_frame, text="Processed Signal (Recommended)", 
                       variable=self.use_processed_signal, value=True).grid(row=2, column=0, columnspan=2, sticky=tk.W)
        ttk.Radiobutton(adv_frame, text="Original Signal", 
                       variable=self.use_processed_signal, value=False).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ttk.Label(adv_frame, textvariable=self.signal_mode_hint_var, style='Muted.TLabel',
                  wraplength=440).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0)
        )
        
        # Low-pass cutoff
        ttk.Label(adv_frame, text="Low-pass Cutoff (Hz)", style='Muted.TLabel').grid(
            row=5, column=0, sticky=tk.W, pady=(12, 4)
        )
        self.cutoff_freq_var = tk.StringVar(value="5.0")
        self.cutoff_entry = ttk.Entry(adv_frame, textvariable=self.cutoff_freq_var, width=10)
        self.cutoff_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=(12, 4))
        self.cutoff_hint_label = ttk.Label(
            adv_frame,
            text="A cutoff at or above the Nyquist frequency switches the filter off.",
            style='Muted.TLabel'
        )
        self.cutoff_hint_label.grid(row=6, column=0, columnspan=2, sticky=tk.W)
        
        # ========== Section 5: Actions ==========
        action_frame = ttk.LabelFrame(left_frame, text="5. Actions", padding="12", style='Section.TLabelframe')
        action_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=6)
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        
        self.start_button = ttk.Button(action_frame, text="Start Analysis", 
                                       command=self.start_analysis, style='Primary.TButton')
        self.start_button.grid(row=0, column=0, sticky='ew', padx=(0, 6), pady=(0, 8))
        
        self.stop_button = ttk.Button(action_frame, text="Stop", 
                                      command=self.stop_analysis, style='Danger.TButton', state="disabled")
        self.stop_button.grid(row=0, column=1, sticky='ew', padx=(6, 0), pady=(0, 8))
        
        ttk.Button(action_frame, text="Quick Diagnostic", style='Secondary.TButton',
                  command=self.quick_diagnostic).grid(row=1, column=0, sticky='ew', padx=(0, 6))
        self.open_results_button = ttk.Button(
            action_frame, text="Open Results Folder", style='Secondary.TButton',
            command=self.open_results_folder, state='disabled'
        )
        self.open_results_button.grid(row=1, column=1, sticky='ew', padx=(6, 0))

        self.clear_log_button = ttk.Button(
            action_frame, text="Clear Log", style='Secondary.TButton',
            command=self.clear_log
        )
        self.clear_log_button.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(8, 0))
        
        # Progress Bar
        self.progress = ttk.Progressbar(left_frame, mode='indeterminate', style='Accent.Horizontal.TProgressbar')
        self.progress.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(8, 10))
        
        # Status
        status_frame = tk.Frame(
            left_frame,
            bg=self.colors['surface_alt'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            padx=12,
            pady=10
        )
        status_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 4))
        tk.Label(
            status_frame, text="STATUS", bg=self.colors['surface_alt'],
            fg=self.colors['muted'], font=('Segoe UI Semibold', 9)
        ).grid(row=0, column=0, sticky=tk.W)
        self.status_label = tk.Label(
            status_frame, text="Ready", bg=self.colors['surface_alt'],
            fg=self.colors['success'], font=('Segoe UI Semibold', 12)
        )
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(4, 2))
        self.status_detail_label = tk.Label(
            status_frame, text="Waiting for input files.", bg=self.colors['surface_alt'],
            fg=self.colors['muted'], font=('Segoe UI', 9)
        )
        self.status_detail_label.grid(row=2, column=0, sticky=tk.W)
        
        # ========== Right Display Area ==========
        right_frame = ttk.Frame(self.root, style='Panel.TFrame', padding="14")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(8, 14), pady=14)
        
        self.notebook = ttk.Notebook(right_frame, style='Custom.TNotebook')
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_frame = ttk.Frame(self.notebook, style='Panel.TFrame')
        self.notebook.add(log_frame, text="Processing Log")
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                  width=80, height=40, 
                                                  font=("Consolas", 9),
                                                  bg=self.colors['console_bg'],
                                                  fg=self.colors['console_fg'],
                                                  insertbackground='white',
                                                  relief='flat',
                                                  padx=12,
                                                  pady=12)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        result_frame = ttk.Frame(self.notebook, style='Panel.TFrame')
        self.notebook.add(result_frame, text="Analysis Results")
        
        self.result_text = scrolledtext.ScrolledText(result_frame, wrap=tk.NONE, 
                                                     width=80, height=40,
                                                     font=("Consolas", 9),
                                                     bg='#f7fbff',
                                                     fg=self.colors['text'],
                                                     insertbackground=self.colors['text'],
                                                     relief='flat',
                                                     padx=12,
                                                     pady=12)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure weights
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        self.use_processed_signal.trace_add('write', self.update_signal_mode_hint)
        self.update_signal_mode_hint()
        self.refresh_file_summary()
        self.set_status("Ready", "success", "Waiting for input files.")
        
        # Initial welcome message
        self.log("="*70)
        self.log("Welcome to Sensor Reading Analysis")
        self.log("FFT-guided drift removal with signed recovery-force preservation")
        self.log("="*70)
        self.log("")
        
        self.root.bind('<Control-o>', lambda _event: self.select_single_file())
        self.root.bind('<Control-Return>', lambda _event: self.start_analysis())
        self.root.bind('<Escape>', lambda _event: self.stop_analysis())

    def create_scrollable_panel(self):
        """Return the left control frame inside a vertically scrolling canvas."""
        container = ttk.Frame(self.root, style='Panel.TFrame')
        container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(14, 8), pady=14)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.left_canvas = tk.Canvas(
            container, bg=self.colors['surface'], highlightthickness=0, borderwidth=0
        )
        scrollbar = ttk.Scrollbar(container, orient='vertical', command=self.left_canvas.yview)
        self.left_canvas.configure(yscrollcommand=scrollbar.set)
        self.left_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        panel = ttk.Frame(self.left_canvas, style='Panel.TFrame', padding="16")
        window_id = self.left_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda _event: self.left_canvas.configure(
            scrollregion=self.left_canvas.bbox('all'), width=panel.winfo_reqwidth()
        ))
        self.left_canvas.bind('<Configure>', lambda event: self.left_canvas.itemconfigure(
            window_id, width=event.width
        ))
        # Wheel scrolling applies only while the pointer is over the panel.
        container.bind('<Enter>', lambda _event: self.root.bind_all('<MouseWheel>', self._scroll_left_panel))
        container.bind('<Leave>', lambda _event: self.root.unbind_all('<MouseWheel>'))
        return panel

    def _scroll_left_panel(self, event):
        if isinstance(event.widget, tk.Listbox):
            return
        self.left_canvas.yview_scroll(int(-event.delta / 120), 'units')

    def format_file_display_name(self, file_path):
        """Keep file list entries readable when multiple folders are selected."""
        path_obj = Path(file_path)
        parent_name = path_obj.parent.name if path_obj.parent.name else str(path_obj.parent)
        return f"{parent_name} / {path_obj.name}"

    def refresh_file_summary(self):
        """Refresh the helper text below the file list."""
        if not self.file_paths:
            self.file_summary_var.set("No files selected yet.")
            return
        folder_count = len({str(Path(file_path).parent) for file_path in self.file_paths})
        self.file_summary_var.set(
            f"{len(self.file_paths)} file(s) loaded from {folder_count} folder(s)."
        )

    def update_signal_mode_hint(self, *_):
        """Explain the currently selected signal mode."""
        if self.use_processed_signal.get():
            self.signal_mode_hint_var.set(
                "Using the low-pass filtered, drift-corrected signal for peak and recovery-force "
                "measurement (unfiltered when the cutoff is at or above Nyquist)."
            )
        else:
            self.signal_mode_hint_var.set(
                "Using the unfiltered, drift-corrected signal when amplitude detail matters more than noise suppression."
            )

    def set_status(self, text, tone='success', detail=''):
        """Update the status card consistently (safe to call from any thread)."""
        self._call_in_ui(lambda: self._apply_status(text, tone, detail))

    def _apply_status(self, text, tone, detail):
        tone_map = {
            'success': self.colors['success'],
            'warning': self.colors['warning'],
            'danger': self.colors['danger'],
            'accent': self.colors['accent']
        }
        self.status_label.config(text=text, fg=tone_map.get(tone, self.colors['text']))
        self.status_detail_label.config(text=detail)
    
    def log(self, message):
        """Append a line to the processing log (safe to call from any thread)."""
        self._call_in_ui(lambda: self._append_log(message))
    
    def _append_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def _call_in_ui(self, callback):
        """Run callback on the Tk thread; other threads queue it for _drain_ui_queue."""
        if threading.current_thread() is self._ui_thread:
            callback()
        else:
            self._ui_queue.put(callback)

    def _drain_ui_queue(self):
        """Apply UI updates queued by worker threads, then poll again."""
        try:
            while True:
                try:
                    callback = self._ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    callback()
                except Exception:
                    details = traceback.format_exc()
                    print(details, file=sys.stderr)
                    try:
                        self._append_log("UI update failed:\n" + details)
                    except tk.TclError:
                        pass  # the log panel itself is unusable; stderr has the error
        finally:
            self.root.after(UI_POLL_INTERVAL_MS, self._drain_ui_queue)

    def clear_log(self):
        """Clear the processing log without changing queued files or settings."""
        self.log_text.delete('1.0', tk.END)
        self.log("Log cleared. Ready for the next operation.")

    def open_results_folder(self):
        """Open the most recently generated result folder in the file manager."""
        if not self.last_output_dirs:
            messagebox.showinfo("Results", "Run an analysis first to create a results folder.")
            return
        if len(self.last_output_dirs) == 1:
            os.startfile(self.last_output_dirs[0])
            return
        folder_list = "\n".join(str(path) for path in self.last_output_dirs)
        if messagebox.askyesno(
            "Multiple result folders",
            f"Open the first folder?\n\n{folder_list}"
        ):
            os.startfile(self.last_output_dirs[0])
        
    def select_single_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Excel or CSV File",
            filetypes=[("Data files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
        )
        if file_path:
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
                self.file_listbox.insert(tk.END, self.format_file_display_name(file_path))
                self.refresh_file_summary()
                self.set_status("Ready", "success", f"{len(self.file_paths)} file(s) queued for analysis.")
                self.log(f"Added file: {os.path.basename(file_path)}")
                if len(self.file_paths) == 1:
                    self.detect_sampling_rate(file_path)
            
    def select_multiple_files(self):
        file_paths = filedialog.askopenfilenames(
            title="Select Excel or CSV Files",
            filetypes=[("Data files", "*.xlsx *.xls *.csv"), ("All files", "*.*")]
        )
        for file_path in file_paths:
            if file_path not in self.file_paths:
                self.file_paths.append(file_path)
                self.file_listbox.insert(tk.END, self.format_file_display_name(file_path))
        if file_paths:
            self.refresh_file_summary()
            self.set_status("Ready", "success", f"{len(self.file_paths)} file(s) queued for analysis.")
            self.log(f"Added {len(file_paths)} files")
            if self.detected_sampling_rate is None and self.file_paths:
                self.detect_sampling_rate(self.file_paths[0])
    
    def select_folders(self):
        """Folder selection dialog with recursive option"""
        folder_dialog = tk.Toplevel(self.root)
        folder_dialog.title("Select Folders")
        folder_dialog.geometry("500x400")
        folder_dialog.transient(self.root)
        folder_dialog.grab_set()
        
        info_label = ttk.Label(folder_dialog, 
                              text="Add folders to automatically read all Excel/CSV files within",
                              font=("Arial", 10))
        info_label.pack(pady=10)
        
        folder_frame = ttk.Frame(folder_dialog)
        folder_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        folder_listbox = tk.Listbox(folder_frame, height=12)
        folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        folder_scrollbar = ttk.Scrollbar(folder_frame, orient="vertical", command=folder_listbox.yview)
        folder_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        folder_listbox.config(yscrollcommand=folder_scrollbar.set)
        
        selected_folders = []
        
        def add_folder():
            directory = filedialog.askdirectory(title="Select Folder")
            if directory and directory not in selected_folders:
                selected_folders.append(directory)
                folder_listbox.insert(tk.END, directory)
                
        def remove_folder():
            selection = folder_listbox.curselection()
            if selection:
                idx = selection[0]
                folder_to_remove = folder_listbox.get(idx)
                folder_listbox.delete(idx)
                selected_folders.remove(folder_to_remove)
        
        def confirm_folders():
            if not selected_folders:
                messagebox.showwarning("Warning", "Please select at least one folder")
                return
            
            total_files_added = 0
            for folder in selected_folders:
                search_pattern = ('.xlsx', '.xls', '.csv')
                
                if recursive_var.get():
                    for root_dir, _, files in os.walk(folder):
                        for file in files:
                            if file.lower().endswith(search_pattern) and not file.startswith('~'):
                                full_path = os.path.join(root_dir, file)
                                if full_path not in self.file_paths:
                                    self.file_paths.append(full_path)
                                    self.file_listbox.insert(tk.END, self.format_file_display_name(full_path))
                                    total_files_added += 1
                else:
                    for file in os.listdir(folder):
                        if file.lower().endswith(search_pattern) and not file.startswith('~'):
                            full_path = os.path.join(folder, file)
                            if full_path not in self.file_paths:
                                self.file_paths.append(full_path)
                                self.file_listbox.insert(tk.END, self.format_file_display_name(full_path))
                                total_files_added += 1
            
            self.refresh_file_summary()
            self.set_status("Ready", "success", f"{len(self.file_paths)} file(s) queued for analysis.")
            self.log(f"Added {total_files_added} files from folders")
            if self.detected_sampling_rate is None and self.file_paths:
                self.detect_sampling_rate(self.file_paths[0])
            folder_dialog.destroy()
        
        button_frame = ttk.Frame(folder_dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Add Folder", command=add_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove", command=remove_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Confirm", command=confirm_folders).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=folder_dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(folder_dialog, text="Include subfolders (recursive scan)", 
                       variable=recursive_var).pack(pady=5)
    
    def detect_sampling_rate(self, file_path):
        """Detect sampling rate from time column"""
        try:
            self.log(f"\nDetecting sampling rate from: {os.path.basename(file_path)}")
            
            df = read_recording_table(file_path)
            
            time_col = None
            for col in df.columns:
                if 'time' in str(col).lower():
                    time_col = col
                    break
            
            if time_col is None:
                time_col = df.columns[0]
            
            time_data = pd.to_numeric(df[time_col], errors='coerce').values
            time_data = time_data[~np.isnan(time_data)]
            
            # Repeated timestamps (the pre-actuation rest block) carry no interval.
            time_diffs = np.diff(time_data)
            time_diffs = time_diffs[time_diffs > 0]
            if time_diffs.size > 0:
                median_dt = np.median(time_diffs)
                
                if median_dt > 0:
                    detected_fs = 1.0 / median_dt
                    self.detected_sampling_rate = detected_fs
                    self.sampling_rate_var.set(f"{detected_fs:.2f} Hz")
                    self.set_status("Ready", "success", f"Sampling rate detected: {detected_fs:.2f} Hz")
                    self.log(f"  Detected: {detected_fs:.2f} Hz (dt={median_dt*1000:.2f} ms)")
                else:
                    raise ValueError("Invalid time differences")
            else:
                raise ValueError("Not enough data points")
                
        except Exception as e:
            self.log(f"  Warning: Could not auto-detect: {e}")
            self.detected_sampling_rate = 100.0
            self.sampling_rate_var.set("100.0 Hz (default)")
            self.set_status("Ready", "warning", "Sampling rate fallback applied: 100.0 Hz")
            
    def clear_files(self):
        self.file_paths = []
        self.file_listbox.delete(0, tk.END)
        self.detected_sampling_rate = None
        self.sampling_rate_var.set("Auto-detect")
        self.refresh_file_summary()
        self.set_status("Ready", "success", "Waiting for input files.")
        self.log("File list cleared")
    
    def quick_diagnostic(self):
        """Quick diagnostic on first file with auto period detection"""
        if not self.file_paths:
            messagebox.showwarning("Warning", "Please select at least one file first.")
            return
        
        file_path = self.file_paths[0]
        self.log("\n" + "="*70)
        self.log("🔍 QUICK DIAGNOSTIC (Auto Period Detection)")
        self.log(f"File: {os.path.basename(file_path)}")
        self.log("="*70)
        
        try:
            df = read_recording_table(file_path)
            
            time_cols = [c for c in df.columns if 'time' in str(c).lower()]
            thrust_cols = [c for c in df.columns if 'thrust' in str(c).lower() or 'force' in str(c).lower()]
            
            if not time_cols or not thrust_cols:
                self.log("❌ Cannot find time or thrust columns")
                return
            
            t = pd.to_numeric(df[time_cols[0]], errors='coerce').to_numpy()
            y = pd.to_numeric(df[thrust_cols[0]], errors='coerce').to_numpy()
            
            valid_mask = ~np.isnan(t) & ~np.isnan(y)
            t, y = t[valid_mask], y[valid_mask]
            
            duration = t.max() - t.min()
            
            # Auto-detect period using FFT
            if self.detected_sampling_rate:
                fs = self.detected_sampling_rate
                n = len(y)
                yf = np.fft.fft(y - np.mean(y))
                xf = np.fft.fftfreq(n, 1/fs)
                positive_freqs = xf[:n//2]
                positive_power = np.abs(yf[:n//2])
                
                # Find dominant frequency (exclude DC)
                min_freq_idx = max(1, int(0.1 * n / fs))  # At least 0.1 Hz
                peak_idx = min_freq_idx + np.argmax(positive_power[min_freq_idx:])
                peak_freq = positive_freqs[peak_idx]
                detected_period = 1.0 / peak_freq if peak_freq > 0 else 1.0
                
                self.log(f"\n📊 Data Information:")
                self.log(f"   • Data points: {len(t)}")
                self.log(f"   • Duration: {duration:.2f}s")
                self.log(f"   • Thrust range: {y.min():.4f} - {y.max():.4f} N")
                self.log(f"   • 🔄 Auto-detected period: {detected_period:.3f}s")
                self.log(f"   • Estimated cycles: {duration / detected_period:.1f}")
                
                self.detected_period_var.set(f"{detected_period:.3f} s")
            else:
                self.log("⚠️ Please detect sampling rate first")
                
        except Exception as e:
            self.log(f"❌ Diagnostic failed: {e}")
    
    def start_analysis(self):
        if not self.file_paths:
            messagebox.showwarning("Warning", "Please select files first!")
            return
        
        if self.is_processing:
            messagebox.showinfo("Info", "Analysis is already running...")
            return
        
        try:
            if self.detected_sampling_rate is None:
                raise ValueError("Sampling rate not detected.")
            cutoff_freq = float(self.cutoff_freq_var.get())
            if not math.isfinite(cutoff_freq) or cutoff_freq <= 0.0:
                raise ValueError("the low-pass cutoff must be a positive frequency")
            tare_duration_s = parse_optional_seconds(self.tare_duration_var.get(), allow_auto=True)
            post_rest_duration_s = parse_optional_seconds(self.post_rest_var.get(), allow_auto=False)
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return
        
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.is_processing = True
        self.stop_requested = False
        self.progress.start()
        
        # Tk variables are read here on the Tk thread; the worker only gets plain values.
        thread = threading.Thread(target=self.run_analysis, args=(
            list(self.file_paths), self.detected_sampling_rate, cutoff_freq,
            self.use_processed_signal.get(), tare_duration_s, post_rest_duration_s
        ))
        thread.daemon = True
        thread.start()
    
    def stop_analysis(self):
        if not self.is_processing or self.stop_requested:
            return
        self.stop_requested = True
        self.stop_button.config(state="disabled")
        self.set_status("Stopping...", "warning", "Finishing the current file before stopping.")
        self.log("\nStop requested; finishing the current file...\n")

    def run_analysis(self, file_paths, sampling_rate, cutoff_freq, use_processed_signal,
                     tare_duration_s, post_rest_duration_s):
        """Analyze files on a worker thread; every UI update goes through the UI queue."""
        all_results = []
        failed_files = []
        summary_paths = []
        error_message = None
        attempted = 0
        try:
            self.set_status("Processing...", "accent", f"Analyzing {len(file_paths)} file(s).")

            self.log("\n" + "="*70)
            self.log(f"Starting batch analysis - {len(file_paths)} files")
            self.log("="*70)
            self.log(f"Sampling rate: {sampling_rate:.2f} Hz")
            self.log("Cycle period: Auto-detected (FFT)")
            if tare_duration_s is None:
                self.log("Rest before actuation: auto (rest samples at the first timestamp)")
            else:
                self.log(f"Rest before actuation: {tare_duration_s:g} s")
            self.log(f"Rest after actuation: {post_rest_duration_s:g} s")
            self.log("")

            analyzer = ProfessionalThrustAnalyzer(
                sampling_rate,
                self.log,
                use_processed_signal=use_processed_signal,
                tare_duration_s=tare_duration_s,
                post_rest_duration_s=post_rest_duration_s
            )

            for i, file_path in enumerate(file_paths, 1):
                if self.stop_requested:
                    break
                attempted = i

                self.log(f"\n[{i}/{len(file_paths)}] {os.path.basename(file_path)}")
                self.log("-"*70)

                try:
                    result = analyzer.analyze_file(file_path, cutoff_freq=cutoff_freq)
                except Exception as e:
                    failed_files.append(os.path.basename(file_path))
                    self.log(f"Error: {e}")
                    self.log(traceback.format_exc())
                    continue
                if result:
                    all_results.append(result)
                else:
                    failed_files.append(os.path.basename(file_path))

            # Finished files are summarized even when the run was stopped early.
            if all_results:
                summary_paths = self._save_and_show_results(all_results)

        except Exception as e:
            error_message = str(e)
            self.log(f"\nError: {e}")
            self.log(traceback.format_exc())

        finally:
            # A stop pressed while the last file was running skipped nothing.
            stopped = self.stop_requested and attempted < len(file_paths)
            self._call_in_ui(lambda: self._finish_processing(
                len(all_results), failed_files, summary_paths, stopped, error_message
            ))

    def _save_and_show_results(self, all_results):
        """Write the summary workbooks (worker thread) and queue the result display."""
        self.log("\n" + "="*70)
        self.log("Generating summary...")
        self.log("="*70)

        summary_paths = save_summary_reports(all_results)
        output_dirs = sorted({path.parent for path in summary_paths}, key=str)
        average_period = float(np.mean([r['detected_period'] for r in all_results]))

        self.log(f"\n✅ Summary saved for {len(all_results)} file(s):")
        for summary_path in summary_paths:
            self.log(f"   - {summary_path}")

        def show_results():
            self.detected_period_var.set(f"{average_period:.3f} s")
            self.last_output_dirs = output_dirs
            self.open_results_button.config(state='normal')
            self.show_summary(all_results)

        self._call_in_ui(show_results)
        return summary_paths

    def _finish_processing(self, result_count, failed_files, summary_paths, stopped, error_message):
        """Restore the controls and report the outcome on the Tk thread."""
        self.progress.stop()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.is_processing = False
        self.stop_requested = False
    
        if error_message:
            self.set_status("Error", "danger", error_message)
        elif stopped:
            saved = "their summary was saved" if result_count else "nothing was saved"
            self.set_status("Stopped", "warning", f"Stopped after {result_count} analyzed file(s); {saved}.")
            self.log("\nAnalysis stopped\n")
        elif not result_count:
            self.set_status("No results", "warning",
                            f"All {len(failed_files)} file(s) failed; see the processing log.")
        else:
            self._report_completion(result_count, failed_files, summary_paths)
            
    def _report_completion(self, result_count, failed_files, summary_paths):
        """Announce a finished run, naming every file that could not be analyzed."""
        if len(summary_paths) == 1:
            summary_message = f"Summary: {summary_paths[0]}"
        else:
            summary_message = f"Saved {len(summary_paths)} summary files (one per source folder)"
        self.log("\n" + "="*70)
        if failed_files:
            self.log(f"⚠️ Analysis complete; {len(failed_files)} file(s) failed: {', '.join(failed_files)}")
            self.log("="*70)
            self.set_status("Complete with errors", "warning",
                            f"{result_count} file(s) analyzed, {len(failed_files)} failed; see the processing log.")
            failed_list = "\n".join(failed_files)
            messagebox.showwarning(
                "Complete with errors",
                f"Analyzed {result_count} file(s); {len(failed_files)} failed:\n{failed_list}\n\n{summary_message}"
            )
            return
        self.log("✅ Analysis complete!")
        self.log("="*70)
        self.set_status(
            "Complete", "success",
            f"Finished {result_count} file(s); results saved in {len(self.last_output_dirs)} folder(s)."
        )
        messagebox.showinfo("Complete", f"Analyzed {result_count} files!\n\n{summary_message}")
    
    def show_summary(self, results):
        """Display summary in results tab"""
        if not results:
            return
        
        display_data = []
        for r in results:
            display_data.append({
                'File': r['file_name'],
                'Cycles': r['num_cycles'],
                'Mean Force(N)': f"{r['mean_cycle_force']:.4f}",
                'Tare-only(N)': f"{r['mean_cycle_force_tare_only']:.4f}",
                'Peak(N)': f"{r['mean_peak_thrust']:.4f}",
                'Recovery(N)': f"{r['mean_recovery_force']:.4f}",
                'Peak-Valley(N)': f"{r['mean_peak_to_valley']:.4f}",
                'Period(s)': f"{r['detected_period']:.3f}"
            })
        
        summary_df = pd.DataFrame(display_data)
        
        self.result_text.delete(1.0, tk.END)
        header = f"{'='*80}\nTHRUST ANALYSIS SUMMARY\n{'='*80}\n\n"
        self.result_text.insert(tk.END, header + summary_df.to_string(index=False) + "\n")
        self.notebook.select(1)


def _sample_std(values):
    """Sample standard deviation across cycles (0 for a single cycle)."""
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


class ProfessionalThrustAnalyzer:
    """Professional Thrust Analyzer - Core Algorithm (Unified)"""
    
    def __init__(self, sampling_rate, log_func=None, use_processed_signal=True,
                 tare_duration_s=None, post_rest_duration_s=None):
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.log = log_func if log_func else print
        self.use_processed_signal = use_processed_signal
        self.tare_duration_s = tare_duration_s
        self.post_rest_duration_s = post_rest_duration_s
        self.dynamic_baseline_result = None
        self.lowpass_description = "not applied"
    
    def load_data(self, file_path):
        """Load data from file"""
        df = read_recording_table(file_path)
        
        # Auto-detect time column
        time_col = None
        for col in df.columns:
            if 'time' in str(col).lower():
                time_col = col
                break
        
        # Auto-detect force column
        force_col = None
        for col in df.columns:
            if any(name in str(col).lower() for name in ['force', 'thrust']):
                force_col = col
                break
        
        if force_col is None:
            force_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        force_raw = pd.to_numeric(df[force_col], errors='coerce').values
        
        if time_col and time_col in df.columns:
            t = pd.to_numeric(df[time_col], errors='coerce').values
        else:
            t = np.arange(len(force_raw)) * self.dt
        
        # Remove NaN
        valid_mask = ~np.isnan(t) & ~np.isnan(force_raw)
        t = t[valid_mask]
        force_raw = force_raw[valid_mask]
        
        return t, force_raw
    
    def process_signal(self, force_raw, cutoff_freq, t=None):
        """Separate slow sensor drift from the signed periodic force waveform."""
        if t is None:
            t = np.arange(len(force_raw), dtype=float) / self.fs
        result = analyze_dynamic_baseline(
            t, force_raw,
            tare_duration_s=self.tare_duration_s,
            post_rest_duration_s=self.post_rest_duration_s
        )
        self.dynamic_baseline_result = result
        self.fs = result.sampling_rate_hz
        self.dt = 1.0 / self.fs
        baseline_curve = np.asarray(result.baseline)
        force_debiased = np.asarray(result.corrected_force)

        self.log(f"  FFT frequency: {result.frequency_hz:.4f} Hz ({result.period_s:.4f} s)")
        self.log(f"  Pre-actuation tare: {result.baseline_anchor_n:.4f} N ({result.tare_sample_count} samples)")
        if result.post_rest_force_n is not None:
            self.log(f"  Post-run rest: {result.post_rest_force_n:.4f} N")
        self.log(f"  Drift model: {result.drift_model}")
        baseline_min = np.min(baseline_curve)
        baseline_max = np.max(baseline_curve)
        baseline_drift = baseline_max - baseline_min
        self.log(f"  Baseline range: [{baseline_min:.4f}, {baseline_max:.4f}] N, drift: {baseline_drift:.4f} N")

        force_filtered = self._lowpass(t, force_debiased, cutoff_freq, result.frequency_hz)
        return force_debiased, force_filtered, baseline_curve
    
    def _lowpass(self, t, force_debiased, cutoff_freq, cycle_frequency_hz):
        """Low-pass the drift-corrected force; a cutoff at or above Nyquist leaves it unchanged."""
        nyquist = 0.5 * self.fs
        if cutoff_freq >= nyquist:
            # No recorded frequency lies above such a cutoff, so any filter would
            # only remove real signal close to Nyquist.
            self.lowpass_description = f"low-pass off: cutoff >= Nyquist {nyquist:.2f} Hz"
            self.log(f"  Low-pass off: cutoff {cutoff_freq:.2f} Hz >= Nyquist {nyquist:.2f} Hz")
            return np.array(force_debiased, dtype=float, copy=True)
        self.lowpass_description = f"low-pass {cutoff_freq:.2f} Hz"
        if cutoff_freq < HARMONIC_SAFE_CUTOFF_MULTIPLE * cycle_frequency_hz:
            self.log(
                f"  Warning: cutoff {cutoff_freq:.2f} Hz is below "
                f"{HARMONIC_SAFE_CUTOFF_MULTIPLE:g}x the cycle frequency; peaks will be attenuated"
            )
        return lowpass_irregular_signal(t, force_debiased, self.fs, cutoff_freq)

    def measure_cycles(self, t, y_original, y_processed):
        """Measure every complete cycle of the recording last passed to process_signal.

        Peaks and recovery valleys come from the selected measurement signal; impulse
        and mean force always use the unfiltered drift-corrected force.
        """
        result = self.dynamic_baseline_result
        if result is None or len(result.corrected_force) != len(t):
            raise ValueError("process_signal must run on this recording before measure_cycles")
        measurement = y_processed if self.use_processed_signal else y_original
        peaks, valleys = cycle_extrema(result, measurement)
        boundaries = result.cycle_boundaries
        thrust_cycles = []
        for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            peak_val = float(measurement[peaks[index]])
            valley_val = float(measurement[valleys[index]])
            thrust_cycles.append({
                'cycle_start_idx': int(start),
                'cycle_end_idx': int(stop),
                'peak_idx': int(peaks[index]),
                'valley_idx': int(valleys[index]),
                'peak_value': peak_val,
                'recovery_value': valley_val,
                'peak_to_valley': peak_val - valley_val,
                'mean_force': float(result.cycle_mean_forces[index]),
                'mean_force_tare_only': float(result.tare_only_cycle_mean_forces[index]),
                'impulse': float(result.cycle_impulses[index])
            })
        self.log(f"  Detected {len(thrust_cycles)} complete, non-overlapping cycles")
        return thrust_cycles, result.period_s
        
    def _signal_description(self):
        if self.use_processed_signal:
            return f"processed ({self.lowpass_description})"
        return "original (unfiltered)"
    
    def plot_results(self, t, force_raw, force_filtered, cycle_results, 
                    baseline_curve, file_name, output_dir):
        """Plot SVG chart with 6 panels"""
        fig = plt.figure(figsize=(16, 10))
        force_debiased = force_raw - baseline_curve
        
        # Panel 1: Original + baseline
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t, force_raw, 'b-', linewidth=0.8, alpha=0.6, label='Raw')
        ax1.plot(t, baseline_curve, 'r-', linewidth=2,
                 label=f'Sensor baseline ({self.dynamic_baseline_result.drift_model})')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Raw Data & Sensor Baseline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Debiased
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(t, force_debiased, 'g-', linewidth=0.8, alpha=0.6)
        ax2.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax2.set_ylabel('Force (N)')
        ax2.set_title('Drift Removed (Signed Force Preserved)')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Filtered
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(t, force_debiased, 'gray', linewidth=0.5, alpha=0.4, label='Debiased')
        ax3.plot(t, force_filtered, 'r-', linewidth=1.5, label='Filtered')
        ax3.set_ylabel('Force (N)')
        ax3.set_title(f'Low-Pass Filter ({self.lowpass_description})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Noise
        ax4 = plt.subplot(3, 2, 4)
        noise = force_debiased - force_filtered
        ax4.plot(t, noise, 'orange', linewidth=0.5, alpha=0.7)
        ax4.set_ylabel('Force (N)')
        ax4.set_title('Removed Noise' if np.any(noise) else 'Removed Noise (none: low-pass off)')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Cycle detection on the measured signal
        ax5 = plt.subplot(3, 2, 5)
        measurement = force_filtered if self.use_processed_signal else force_debiased
        ax5.plot(t, measurement, 'b-', linewidth=1.5)
        
        for cycle in cycle_results:
            ax5.plot(t[cycle['peak_idx']], cycle['peak_value'], 'ro', markersize=8)
            ax5.plot(t[cycle['valley_idx']], cycle['recovery_value'], 'mo', markersize=6)
        
        mean_force = float(np.mean([c['mean_force'] for c in cycle_results]))
        ax5.axhline(y=mean_force, color='g', linestyle='--', linewidth=1.2,
                    label=f'Mean cycle force {mean_force:.3f} N')
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Force (N)')
        ax5.set_title(f'Cycle Detection (n={len(cycle_results)})')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Thrust vs Drag
        ax6 = plt.subplot(3, 2, 6)
        if cycle_results:
            cycle_nums = range(1, len(cycle_results)+1)
            thrusts = [c['peak_value'] for c in cycle_results]
            recoveries = [c['recovery_value'] for c in cycle_results]
            
            width = 0.35
            x = np.arange(len(cycle_nums))
            
            ax6.bar(x - width/2, thrusts, width, label='Peak thrust', color='green', alpha=0.7)
            ax6.bar(x + width/2, recoveries, width, label='Recovery force', color='red', alpha=0.7)
            ax6.axhline(y=0, color='k', linestyle='-', linewidth=1)
            ax6.set_xlabel('Cycle')
            ax6.set_ylabel('Force (N)')
            ax6.set_title('Peak Thrust & Signed Recovery Force')
            ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Thrust Analysis - {file_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        svg_path = Path(output_dir) / f"{file_name}_analysis.svg"
        plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return svg_path
    
    def analyze_file(self, file_path, cutoff_freq):
        """Analyze single file with auto period detection"""
        file_path_obj = Path(file_path)
        file_name = file_path_obj.stem
        folder_name = file_path_obj.parent.name
        
        # Load data
        t, force_raw = self.load_data(file_path)
        self.log(f"  Data: {len(force_raw)} points, {t[-1] - t[0]:.2f}s")
        
        # Signal processing
        force_debiased, force_filtered, baseline_curve = self.process_signal(
            force_raw, cutoff_freq, t=t
        )
        
        # Cycle measurement
        cycle_results, detected_period = self.measure_cycles(
            t, force_debiased, force_filtered
        )
        
        if not cycle_results:
            self.log("  ⚠️ No cycles detected")
            return None
        
        # Statistics
        result = self.dynamic_baseline_result
        cycle_forces = np.array([c['mean_force'] for c in cycle_results])
        tare_only_forces = np.array([c['mean_force_tare_only'] for c in cycle_results])
        peak_values = np.array([c['peak_value'] for c in cycle_results])
        recovery_values = np.array([c['recovery_value'] for c in cycle_results])
        peak_to_valley = np.array([c['peak_to_valley'] for c in cycle_results])
        impulses = np.array([c['impulse'] for c in cycle_results])
        
        forward_cycles = int(np.sum(impulses > 0))
        
        self.log(f"  Mean cycle force (net thrust): {cycle_forces.mean():.4f} N "
                 f"(tare only, no drift correction: {tare_only_forces.mean():.4f} N)")
        self.log(f"  Mean peak thrust: {peak_values.mean():.4f} N")
        self.log(f"  Mean signed recovery force: {recovery_values.mean():.4f} N")
        self.log(f"  Mean peak-to-valley: {peak_to_valley.mean():.4f} N")
        self.log(f"  Forward: {forward_cycles}/{len(cycle_results)}")
        
        # Create output directory
        output_dir = file_path_obj.parent / RESULTS_DIR_NAME
        output_dir.mkdir(exist_ok=True)
        
        # Plot SVG
        svg_path = self.plot_results(
            t, force_raw, force_filtered, cycle_results,
            baseline_curve, file_name, output_dir
        )
        self.log(f"  SVG: {svg_path.name}")
        
        return {
            'file_name': file_name,
            'folder_name': folder_name,
            'source_folder': str(file_path_obj.parent),
            'data_duration': float(t[-1] - t[0]),
            'num_cycles': len(cycle_results),
            'forward_cycles': forward_cycles,
            'backward_cycles': len(cycle_results) - forward_cycles,
            'mean_cycle_force': float(cycle_forces.mean()),
            'std_cycle_force': _sample_std(cycle_forces),
            'mean_cycle_force_tare_only': float(tare_only_forces.mean()),
            'mean_peak_thrust': float(peak_values.mean()),
            'mean_recovery_force': float(recovery_values.mean()),
            'mean_peak_to_valley': float(peak_to_valley.mean()),
            'std_peak_to_valley': _sample_std(peak_to_valley),
            'mean_impulse': float(impulses.mean()),
            'std_impulse': _sample_std(impulses),
            'total_impulse': float(impulses.sum()),
            'detected_period': float(detected_period),
            'detected_frequency': 1.0 / float(detected_period),
            'drift_model': result.drift_model,
            'tare_force': float(result.baseline_anchor_n),
            'tare_samples': int(result.tare_sample_count),
            'post_rest_force': result.post_rest_force_n,
            'signal_description': self._signal_description(),
            'baseline_start': float(baseline_curve[0]),
            'baseline_end': float(baseline_curve[-1]),
            'baseline_drift': float(np.max(baseline_curve) - np.min(baseline_curve))
        }


def main():
    root = tk.Tk()
    app = ThrustAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

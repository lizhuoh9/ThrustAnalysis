import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues with Tkinter
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from pathlib import Path
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import warnings
warnings.filterwarnings('ignore')

from dynamic_baseline import analyze_dynamic_baseline, lowpass_irregular_signal

# Set up matplotlib for English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ThrustAnalyzerGUI:
    """Unified Thrust Analysis GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Thrust Analysis System V3.0 (Unified)")
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
        self.is_processing = False
        self.detected_sampling_rate = None
        
        # Options from data_analysis.py
        self.auto_mode = tk.BooleanVar(value=True)
        self.adaptive_period = tk.BooleanVar(value=True)
        self.use_processed_signal = tk.BooleanVar(value=True)
        self.file_summary_var = tk.StringVar(value="No files selected yet.")
        self.signal_mode_hint_var = tk.StringVar()
        
        self.configure_styles()
        self.create_widgets()

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

        # Left Control Panel
        left_frame = ttk.Frame(self.root, style='Panel.TFrame', padding="16")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(14, 8), pady=14)
        left_frame.columnconfigure(0, weight=1)
        left_frame.columnconfigure(1, weight=1)
        
        hero_frame = tk.Frame(left_frame, bg=self.colors['accent'], padx=18, pady=14)
        hero_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 12))
        tk.Label(
            hero_frame,
            text="Thrust Analysis Workspace",
            bg=self.colors['accent'],
            fg='white',
            font=('Segoe UI Semibold', 16)
        ).grid(row=0, column=0, sticky=tk.W)
        tk.Label(
            hero_frame,
            text="Processed signal is selected by default for cleaner, more stable cycle measurement.",
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
        sr_frame = ttk.LabelFrame(left_frame, text="2. Sampling", padding="12", style='Section.TLabelframe')
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
        ttk.Label(adv_frame, textvariable=self.signal_mode_hint_var, style='Muted.TLabel').grid(
            row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0)
        )
        
        # Manual parameters
        ttk.Label(adv_frame, text="Low-pass Cutoff (Hz)", style='Muted.TLabel').grid(
            row=5, column=0, sticky=tk.W, pady=(12, 4)
        )
        self.cutoff_freq_var = tk.StringVar(value="5.0")
        self.cutoff_entry = ttk.Entry(adv_frame, textvariable=self.cutoff_freq_var, width=10)
        self.cutoff_entry.grid(row=5, column=1, sticky=tk.W, padx=5, pady=(12, 4))
        self.cutoff_hint_label = ttk.Label(
            adv_frame,
            text="Peak detection is automatic. Keep 5.0 Hz unless your motion is much faster.",
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
                  command=self.quick_diagnostic).grid(row=1, column=0, columnspan=2, sticky='ew')
        
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
        self.log("Welcome to Professional Thrust Analysis System V3.0 (Unified)")
        self.log("Combines the best features from both analysis programs")
        self.log("="*70)
        self.log("")
        
        self.toggle_auto_mode()

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
                "Using the filtered, drift-corrected signal for peak and recovery-force measurement."
            )
        else:
            self.signal_mode_hint_var.set(
                "Using the unfiltered, drift-corrected signal when amplitude detail matters more than noise suppression."
            )

    def set_status(self, text, tone='success', detail=''):
        """Update the status card consistently."""
        tone_map = {
            'success': self.colors['success'],
            'warning': self.colors['warning'],
            'danger': self.colors['danger'],
            'accent': self.colors['accent']
        }
        self.status_label.config(text=text, fg=tone_map.get(tone, self.colors['text']))
        self.status_detail_label.config(text=detail)
    
    def toggle_auto_mode(self):
        """Enable/disable manual parameter inputs"""
        if self.auto_mode.get():
            self.cutoff_entry.config(state='disabled')
            self.cutoff_hint_label.config(
                text="Peak detection is automatic. Keep 5.0 Hz unless your motion is much faster."
            )
            self.log("Auto-detection mode enabled")
        else:
            self.cutoff_entry.config(state='normal')
            self.cutoff_hint_label.config(
                text="Manual mode enabled. Adjust the cutoff only if you need stronger or weaker smoothing."
            )
            self.log("Manual parameter mode enabled")
    
    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
        
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
            
            try:
                df = pd.read_excel(file_path)
            except:
                df = pd.read_csv(file_path)
            
            time_col = None
            for col in df.columns:
                if 'time' in str(col).lower():
                    time_col = col
                    break
            
            if time_col is None:
                time_col = df.columns[0]
            
            time_data = pd.to_numeric(df[time_col], errors='coerce').values
            time_data = time_data[~np.isnan(time_data)]
            
            if len(time_data) > 1:
                time_diffs = np.diff(time_data)
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
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
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
            
            sampling_rate = self.detected_sampling_rate
            cutoff_freq = float(self.cutoff_freq_var.get())
                
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameters: {e}")
            return
        
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.is_processing = True
        self.progress.start()
        
        thread = threading.Thread(target=self.run_analysis, args=(
            sampling_rate, cutoff_freq
        ))
        thread.daemon = True
        thread.start()
    
    def stop_analysis(self):
        self.is_processing = False
        self.progress.stop()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.set_status("Stopped", "warning", "Analysis was stopped before completion.")
        self.log("\nAnalysis stopped\n")
    
    def run_analysis(self, sampling_rate, cutoff_freq):
        """Run analysis in background thread with auto period detection"""
        try:
            self.set_status("Processing...", "accent", f"Analyzing {len(self.file_paths)} file(s).")
            
            self.log("\n" + "="*70)
            self.log(f"Starting batch analysis - {len(self.file_paths)} files")
            self.log("="*70)
            self.log(f"Sampling rate: {sampling_rate:.2f} Hz")
            self.log(f"Cycle period: Auto-detected (FFT)")
            self.log(f"Auto-detect params: {'ON' if self.auto_mode.get() else 'OFF'}")
            self.log("")
            
            # Create analyzer (always use adaptive period)
            analyzer = ProfessionalThrustAnalyzer(
                sampling_rate, 
                self.log,
                auto_mode=self.auto_mode.get(),
                adaptive_period=True,  # Always use FFT
                use_processed_signal=self.use_processed_signal.get()
            )
            
            # Analyze each file
            all_results = []
            detected_periods = []
            for i, file_path in enumerate(self.file_paths, 1):
                if not self.is_processing:
                    break
                
                self.log(f"\n[{i}/{len(self.file_paths)}] {os.path.basename(file_path)}")
                self.log("-"*70)
                
                try:
                    result = analyzer.analyze_file(
                        file_path,
                        cutoff_freq=cutoff_freq
                    )
                    if result:
                        all_results.append(result)
                        if result.get('detected_period'):
                            detected_periods.append(result['detected_period'])
                        
                except Exception as e:
                    self.log(f"Error: {e}")
                    import traceback
                    self.log(traceback.format_exc())
            
            # Update detected period display
            if detected_periods:
                avg_period = np.mean(detected_periods)
                self.detected_period_var.set(f"{avg_period:.3f} s")
            
            if self.is_processing and all_results:
                # Generate summary
                self.log("\n" + "="*70)
                self.log("Generating summary...")
                self.log("="*70)
                
                # Create summary DataFrame - ONE ROW PER FILE
                summary_data = []
                for r in all_results:
                    summary_data.append({
                        'Folder': r.get('folder_name', 'N/A'),
                        'File': r['file_name'],
                        'Duration(s)': f"{r['data_duration']:.2f}",
                        'Cycles': r['num_cycles'],
                        'Forward': r.get('forward_cycles', 'N/A'),
                        'Backward': r.get('backward_cycles', 'N/A'),
                        'Mean_Thrust(N)': f"{r['mean_thrust']:.4f}",
                        'Mean_Recovery_Force(N)': f"{r['mean_recovery_force']:.4f}",
                        'Mean_Cycle_Force(N)': f"{r['mean_cycle_force']:.4f}",
                        'Mean_Net(N)': f"{r['mean_net_thrust']:.4f}",
                        'Std_Net(N)': f"{r['std_net_thrust']:.4f}",
                        'Mean_Impulse(N·s)': f"{r.get('mean_impulse', 0):.6f}",
                        'Total_Impulse(N·s)': f"{r.get('total_impulse', 0):.6f}",
                        'Detected_Period(s)': f"{r['detected_period']:.3f}" if r.get('detected_period') else 'N/A',
                        'Detected_Frequency(Hz)': f"{r['detected_frequency']:.4f}",
                        'Baseline_Drift(N)': f"{r['baseline_drift']:.4f}"
                    })
                
                summary_df = self.build_summary_dataframe(all_results)
                
                # Save summary files grouped by source folder
                summary_paths = self.save_summary_reports(all_results)
                if len(summary_paths) == 1:
                    summary_path = summary_paths[0]
                else:
                    summary_path = Path(f"{len(summary_paths)} summary files")
                output_dir = summary_paths[0].parent
                
                self.log(f"\n✅ Summary saved: {summary_path.name}")
                self.log(f"   Total files: {len(all_results)}")
                self.log(f"   Summary files: {len(summary_paths)}")
                for summary_path in summary_paths:
                    self.log(f"   - {summary_path}")
                
                # Show in results tab
                self.show_summary(all_results)
                
                self.log("\n" + "="*70)
                self.log("✅ Analysis complete!")
                self.log("="*70)
                
                self.set_status("Complete", "success", f"Finished analyzing {len(all_results)} file(s).")
                if len(summary_paths) == 1:
                    summary_message = f"Summary: {summary_paths[0]}"
                else:
                    summary_message = (
                        f"Saved {len(summary_paths)} summary files "
                        f"(one per source folder)"
                    )
                messagebox.showinfo("Complete", 
                    f"Analyzed {len(all_results)} files!\n\n{summary_message}")
            
        except Exception as e:
            self.log(f"\nError: {e}")
            import traceback
            self.log(traceback.format_exc())
            self.set_status("Error", "danger", str(e))
        
        finally:
            self.progress.stop()
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.is_processing = False
    
    def show_summary(self, results):
        """Display summary in results tab"""
        if not results:
            return
        
        display_data = []
        for r in results:
            row = {
                'File': r.get('file_name', 'N/A'),
                'Cycles': r.get('num_cycles', 0),
                'Mean Net(N)': f"{r.get('mean_net_thrust', 0):.4f}",
                'Mean Cycle Force(N)': f"{r.get('mean_cycle_force', 0):.4f}",
                'Mean Recovery(N)': f"{r.get('mean_recovery_force', 0):.4f}",
                'Mean Impulse': f"{r.get('mean_impulse', 0):.6f}",
                'Detected Period(s)': f"{r.get('detected_period', 0):.3f}" if r.get('detected_period') else 'N/A'
            }
            display_data.append(row)
        
        summary_df = pd.DataFrame(display_data)
        
        self.result_text.delete(1.0, tk.END)
        header = f"{'='*80}\nTHRUST ANALYSIS SUMMARY\n{'='*80}\n\n"
        self.result_text.insert(tk.END, header + summary_df.to_string(index=False) + "\n")
        self.notebook.select(1)

    def build_summary_dataframe(self, results):
        """Create a summary table with one row per analyzed file."""
        summary_data = []
        for r in results:
            summary_data.append({
                'Folder': r.get('folder_name', 'N/A'),
                'Folder_Path': r.get('source_folder', 'N/A'),
                'File': r['file_name'],
                'Duration(s)': f"{r['data_duration']:.2f}",
                'Cycles': r['num_cycles'],
                'Forward': r.get('forward_cycles', 'N/A'),
                'Backward': r.get('backward_cycles', 'N/A'),
                'Mean_Thrust(N)': f"{r['mean_thrust']:.4f}",
                'Mean_Recovery_Force(N)': f"{r['mean_recovery_force']:.4f}",
                'Mean_Cycle_Force(N)': f"{r['mean_cycle_force']:.4f}",
                'Mean_Net(N)': f"{r['mean_net_thrust']:.4f}",
                'Std_Net(N)': f"{r['std_net_thrust']:.4f}",
                'Mean_Impulse(N路s)': f"{r.get('mean_impulse', 0):.6f}",
                'Total_Impulse(N路s)': f"{r.get('total_impulse', 0):.6f}",
                'Detected_Period(s)': f"{r['detected_period']:.3f}" if r.get('detected_period') else 'N/A',
                'Detected_Frequency(Hz)': f"{r['detected_frequency']:.4f}",
                'Baseline_Start(N)': f"{r['baseline_start']:.4f}",
                'Baseline_End(N)': f"{r['baseline_end']:.4f}",
                'Baseline_Drift(N)': f"{r['baseline_drift']:.4f}"
            })
        return pd.DataFrame(summary_data)

    def save_summary_reports(self, results):
        """Save one batch summary per source folder."""
        if not results:
            return []

        grouped_results = {}
        for result in results:
            source_folder = result.get('source_folder')
            if not source_folder:
                source_folder = str(Path(self.file_paths[0]).parent)
            grouped_results.setdefault(source_folder, []).append(result)

        summary_paths = []
        for source_folder in sorted(grouped_results):
            summary_df = self.build_summary_dataframe(grouped_results[source_folder])
            output_dir = Path(source_folder) / "thrust_analysis_results"
            output_dir.mkdir(exist_ok=True)
            summary_path = output_dir / "batch_summary.xlsx"
            summary_df.to_excel(summary_path, index=False, engine='openpyxl')
            summary_paths.append(summary_path)

        return summary_paths


class ProfessionalThrustAnalyzer:
    """Professional Thrust Analyzer - Core Algorithm (Unified)"""
    
    def __init__(self, sampling_rate, log_func=None, auto_mode=True, 
                 adaptive_period=True, use_processed_signal=False):
        self.fs = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.log = log_func if log_func else print
        self.auto_mode = auto_mode
        self.adaptive_period = adaptive_period
        self.use_processed_signal = use_processed_signal
        self.dynamic_baseline_result = None
    
    def load_data(self, file_path):
        """Load data from file"""
        try:
            df = pd.read_excel(file_path)
        except:
            df = pd.read_csv(file_path)
        
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
    
    def auto_detect_parameters(self, y_processed, cycle_period):
        """Auto-detect optimal parameters based on signal characteristics"""
        signal_std = np.std(y_processed)
        signal_max = np.max(np.abs(y_processed))
        
        if signal_std > 0:
            signal_snr = signal_max / signal_std
        else:
            signal_snr = 0
        
        if signal_snr > 5:
            prominence_factor = 0.3
            min_distance_factor = 0.85
        elif signal_snr > 3:
            prominence_factor = 0.4
            min_distance_factor = 0.85
        elif signal_snr > 2:
            prominence_factor = 0.5
            min_distance_factor = 0.85
        else:
            prominence_factor = 0.7
            min_distance_factor = 0.9
        
        self.log(f"  Auto-params: SNR={signal_snr:.2f}, prominence={prominence_factor}, distance={min_distance_factor}")
        
        return min_distance_factor, prominence_factor
    
    def estimate_actual_period(self, t, y_processed, expected_period=None):
        """Estimate actual period using FFT
        
        If expected_period is None, searches the full frequency range.
        Otherwise, searches within ±50% of the expected frequency.
        """
        try:
            n = len(y_processed)
            # Remove mean before FFT
            yf = fft(y_processed - np.mean(y_processed))
            xf = fftfreq(n, 1/self.fs)
            
            positive_freqs = xf[:n//2]
            positive_power = np.abs(yf[:n//2])
            
            nyq = 0.5 * self.fs
            
            if expected_period is not None and expected_period > 0:
                # Search within ±50% of expected frequency
                f0 = 1.0 / expected_period
                min_freq = max(0.1, 0.5 * f0)  # At least 0.1 Hz
                max_freq = min(1.5 * f0, 0.95 * nyq)
            else:
                # No hint - search full range (0.1 Hz to Nyquist)
                # Typical robotic motion is 0.3 - 5 Hz
                min_freq = 0.1
                max_freq = min(10.0, 0.95 * nyq)
            
            freq_mask = (positive_freqs >= min_freq) & (positive_freqs <= max_freq)
            if np.any(freq_mask):
                masked_freqs = positive_freqs[freq_mask]
                masked_power = positive_power[freq_mask]
                peak_freq = masked_freqs[np.argmax(masked_power)]
                estimated_period = 1 / peak_freq if peak_freq > 0 else 1.0
                return estimated_period
            else:
                return 1.0  # Default fallback
                
        except Exception as e:
            self.log(f"  Period estimation failed: {e}")
            return 1.0  # Default fallback
    
    def process_signal(self, force_raw, cutoff_freq, t=None):
        """Separate slow sensor drift from the signed periodic force waveform."""
        if t is None:
            t = np.arange(len(force_raw), dtype=float) / self.fs
        result = analyze_dynamic_baseline(t, force_raw)
        self.dynamic_baseline_result = result
        self.fs = result.sampling_rate_hz
        self.dt = 1.0 / self.fs
        baseline_curve = np.asarray(result.baseline)
        force_debiased = np.asarray(result.corrected_force)

        self.log(f"  FFT frequency: {result.frequency_hz:.4f} Hz ({result.period_s:.4f} s)")
        self.log(f"  Pre-actuation baseline anchor: {result.baseline_anchor_n:.4f} N")
        baseline_min = np.min(baseline_curve)
        baseline_max = np.max(baseline_curve)
        baseline_drift = baseline_max - baseline_min
        self.log(f"  Baseline range: [{baseline_min:.4f}, {baseline_max:.4f}] N, drift: {baseline_drift:.4f} N")

        nyquist = 0.5 * self.fs
        # Low-pass filter for noise removal
        max_cutoff = nyquist * 0.95
        if cutoff_freq >= max_cutoff:
            cutoff_freq = max_cutoff * 0.8
            self.log(f"  Cutoff adjusted to: {cutoff_freq:.2f} Hz")
        
        force_filtered = lowpass_irregular_signal(
            t, force_debiased, self.fs, cutoff_freq
        )
        
        return force_debiased, force_filtered, baseline_curve
    
    def detect_cycles_with_fallback(self, t, y_original, y_processed):
        """Build signed, non-overlapping cycle statistics from the drift result."""
        result = self.dynamic_baseline_result
        if result is not None and len(result.corrected_force) == len(t):
            measurement = y_processed if self.use_processed_signal else y_original
            thrust_cycles = []
            for index, (start, stop) in enumerate(
                zip(result.cycle_boundaries[:-1], result.cycle_boundaries[1:])
            ):
                cycle_values = measurement[int(start) : int(stop)]
                peak_idx = int(start + np.argmax(cycle_values))
                valley_idx = int(start + np.argmin(cycle_values))
                peak_val = float(measurement[peak_idx])
                valley_val = float(measurement[valley_idx])
                thrust_cycles.append({
                    'cycle_start_idx': int(start),
                    'cycle_end_idx': int(stop),
                    'peak_idx': peak_idx,
                    'baseline_idx': valley_idx,
                    'valley_idx': valley_idx,
                    'peak_value': peak_val,
                    'baseline_value': valley_val,
                    'recovery_value': valley_val,
                    'net_thrust': peak_val - valley_val,
                    'mean_force': float(result.cycle_mean_forces[index]),
                    'impulse': float(result.cycle_impulses[index])
                })
            self.log(f"  Detected {len(thrust_cycles)} complete, non-overlapping cycles")
            return thrust_cycles, result.period_s
        
        # Auto-detect period using FFT (no user input needed)
        actual_period = self.estimate_actual_period(t, y_processed, None)
        self.log(f"  Auto-detected period: {actual_period:.3f}s")
        
        # Estimate push/rest duration as half the period each
        push_duration = actual_period * 0.5
        rest_duration = actual_period * 0.5
        
        # Auto-detect parameters
        if self.auto_mode:
            min_dist_factor, prom_factor = self.auto_detect_parameters(y_processed, actual_period)
        else:
            min_dist_factor, prom_factor = 0.85, 0.5
        
        # Peak detection
        min_distance_samples = max(1, int(actual_period * min_dist_factor * self.fs))
        signal_std = np.std(y_processed)
        prominence_threshold = prom_factor * signal_std
        
        peaks, _ = find_peaks(
            y_processed,
            prominence=prominence_threshold,
            distance=min_distance_samples
        )
        
        self.log(f"  Detected {len(peaks)} peaks")
        
        # Fallback if too few peaks
        actual_duration = t.max() - t.min()
        expected_cycles = max(1, int(actual_duration / actual_period))
        
        if len(peaks) < max(2, int(0.6 * expected_cycles)):
            self.log("  Fallback: state machine detection...")
            peaks, baselines = self._detect_cycles_state_machine(
                t, y_processed, push_duration, rest_duration, actual_period
            )
        else:
            # Find baselines before each peak
            baselines = []
            for i, peak_idx in enumerate(peaks):
                search_start = peaks[i-1] if i > 0 else 0
                search_end = peak_idx
                if search_start >= search_end:
                    baselines.append(max(0, peak_idx - 1))
                    continue
                baseline_idx = search_start + np.argmin(y_processed[search_start:search_end])
                baselines.append(baseline_idx)
        
        # Build cycle results
        thrust_cycles = []
        for i, peak_idx in enumerate(peaks):
            baseline_idx = baselines[i] if i < len(baselines) else max(0, peak_idx-1)
            
            if self.use_processed_signal:
                peak_val = y_processed[peak_idx]
                base_val = y_processed[baseline_idx]
            else:
                peak_val = y_original[peak_idx]
                base_val = y_original[baseline_idx]
            
            net_thrust = peak_val - base_val
            
            if net_thrust > 0:
                # Calculate impulse for this cycle
                if i < len(peaks) - 1:
                    cycle_end = peaks[i+1]
                else:
                    cycle_end = len(y_processed)
                cycle_start = baseline_idx
                
                if cycle_end > cycle_start:
                    impulse = np.trapz(y_processed[cycle_start:cycle_end], t[cycle_start:cycle_end])
                else:
                    impulse = 0
                
                thrust_cycles.append({
                    'peak_idx': peak_idx,
                    'baseline_idx': baseline_idx,
                    'peak_value': peak_val,
                    'baseline_value': base_val,
                    'net_thrust': net_thrust,
                    'impulse': impulse
                })
        
        return thrust_cycles, actual_period
    
    def _detect_cycles_state_machine(self, t, y_proc, push_duration, rest_duration, expected_period):
        """State machine detection (fallback)"""
        n = len(y_proc)
        if n == 0:
            return np.array([], dtype=int), np.array([], dtype=int)
        
        q20 = float(np.quantile(y_proc, 0.20))
        q80 = float(np.quantile(y_proc, 0.80))
        upper = q80
        lower = q20
        
        min_on = max(1, int(push_duration * self.fs * 0.5))
        cooldown = max(1, int(expected_period * self.fs * 0.5))
        
        on = False
        on_start = -1
        last_off_end = 0
        peaks = []
        baselines = []
        current_peak_idx = None
        last_peak_idx = -cooldown
        
        for i in range(n):
            val = y_proc[i]
            if not on:
                if i > last_off_end:
                    seg = y_proc[last_off_end:i+1]
                    baseline_idx = last_off_end + int(np.argmin(seg)) if len(seg) > 0 else last_off_end
                else:
                    baseline_idx = last_off_end
                
                if val > upper and (i - last_peak_idx) >= cooldown:
                    on = True
                    on_start = i
                    current_peak_idx = i
                    baselines.append(baseline_idx)
            else:
                if current_peak_idx is None or val > y_proc[current_peak_idx]:
                    current_peak_idx = i
                
                if val < lower and (i - on_start) >= min_on:
                    peaks.append(current_peak_idx)
                    last_peak_idx = current_peak_idx
                    on = False
                    last_off_end = i
                    current_peak_idx = None
        
        if on and current_peak_idx is not None:
            peaks.append(current_peak_idx)
        
        return np.array(peaks, dtype=int), np.array(baselines[:len(peaks)], dtype=int)
    
    def plot_results(self, t, force_raw, force_filtered, cycle_results, 
                    baseline_curve, file_name, output_dir):
        """Plot SVG chart with 6 panels"""
        fig = plt.figure(figsize=(16, 10))
        
        # Panel 1: Original + baseline
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(t, force_raw, 'b-', linewidth=0.8, alpha=0.6, label='Raw')
        ax1.plot(t, baseline_curve, 'r-', linewidth=2, label='Dynamic sensor baseline')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Raw Data & Dynamic Sensor Baseline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Debiased
        ax2 = plt.subplot(3, 2, 2)
        force_debiased = force_raw - baseline_curve
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
        ax3.set_title('Low-Pass Filter')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Noise
        ax4 = plt.subplot(3, 2, 4)
        noise = force_debiased - force_filtered
        ax4.plot(t, noise, 'orange', linewidth=0.5, alpha=0.7)
        ax4.set_ylabel('Force (N)')
        ax4.set_title('Removed Noise')
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Cycle detection
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(t, force_filtered, 'b-', linewidth=1.5)
        
        for cycle in cycle_results:
            peak_idx = cycle['peak_idx']
            base_idx = cycle['valley_idx']
            ax5.plot(t[peak_idx], force_filtered[peak_idx], 'ro', markersize=8)
            ax5.plot(t[base_idx], force_filtered[base_idx], 'mo', markersize=6)
        
        ax5.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Force (N)')
        ax5.set_title(f'Cycle Detection (n={len(cycle_results)})')
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Thrust vs Drag
        ax6 = plt.subplot(3, 2, 6)
        if cycle_results:
            cycle_nums = range(1, len(cycle_results)+1)
            thrusts = [c['peak_value'] for c in cycle_results]
            recoveries = [c['recovery_value'] for c in cycle_results]
            
            width = 0.35
            x = np.arange(len(cycle_nums))
            
            ax6.bar(x - width/2, thrusts, width, label='Thrust', color='green', alpha=0.7)
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
        plt.close()
        
        return svg_path
    
    def analyze_file(self, file_path, cutoff_freq):
        """Analyze single file with auto period detection"""
        file_path_obj = Path(file_path)
        file_name = file_path_obj.stem
        folder_name = file_path_obj.parent.name
        
        # Load data
        t, force_raw = self.load_data(file_path)
        self.log(f"  Data: {len(force_raw)} points, {t[-1]:.2f}s")
        
        # Signal processing
        force_debiased, force_filtered, baseline_curve = self.process_signal(
            force_raw, cutoff_freq, t=t
        )
        
        # Cycle detection with auto period detection
        cycle_results, detected_period = self.detect_cycles_with_fallback(
            t, force_debiased, force_filtered
        )
        
        if not cycle_results:
            self.log("  ⚠️ No cycles detected")
            return None
        
        # Statistics
        net_thrusts = [c['net_thrust'] for c in cycle_results]
        impulses = [c['impulse'] for c in cycle_results]
        peak_values = [c['peak_value'] for c in cycle_results]
        base_values = [c['baseline_value'] for c in cycle_results]
        mean_forces = [c['mean_force'] for c in cycle_results]
        
        forward_cycles = sum(1 for imp in impulses if imp > 0)
        
        self.log(f"  Detected {len(cycle_results)} cycles")
        self.log(f"  Mean net thrust: {np.mean(net_thrusts):.4f} N")
        self.log(f"  Mean cycle force: {np.mean(mean_forces):.4f} N")
        self.log(f"  Mean signed recovery force: {np.mean(base_values):.4f} N")
        self.log(f"  Forward: {forward_cycles}/{len(cycle_results)}")
        
        # Create output directory
        output_dir = file_path_obj.parent / "thrust_analysis_results"
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
            'data_duration': t[-1],
            'num_cycles': len(cycle_results),
            'forward_cycles': forward_cycles,
            'backward_cycles': len(cycle_results) - forward_cycles,
            'mean_thrust': np.mean(peak_values),
            'mean_drag': np.mean(base_values),
            'mean_recovery_force': np.mean(base_values),
            'mean_net_thrust': np.mean(net_thrusts),
            'mean_cycle_force': np.mean(mean_forces),
            'std_net_thrust': np.std(net_thrusts),
            'mean_impulse': np.mean(impulses),
            'std_impulse': np.std(impulses),
            'total_impulse': np.sum(impulses),
            'detected_period': detected_period,
            'detected_frequency': 1.0 / detected_period,
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

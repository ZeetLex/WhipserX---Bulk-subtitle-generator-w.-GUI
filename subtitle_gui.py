"""
WhisperX Subtitle Generator - GUI
----------------------------------
Batch transcribe + translate subtitles using your NVIDIA GPU.
Requires: pip install whisperx transformers sentencepiece sacremoses torch
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import os
import sys
import queue
from pathlib import Path

# ─── Color palette ───────────────────────────────────────────────────────────
BG          = "#0f0f0f"
BG_PANEL    = "#181818"
BG_INPUT    = "#1e1e1e"
BG_HOVER    = "#252525"
ACCENT      = "#1db954"
ACCENT_DIM  = "#158a3e"
TEXT        = "#e8e8e8"
TEXT_DIM    = "#888888"
TEXT_MUTED  = "#555555"
BORDER      = "#2a2a2a"
WARN        = "#e8a020"
ERROR_COL   = "#e84040"
SUCCESS_COL = "#1db954"

VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.m2ts', '.ts', '.mov', '.wmv'}

log_queue = queue.Queue()


# ─── Styled button ────────────────────────────────────────────────────────────
class FlatButton(tk.Button):
    def __init__(self, parent, text, command=None, accent=False, width=None, height=None, **kw):
        bg = ACCENT if accent else BG_INPUT
        fg = "#000000" if accent else TEXT
        abg = ACCENT_DIM if accent else BG_HOVER
        super().__init__(
            parent, text=text, command=command,
            bg=bg, fg=fg, activebackground=abg, activeforeground=fg,
            relief="flat", borderwidth=0,
            font=("Consolas", 10, "bold"),
            cursor="hand2", padx=10, pady=4,
            **kw
        )
        self._accent = accent
        self._bg = bg
        self._abg = abg
        self.bind("<Enter>", lambda e: self.configure(bg=abg))
        self.bind("<Leave>", lambda e: self.configure(bg=bg))

    def configure_text(self, text):
        self.configure(text=text)


# ─── Section label ────────────────────────────────────────────────────────────
def section_label(parent, text):
    f = tk.Frame(parent, bg=BG_PANEL)
    tk.Label(f, text=text.upper(), font=("Consolas", 9, "bold"),
             fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left")
    tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=6)
    return f


def styled_entry(parent, textvariable, width=30):
    e = tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=BG_INPUT, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Consolas", 10),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT)
    return e


def styled_combo(parent, values, textvariable, width=28):
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Dark.TCombobox",
                     fieldbackground=BG_INPUT, background=BG_INPUT,
                     foreground=TEXT, arrowcolor=TEXT_DIM,
                     bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
                     selectbackground=BG_INPUT, selectforeground=TEXT)
    style.map("Dark.TCombobox",
              fieldbackground=[("readonly", BG_INPUT)],
              foreground=[("readonly", TEXT)],
              background=[("active", BG_HOVER)])
    c = ttk.Combobox(parent, values=values, textvariable=textvariable,
                     width=width, style="Dark.TCombobox",
                     state="readonly", font=("Consolas", 10))
    return c


# ─── Main App ─────────────────────────────────────────────────────────────────
class SubtitleApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhisperX Subtitle Generator")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.geometry("780x740")
        self.minsize(680, 600)

        # Variables
        self.folder_var      = tk.StringVar(value="")
        self.queue_folders   = []
        self.model_var       = tk.StringVar(value="medium")
        self.compute_var     = tk.StringVar(value="float16")
        self.lang_var        = tk.StringVar(value="auto-detect")
        self.target_var      = tk.StringVar(value="English (en)")
        self.trans_engine_var = tk.StringVar(value="Argos (local, offline)")
        self.skip_var        = tk.BooleanVar(value=True)
        self.skip_no_var     = tk.BooleanVar(value=True)
        self.keep_orig_var   = tk.BooleanVar(value=True)
        self.dry_run_var     = tk.BooleanVar(value=False)
        self.delete_en_var   = tk.BooleanVar(value=False)
        self.delete_no_var   = tk.BooleanVar(value=False)
        self.batch_size_var  = tk.StringVar(value="4")

        self.video_files     = []
        self.is_running      = False

        self._build_ui()
        self._poll_log()

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG, pady=16)
        hdr.pack(fill="x", padx=24)
        tk.Label(hdr, text="◈ WhisperX", font=("Consolas", 18, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hdr, text=" Subtitle Generator", font=("Consolas", 18),
                 fg=TEXT, bg=BG).pack(side="left")
        self.gpu_badge = tk.Label(hdr, text="  checking GPU…  ",
                                  font=("Consolas", 9), fg=TEXT_DIM,
                                  bg=BG_INPUT, relief="flat", padx=6, pady=2)
        self.gpu_badge.pack(side="right")
        self.after(200, self._check_gpu)

        # Main scroll area
        outer = tk.Frame(self, bg=BG)
        outer.pack(fill="both", expand=True, padx=24, pady=(0, 8))

        # ── Queue card ────────────────────────────────────────────────────────
        queue_card = tk.Frame(outer, bg=BG_PANEL, padx=16, pady=14)
        queue_card.pack(fill="x", pady=(0, 8))

        section_label(queue_card, "Processing Queue").pack(fill="x", pady=(0, 8))

        # Add folder row
        add_row = tk.Frame(queue_card, bg=BG_PANEL)
        add_row.pack(fill="x", pady=(0, 8))
        self.add_entry = styled_entry(add_row, self.folder_var, width=50)
        self.add_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        FlatButton(add_row, "Browse", command=self._browse).pack(side="left")
        FlatButton(add_row, "Add to Queue", command=self._add_to_queue).pack(side="left", padx=(6, 0))

        # Queue listbox with scrollbar
        list_frame = tk.Frame(queue_card, bg=BG_PANEL)
        list_frame.pack(fill="x", pady=(0, 6))

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", bg=BG_INPUT,
                                  troughcolor=BG_INPUT, width=10)
        self.queue_listbox = tk.Listbox(
            list_frame, height=6,
            bg=BG_INPUT, fg=TEXT, selectbackground=ACCENT, selectforeground="#000",
            font=("Consolas", 9), relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=BORDER,
            activestyle="none", yscrollcommand=scrollbar.set
        )
        scrollbar.config(command=self.queue_listbox.yview)
        self.queue_listbox.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Queue controls
        ctrl_row = tk.Frame(queue_card, bg=BG_PANEL)
        ctrl_row.pack(fill="x")
        FlatButton(ctrl_row, "▲ Move Up",    command=self._queue_up).pack(side="left", padx=(0, 6))
        FlatButton(ctrl_row, "▼ Move Down",  command=self._queue_down).pack(side="left", padx=(0, 6))
        FlatButton(ctrl_row, "✕ Remove",     command=self._queue_remove).pack(side="left", padx=(0, 6))
        FlatButton(ctrl_row, "Clear All",    command=self._queue_clear).pack(side="left", padx=(0, 6))
        FlatButton(ctrl_row, "Scan Queue",   command=self._scan_queue).pack(side="right")

        # File count label
        self.file_label = tk.Label(queue_card,
                                   text="Add folders to the queue and click Scan Queue…",
                                   font=("Consolas", 9), fg=TEXT_MUTED,
                                   bg=BG_PANEL, anchor="w")
        self.file_label.pack(fill="x", pady=(8, 0))

        # ── Settings card ─────────────────────────────────────────────────────
        settings_card = tk.Frame(outer, bg=BG_PANEL, padx=16, pady=14)
        settings_card.pack(fill="x", pady=(0, 8))

        section_label(settings_card, "Model Settings").pack(fill="x", pady=(0, 10))

        grid = tk.Frame(settings_card, bg=BG_PANEL)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        def lbl(parent, text, row, col):
            tk.Label(parent, text=text, font=("Consolas", 10), fg=TEXT_DIM,
                     bg=BG_PANEL, anchor="w").grid(row=row, column=col,
                                                    sticky="w", padx=(0, 8), pady=4)

        lbl(grid, "Whisper model", 0, 0)
        styled_combo(grid, ["tiny","base","small","medium","large-v2","large-v3"],
                     self.model_var, width=20).grid(row=0, column=1, sticky="w", padx=(0, 24))

        lbl(grid, "Compute type", 0, 2)
        styled_combo(grid, ["float16 (GPU recommended)",
                             "int8_float16 (lower VRAM)",
                             "int8 (CPU fallback)"],
                     self.compute_var, width=26).grid(row=0, column=3, sticky="w")

        lbl(grid, "Source language", 1, 0)
        langs_src = ["auto-detect","Norwegian (no)","Swedish (sv)","Danish (da)",
                     "Finnish (fi)","German (de)","French (fr)","Spanish (es)",
                     "Italian (it)","Japanese (ja)","Korean (ko)","Chinese (zh)"]
        styled_combo(grid, langs_src, self.lang_var, width=20).grid(row=1, column=1,
                                                                      sticky="w", padx=(0, 24))

        lbl(grid, "Translate to", 1, 2)
        styled_combo(grid, [
            "English (en)",
            "Norwegian (no)",
            "Swedish (sv)",
            "Danish (da)",
            "German (de)",
            "French (fr)",
            "Spanish (es)",
            "None (keep original)",
        ], self.target_var, width=26).grid(row=1, column=3, sticky="w")

        lbl(grid, "Batch size", 2, 0)
        styled_entry(grid, self.batch_size_var, width=6).grid(row=2, column=1,
                                                               sticky="w", padx=(0, 24))

        lbl(grid, "Translation engine", 2, 2)
        styled_combo(grid, [
            "Argos (local, offline)",
            "Google Translate (online)",
        ], self.trans_engine_var, width=26).grid(row=2, column=3, sticky="w")

        # ── Options card ──────────────────────────────────────────────────────
        opts_card = tk.Frame(outer, bg=BG_PANEL, padx=16, pady=14)
        opts_card.pack(fill="x", pady=(0, 8))

        section_label(opts_card, "Options").pack(fill="x", pady=(0, 8))

        def chk(parent, text, var):
            f = tk.Frame(parent, bg=BG_PANEL)
            cb = tk.Checkbutton(f, variable=var, bg=BG_PANEL,
                                activebackground=BG_PANEL,
                                selectcolor=BG_INPUT,
                                fg=ACCENT, activeforeground=ACCENT)
            cb.pack(side="left")
            tk.Label(f, text=text, font=("Consolas", 10), fg=TEXT,
                     bg=BG_PANEL).pack(side="left")
            return f

        chk(opts_card, "Skip files that already have a .en.srt", self.skip_var).pack(anchor="w", pady=2)
        chk(opts_card, "Skip files that already have a .no.srt", self.skip_no_var).pack(anchor="w", pady=2)
        chk(opts_card, "Also save original-language .srt", self.keep_orig_var).pack(anchor="w", pady=2)
        chk(opts_card, "Dry run (list files only, no processing)", self.dry_run_var).pack(anchor="w", pady=2)

        tk.Frame(opts_card, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))
        tk.Label(opts_card, text="DELETE EXISTING SUBTITLES BEFORE PROCESSING",
                 font=("Consolas", 8, "bold"), fg=TEXT_MUTED, bg=BG_PANEL).pack(anchor="w")
        chk(opts_card, "Delete existing .en.srt files (replaces Bazarr subs)", self.delete_en_var).pack(anchor="w", pady=2)
        chk(opts_card, "Delete existing .no.srt files", self.delete_no_var).pack(anchor="w", pady=2)

        # ── Log card ──────────────────────────────────────────────────────────
        log_card = tk.Frame(outer, bg=BG_PANEL, padx=16, pady=14)
        log_card.pack(fill="both", expand=True, pady=(0, 8))

        section_label(log_card, "Output Log").pack(fill="x", pady=(0, 8))

        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar",
                         troughcolor=BG_INPUT, background=ACCENT,
                         bordercolor=BG_INPUT, lightcolor=ACCENT, darkcolor=ACCENT)
        self.pbar = ttk.Progressbar(log_card, variable=self.progress_var,
                                    maximum=100, style="Green.Horizontal.TProgressbar")
        self.pbar.pack(fill="x", pady=(0, 8))

        self.log_box = scrolledtext.ScrolledText(
            log_card, height=10, bg=BG_INPUT, fg=TEXT,
            insertbackground=TEXT, font=("Consolas", 9),
            relief="flat", borderwidth=0,
            highlightthickness=1, highlightbackground=BORDER)
        self.log_box.pack(fill="both", expand=True)
        self.log_box.tag_config("ok",   foreground=SUCCESS_COL)
        self.log_box.tag_config("info", foreground="#5b9bd5")
        self.log_box.tag_config("warn", foreground=WARN)
        self.log_box.tag_config("err",  foreground=ERROR_COL)
        self.log_box.tag_config("dim",  foreground=TEXT_MUTED)
        self.log_box.configure(state="disabled")

        # ── Bottom bar ────────────────────────────────────────────────────────
        bar = tk.Frame(self, bg=BG, pady=12)
        bar.pack(fill="x", padx=24, side="bottom")

        self.status_label = tk.Label(bar, text="Ready.", font=("Consolas", 9),
                                     fg=TEXT_MUTED, bg=BG, anchor="w")
        self.status_label.pack(side="left", fill="x", expand=True)

        self.run_btn = FlatButton(bar, "▶  Start Processing",
                                  command=self._start, accent=True)
        self.run_btn.pack(side="right")

        FlatButton(bar, "Clear Log", command=self._clear_log).pack(side="right", padx=(0, 8))

    # ── GPU detection ─────────────────────────────────────────────────────────
    def _check_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
                self.gpu_badge.configure(text=f"  ✓ {name} ({vram}GB)  ",
                                         fg="#000", bg=ACCENT)
            else:
                self.gpu_badge.configure(text="  ⚠ No CUDA GPU found  ",
                                         fg="#000", bg=WARN)
        except ImportError:
            self.gpu_badge.configure(text="  torch not installed  ",
                                     fg=TEXT_DIM, bg=BG_INPUT)

    # ── Queue management ──────────────────────────────────────────────────────
    def _browse(self):
        folder = filedialog.askdirectory(title="Select folder to add to queue")
        if folder:
            self.folder_var.set(folder)

    def _add_to_queue(self):
        folder = self.folder_var.get().strip()
        if not folder:
            return
        if not os.path.isdir(folder):
            self.file_label.configure(text="⚠  Folder not found.", fg=WARN)
            return
        if folder in self.queue_folders:
            self.file_label.configure(text="⚠  Folder already in queue.", fg=WARN)
            return
        self.queue_folders.append(folder)
        self.queue_listbox.insert("end", folder)
        self.folder_var.set("")
        self.file_label.configure(
            text=f"✓  {len(self.queue_folders)} folder(s) in queue. Click Scan Queue to count files.",
            fg=ACCENT)

    def _queue_up(self):
        sel = self.queue_listbox.curselection()
        if not sel or sel[0] == 0:
            return
        i = sel[0]
        self.queue_folders[i], self.queue_folders[i-1] = self.queue_folders[i-1], self.queue_folders[i]
        self._refresh_queue_listbox(i-1)

    def _queue_down(self):
        sel = self.queue_listbox.curselection()
        if not sel or sel[0] >= len(self.queue_folders) - 1:
            return
        i = sel[0]
        self.queue_folders[i], self.queue_folders[i+1] = self.queue_folders[i+1], self.queue_folders[i]
        self._refresh_queue_listbox(i+1)

    def _queue_remove(self):
        sel = self.queue_listbox.curselection()
        if not sel:
            return
        i = sel[0]
        self.queue_folders.pop(i)
        self._refresh_queue_listbox(min(i, len(self.queue_folders)-1))
        self.file_label.configure(
            text=f"{len(self.queue_folders)} folder(s) in queue.", fg=TEXT_DIM)

    def _queue_clear(self):
        self.queue_folders.clear()
        self.queue_listbox.delete(0, "end")
        self.video_files = []
        self.file_label.configure(text="Queue cleared.", fg=TEXT_MUTED)

    def _refresh_queue_listbox(self, select_idx=None):
        self.queue_listbox.delete(0, "end")
        for f in self.queue_folders:
            self.queue_listbox.insert("end", f)
        if select_idx is not None and 0 <= select_idx < len(self.queue_folders):
            self.queue_listbox.selection_set(select_idx)
            self.queue_listbox.see(select_idx)

    def _scan_queue(self):
        if not self.queue_folders:
            self.file_label.configure(text="⚠  No folders in queue.", fg=WARN)
            return
        skip_en = self.skip_var.get()
        skip_no = self.skip_no_var.get()
        tgt_raw = self.target_var.get()
        tgt_lang = None if "None" in tgt_raw else tgt_raw.split("(")[-1].rstrip(")")

        self.video_files = []
        for folder in self.queue_folders:
            folder_files = []
            for root, dirs, files in os.walk(folder):
                # Sort dirs so Season 1 comes before Season 2 etc.
                dirs.sort(key=lambda d: (
                    # extract leading number for natural sort
                    int(''.join(filter(str.isdigit, d)) or 0),
                    d.lower()
                ))
                for f in sorted(files):
                    p = Path(root) / f
                    if p.suffix.lower() in VIDEO_EXTENSIONS:
                        skip_suf = f".{tgt_lang}.srt" if tgt_lang else ".en.srt"
                        if skip_en and p.with_suffix('.en.srt').exists():
                            continue
                        if skip_no and p.with_suffix('.no.srt').exists():
                            continue
                        folder_files.append(p)
            self.video_files.extend(folder_files)

        count = len(self.video_files)
        if count == 0:
            self.file_label.configure(
                text="No video files found (or all already have subtitles).",
                fg=TEXT_MUTED)
        else:
            self.file_label.configure(
                text=f"✓  {count} file(s) queued across {len(self.queue_folders)} folder(s)",
                fg=ACCENT)
        self.status_label.configure(text=f"{count} file(s) ready.")

    # keep _scan_folder as alias used by _start
    def _scan_folder(self):
        self._scan_queue()

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _log(self, msg, tag=""):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.progress_var.set(0)

    def _poll_log(self):
        try:
            while True:
                msg, tag = log_queue.get_nowait()
                self._log(msg, tag)
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    # ── Start processing ──────────────────────────────────────────────────────
    def _start(self):
        if self.is_running:
            return
        if not self.queue_folders:
            self._log("⚠  Please add at least one folder to the queue first.", "warn")
            return
        self._scan_queue()
        if not self.video_files:
            self._log("⚠  No files to process.", "warn")
            return
        self.is_running = True
        self.run_btn.configure_text("⏳  Running…")
        self.progress_var.set(0)
        t = threading.Thread(target=self._run_batch, daemon=True)
        t.start()

    def _run_batch(self):
        def q(msg, tag=""):
            log_queue.put((msg, tag))

        try:
            import torch
            import whisperx
        except ImportError as e:
            q(f"✗  Missing dependency: {e}", "err")
            q("   Run:  pip install whisperx transformers sentencepiece sacremoses torch", "warn")
            self._finish()
            return

        model_name   = self.model_var.get()
        compute_raw  = self.compute_var.get().split()[0]
        lang_raw     = self.lang_var.get()
        target_raw   = self.target_var.get()
        trans_engine = self.trans_engine_var.get()
        keep_orig    = self.keep_orig_var.get()
        dry_run      = self.dry_run_var.get()
        delete_en    = self.delete_en_var.get()
        delete_no    = self.delete_no_var.get()
        batch_size   = int(self.batch_size_var.get() or 16)

        src_lang = None if lang_raw == "auto-detect" else lang_raw.split("(")[-1].rstrip(")")
        tgt_lang = None if "None" in target_raw else target_raw.split("(")[-1].rstrip(")")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        q(f"─── Starting batch ───────────────────────────", "dim")
        q(f"  Model    : {model_name}", "info")
        q(f"  Device   : {device} ({compute_raw})", "info")
        q(f"  Engine   : {trans_engine}", "info")
        q(f"  Files    : {len(self.video_files)}", "info")
        if dry_run:
            q("  Mode     : DRY RUN", "warn")
        q("", "dim")

        if dry_run:
            for p in self.video_files:
                q(f"  [would process] {p}", "dim")
            q(f"\nDry run complete. {len(self.video_files)} file(s) listed.", "ok")
            self._finish()
            return

        # Load Whisper model
        q(f"Loading model '{model_name}' …", "info")
        try:
            task = "translate" if tgt_lang == "en" and src_lang and src_lang != "en" else "transcribe"
            whisper_model = whisperx.load_model(
                model_name, device, compute_type=compute_raw,
                language=src_lang
            )
            q(f"✓  Model loaded.", "ok")
        except Exception as e:
            q(f"✗  Failed to load model: {e}", "err")
            self._finish()
            return


        # ── Translation engines ───────────────────────────────────────────────
        LANG_MAP = {
            "no": "nb", "nb": "nb", "sv": "sv", "da": "da",
            "de": "de", "fr": "fr", "es": "es", "fi": "fi",
            "nl": "nl", "it": "it", "pl": "pl", "en": "en",
            "ja": "ja", "ko": "ko", "zh": "zh",
        }
        GOOGLE_LANG = {**LANG_MAP, "nb": "no"}

        # Cache loaded translation models
        ct2_model_cache = {}

        # Enable Argos GPU usage, disable Stanza SBD to avoid interference
        import os as _os
        _os.environ["ARGOS_DEVICE_TYPE"] = "cuda" if device == "cuda" else "cpu"
        _os.environ["ARGOS_STANZA_AVAILABLE"] = "False"

        def translate_ct2(texts, src, tgt):
            """Translate using Argos Translate with GPU support."""
            try:
                import argostranslate.package as ap
                import argostranslate.translate as at

                al_src = LANG_MAP.get(src, src)
                al_tgt = LANG_MAP.get(tgt, tgt)
                key = (al_src, al_tgt)

                if key not in ct2_model_cache:
                    tgt_codes = ["nb", "no"] if al_tgt in ("nb", "no") else [al_tgt]

                    installed = at.get_installed_languages()
                    src_lang  = next((l for l in installed if l.code == al_src), None)
                    tgt_lang  = None
                    for tc in tgt_codes:
                        tgt_lang = next((l for l in installed if l.code == tc), None)
                        if tgt_lang:
                            al_tgt = tc
                            break

                    if not src_lang or not tgt_lang:
                        q(f"  Downloading Argos pack {al_src}→{tgt_codes[0]}…", "info")
                        try:
                            ap.update_package_index()
                            available = ap.get_available_packages()
                            pkg = next((p for p in available
                                       if p.from_code == al_src and p.to_code in tgt_codes), None)
                            if not pkg:
                                q(f"  ⚠  No Argos pack for {al_src}→{tgt_codes}, falling back to Google", "warn")
                                return translate_google(texts, src, tgt)
                            ap.install_from_path(pkg.download())
                            q(f"  ✓  Pack installed", "ok")
                            installed = at.get_installed_languages()
                            src_lang  = next((l for l in installed if l.code == al_src), None)
                            for tc in tgt_codes:
                                tgt_lang = next((l for l in installed if l.code == tc), None)
                                if tgt_lang:
                                    al_tgt = tc
                                    break
                        except Exception as e:
                            q(f"  ⚠  Pack install failed: {e}, falling back to Google", "warn")
                            return translate_google(texts, src, tgt)

                    if not src_lang or not tgt_lang:
                        q(f"  ⚠  Languages not available, falling back to Google", "warn")
                        return translate_google(texts, src, tgt)

                    translation = src_lang.get_translation(tgt_lang)
                    if not translation:
                        q(f"  ⚠  No translation object for {al_src}→{al_tgt}, falling back to Google", "warn")
                        return translate_google(texts, src, tgt)

                    ct2_model_cache[key] = translation
                    q(f"  ✓  Argos ready ({al_src}→{al_tgt}, GPU: {device == 'cuda'})", "ok")

                translation = ct2_model_cache[key]
                results = list(texts)
                for i, text in enumerate(texts):
                    if text.strip():
                        results[i] = translation.translate(text)
                return results

            except ImportError:
                q("  ⚠  argostranslate not installed. Run: pip install argostranslate", "warn")
                q("  ⚠  Falling back to Google Translate", "warn")
                return translate_google(texts, src, tgt)
            except Exception as e:
                q(f"  ⚠  Argos error: {e}, falling back to Google", "warn")
                return translate_google(texts, src, tgt)

        def translate_google(texts, src, tgt):
            import time
            try:
                from deep_translator import GoogleTranslator
                gl_src = GOOGLE_LANG.get(src, src)
                gl_tgt = GOOGLE_LANG.get(tgt, tgt)
                SEPARATOR = " ||| "
                MAX_CHARS = 4500
                results = [""] * len(texts)
                batch_indices = []
                batch_chars = 0
                batch_lines = []

                def flush_batch():
                    if not batch_lines:
                        return
                    joined = SEPARATOR.join(batch_lines)
                    for attempt in range(3):
                        try:
                            translator = GoogleTranslator(source=gl_src, target=gl_tgt)
                            translated_joined = translator.translate(joined)
                            if not translated_joined:
                                raise ValueError("Empty response")
                            parts = translated_joined.split(SEPARATOR)
                            if len(parts) == len(batch_indices):
                                for idx, part in zip(batch_indices, parts):
                                    results[idx] = part.strip()
                            else:
                                for i, idx in enumerate(batch_indices):
                                    results[idx] = parts[i].strip() if i < len(parts) else texts[idx]
                            time.sleep(0.3)
                            break
                        except Exception:
                            if attempt < 2:
                                wait = (attempt + 1) * 3
                                q(f"  ⚠  Google attempt {attempt+1} failed, retrying in {wait}s…", "warn")
                                time.sleep(wait)
                            else:
                                q(f"  ⚠  Google translation failed, keeping original", "warn")
                                for idx in batch_indices:
                                    results[idx] = texts[idx]
                    batch_indices.clear()
                    batch_lines.clear()

                for i, text in enumerate(texts):
                    if not text.strip():
                        results[i] = text
                        continue
                    line_len = len(text) + len(SEPARATOR)
                    if batch_chars + line_len > MAX_CHARS and batch_lines:
                        flush_batch()
                        batch_chars = 0
                    batch_indices.append(i)
                    batch_lines.append(text)
                    batch_chars += line_len
                flush_batch()

                translated_count = sum(1 for i, r in enumerate(results)
                                       if r.strip() and r.strip() != texts[i].strip())
                if translated_count == 0 and any(t.strip() for t in texts):
                    q(f"  ⚠  Translation appears to have failed (all text unchanged)", "warn")
                return results

            except ImportError:
                q("  ⚠  deep-translator not installed. Run: pip install deep-translator", "warn")
                return texts

        def translate_texts(texts, src, tgt):
            if "Argos" in trans_engine:
                return translate_ct2(texts, src, tgt)
            else:
                return translate_google(texts, src, tgt)

        def fix_caps(text):
            # Fix ALL CAPS output from Whisper
            if text == text.upper() and len(text) > 3:
                return text.capitalize()
            return text

        total = len(self.video_files)
        done  = 0

        for idx, video in enumerate(self.video_files):
            rel = video.name
            q(f"\n[{idx+1}/{total}]  {rel}", "info")
            self.status_label.configure(text=f"Processing {idx+1}/{total}: {rel}")

            # Delete existing subtitle files if requested
            if delete_en or delete_no:
                import glob
                base = str(video.with_suffix(""))
                for existing in glob.glob(base + "*.srt"):
                    ep = Path(existing)
                    name = ep.name.lower()
                    if delete_en and (".en." in name or name.endswith(".en.srt")):
                        ep.unlink()
                        q(f"  deleted: {ep.name}", "dim")
                    elif delete_no and (".no." in name or ".nb." in name or
                                        name.endswith(".no.srt") or name.endswith(".nb.srt")):
                        ep.unlink()
                        q(f"  deleted: {ep.name}", "dim")

            try:
                import whisperx as wx
                import torch as _t
                import gc

                # Clear GPU memory before each file
                _t.cuda.empty_cache()
                gc.collect()

                audio  = wx.load_audio(str(video))
                result = whisper_model.transcribe(audio, batch_size=batch_size)
                segs   = result["segments"]
                detected = result.get("language", src_lang or "?")
                q(f"  language detected: {detected}", "dim")

                # Run alignment on GPU to get word-level timestamps
                try:
                    align_model, align_meta = wx.load_align_model(
                        language_code=detected, device=device)
                    aligned = wx.align(segs, align_model, align_meta, audio, device,
                                       return_char_alignments=False)
                    segs = aligned["segments"]
                    del align_model
                    _t.cuda.empty_cache()
                    gc.collect()
                    q(f"  aligned", "dim")
                except Exception as e:
                    q(f"  ⚠  alignment skipped, using proportional split", "warn")
                    try:
                        _t.cuda.empty_cache()
                        gc.collect()
                    except Exception:
                        pass

                # Fix ALL CAPS segments from Whisper
                for s in segs:
                    s["text"] = fix_caps(s["text"].strip())

                if keep_orig:
                    orig_path = video.with_suffix(f".{detected}.srt")
                    orig_path.write_text(_to_srt(segs), encoding="utf-8-sig")
                    q(f"  saved: {orig_path.name}", "dim")

                # Translate if target language differs from detected language
                if tgt_lang and tgt_lang != detected:
                    q(f"  translating {detected}→{tgt_lang} ({trans_engine.split('(')[0].strip()})…", "dim")
                    texts = [s["text"].strip() for s in segs]
                    translated = translate_texts(texts, detected, tgt_lang)
                    for s, t in zip(segs, translated):
                        s["text"] = t
                        # Clear word-level data so _to_srt uses translated text
                        # (word entries still contain original English)
                        s.pop("words", None)
                    q(f"  translated {detected}→{tgt_lang}", "dim")

                # Plex naming: ShowName.S01E01.LANG.srt next to video file
                out_lang = tgt_lang if tgt_lang else detected
                out_path = video.with_suffix(f".{out_lang}.srt")
                out_path.write_text(_to_srt(segs), encoding="utf-8-sig")
                q(f"  ✓  saved: {out_path.name}", "ok")

            except Exception as e:
                q(f"  ✗  error: {e}", "err")
                try:
                    import torch as _t, gc
                    _t.cuda.empty_cache()
                    gc.collect()
                    # If CUDA error, reload the whisper model to reset GPU context
                    if "CUDA error" in str(e):
                        q(f"  ↺  CUDA error detected — reloading model…", "warn")
                        try:
                            whisper_model = whisperx.load_model(
                                model_name, device, compute_type=compute_raw,
                                language=src_lang
                            )
                            q(f"  ✓  Model reloaded.", "ok")
                        except Exception as reload_err:
                            q(f"  ✗  Could not reload model: {reload_err}", "err")
                    else:
                        q(f"  ↺  GPU memory cleared, continuing…", "warn")
                except Exception:
                    pass

            done += 1
            pct = (done / total) * 100
            self.progress_var.set(pct)

        q(f"\n─── Done! {done}/{total} file(s) processed. ───", "ok")
        self._finish()

    def _finish(self):
        self.is_running = False
        self.after(0, lambda: self.run_btn.configure_text("▶  Start Processing"))
        self.after(0, lambda: self.status_label.configure(text="Finished."))


# ─── SRT helper ───────────────────────────────────────────────────────────────
def _to_srt(segments):
    def ts(t):
        t = max(0.0, float(t))
        h  = int(t // 3600)
        m  = int((t % 3600) // 60)
        s  = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def split_segment(seg, max_chars=80):
        """Split a long segment using word timings if available, else proportional."""
        text = seg.get("text", "").strip()
        start = float(seg.get("start", 0))
        end   = float(seg.get("end", 0))
        duration = end - start
        words = seg.get("words", [])

        if len(text) <= max_chars and not words:
            return [{"start": start, "end": end, "text": text}]

        # Use word-level timings if available (from alignment)
        if words:
            result = []
            chunk_words = []
            chunk_start = None

            for w in words:
                w_start = w.get("start")
                w_end   = w.get("end")
                w_word  = w.get("word", "").strip()
                if not w_word:
                    continue
                if chunk_start is None:
                    chunk_start = w_start if w_start is not None else start

                chunk_words.append(w)
                chunk_text = " ".join(x.get("word","").strip() for x in chunk_words)

                if len(chunk_text) >= max_chars and len(chunk_words) > 1:
                    # Save chunk up to previous word
                    prev = chunk_words[:-1]
                    prev_text = " ".join(x.get("word","").strip() for x in prev)
                    prev_end  = prev[-1].get("end") or end
                    result.append({"start": chunk_start, "end": prev_end, "text": prev_text})
                    chunk_words = [w]
                    chunk_start = w_start if w_start is not None else prev_end

            if chunk_words:
                chunk_text = " ".join(x.get("word","").strip() for x in chunk_words)
                chunk_end  = chunk_words[-1].get("end") or end
                result.append({"start": chunk_start, "end": chunk_end, "text": chunk_text})

            return result if result else [{"start": start, "end": end, "text": text}]

        # Split on sentence boundaries first, then on commas if still too long
        import re
        # Split on .!? followed by space, or on comma+space
        parts = re.split(r'(?<=[.!?])\s+', text)
        if len(parts) == 1:
            parts = re.split(r'(?<=,)\s+', text)
        if len(parts) == 1:
            # Last resort: split roughly in half at a space
            mid = len(text) // 2
            split_at = text.rfind(' ', 0, mid + 20)
            if split_at == -1:
                split_at = mid
            parts = [text[:split_at].strip(), text[split_at:].strip()]

        # Merge very short parts with the next one
        merged = []
        buf = ""
        for p in parts:
            if not p.strip():
                continue
            if buf and len(buf) + len(p) + 1 <= max_chars:
                buf = buf + " " + p
            else:
                if buf:
                    merged.append(buf)
                buf = p
        if buf:
            merged.append(buf)

        if not merged:
            return [{"start": start, "end": end, "text": text}]

        # Assign timing proportionally by character count
        total_chars = sum(len(p) for p in merged)
        result = []
        t = start
        for p in merged:
            ratio = len(p) / total_chars if total_chars > 0 else 1 / len(merged)
            seg_duration = duration * ratio
            result.append({
                "start": round(t, 3),
                "end":   round(min(t + seg_duration, end), 3),
                "text":  p.strip()
            })
            t += seg_duration

        return result

    lines = []
    counter = 1
    for seg in segments:
        for chunk in split_segment(seg):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            lines.append(str(counter))
            lines.append(f"{ts(chunk['start'])} --> {ts(chunk['end'])}")
            lines.append(text)
            lines.append("")
            counter += 1
    return "\n".join(lines)


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SubtitleApp()
    app.mainloop()
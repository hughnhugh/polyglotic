"""The app's persistent Settings window - target languages, speech model,
overlay position, mic capture, and the Discord bridge port, all editable
live while the pipeline keeps running (see overlay_core.PipelineController).
Window-level behavior (taskbar presence, minimize-to-tray, the X button)
is wired up by overlay_core.run_translator; this module only builds the
widgets inside it."""

import queue
import tkinter as tk
from tkinter import ttk

import torch

import settings
from lang_codes import LANGUAGE_NAMES
from overlay_core import STATUS_FIELDS, live_options, speech_context, speech_options
from overlay_core import status as system_status

# torch.cuda.memory_*() only tracks memory PyTorch's own allocator claimed -
# Whisper runs through faster-whisper/ctranslate2, a separate native
# inference engine with its own CUDA memory management invisible to torch,
# so a torch-only reading would never move when the Whisper model changes.
# NVML queries the driver directly, seeing every process/library's usage.
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    _NVML_HANDLE = None

# Whisper sizes, plus Qwen3-ASR (see qwen_asr_client.py) - an alternative
# engine behind the same dropdown, stronger on Chinese/Cantonese.
SPEECH_MODELS = [
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "distil-large-v3", "large-v1", "large-v2", "large-v3",
    "qwen3-asr-0.6b", "qwen3-asr-1.7b",
]

LANG_COLUMNS = 4
BG = "#1c1c1c"
PANEL_BG = "#111111"
FG = "white"


class SettingsWindow:
    def __init__(self, root, controller):
        self.root = root
        self.controller = controller
        self.status_queue: "queue.Queue" = queue.Queue()  # free-text broadcast() messages
        self._status_fields_queue: "queue.Queue" = queue.Queue()  # structured per-subsystem status
        system_status.subscribe(self._status_fields_queue)
        self._lang_vars: "dict[str, tk.BooleanVar]" = {}
        self._gpu_available = torch.cuda.is_available()
        self._gpu_total_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if self._gpu_available else 0.0
        )

        root.configure(bg=BG)
        root.geometry("520x830")
        root.minsize(420, 480)

        def section(text):
            lbl = tk.Label(root, text=text, fg="#7ec4ff", bg=BG, font=("Segoe UI", 10, "bold"), anchor="w")
            lbl.pack(fill="x", padx=12, pady=(12, 4))

        # --- Target languages ---------------------------------------------
        section("Target languages")

        lang_outer = tk.Frame(root, bg=BG)
        lang_outer.pack(fill="both", expand=True, padx=12)

        canvas = tk.Canvas(lang_outer, bg=PANEL_BG, highlightthickness=0, height=220)
        scrollbar = tk.Scrollbar(lang_outer, orient="vertical", command=canvas.yview)
        lang_grid = tk.Frame(canvas, bg=PANEL_BG)

        lang_grid.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=lang_grid, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        by_name = sorted(LANGUAGE_NAMES, key=lambda code: LANGUAGE_NAMES[code])
        for i, code in enumerate(by_name):
            var = tk.BooleanVar(value=code in controller.targets)
            self._lang_vars[code] = var
            cb = tk.Checkbutton(
                lang_grid, text=LANGUAGE_NAMES.get(code, code), variable=var, fg=FG, bg=PANEL_BG,
                selectcolor="#333333", activebackground=PANEL_BG, activeforeground=FG,
                font=("Segoe UI", 9), anchor="w", command=lambda code=code: self._on_lang_toggle(code),
            )
            cb.grid(row=i // LANG_COLUMNS, column=i % LANG_COLUMNS, sticky="w", padx=4, pady=1)

        # --- Speech model ----------------------------------------------------
        section("Speech model")
        self.model_var = tk.StringVar(value=controller.model_holder.size)
        self.model_combo = ttk.Combobox(
            root, textvariable=self.model_var, values=SPEECH_MODELS, state="disabled"
        )
        self.model_combo.pack(fill="x", padx=12)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

        lang_frame = tk.Frame(root, bg=BG)
        lang_frame.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(lang_frame, text="Conversation language:", fg=FG, bg=BG, font=("Segoe UI", 9)).pack(side="left")
        self._language_by_label = {label: code for code, label in settings.CONVERSATION_LANGUAGES.items()}
        self.language_var = tk.StringVar(value=settings.CONVERSATION_LANGUAGES.get(speech_options["language"],
                                                                                   "Auto-detect"))
        language_combo = ttk.Combobox(lang_frame, textvariable=self.language_var, state="readonly", width=18,
                                      values=list(settings.CONVERSATION_LANGUAGES.values()))
        language_combo.pack(side="left", padx=6)
        language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)

        context_frame = tk.Frame(root, bg=BG)
        context_frame.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(context_frame, text="Names & slang:", fg=FG, bg=BG, font=("Segoe UI", 9)).pack(side="left")
        self.context_var = tk.StringVar(value=speech_options["context"])
        tk.Entry(context_frame, textvariable=self.context_var, bg="#333333", fg=FG, insertbackground=FG).pack(
            side="left", fill="x", expand=True, padx=6
        )
        tk.Button(
            context_frame, text="Apply", command=self._on_apply_context, bg="#333333", fg=FG,
            activebackground="#444444", activeforeground=FG, relief="flat", padx=8,
        ).pack(side="left")

        self.auto_names_var = tk.BooleanVar(value=speech_options["auto_names"])
        tk.Checkbutton(
            root, text="Also add the Discord channel's names", variable=self.auto_names_var, fg=FG, bg=BG,
            selectcolor="#333333", activebackground=BG, activeforeground=FG, font=("Segoe UI", 9),
            command=self._on_auto_names_toggle,
        ).pack(anchor="w", padx=12)
        live_frame = tk.Frame(root, bg=BG)
        live_frame.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(live_frame, text="Live caption updates every", fg=FG, bg=BG, font=("Segoe UI", 9)).pack(side="left")
        self.live_interval_var = tk.StringVar(value=self._format_interval(live_options["interval"]))
        # Editable: pick a suggestion or type any number of seconds in range.
        live_combo = ttk.Combobox(live_frame, textvariable=self.live_interval_var, width=6,
                                  values=["Off"] + [self._format_interval(v) for v in settings.LIVE_INTERVAL_CHOICES])
        live_combo.pack(side="left", padx=6)
        for event in ("<<ComboboxSelected>>", "<Return>", "<FocusOut>"):
            live_combo.bind(event, self._on_live_interval_changed)
        tk.Label(live_frame, text="seconds", fg=FG, bg=BG, font=("Segoe UI", 9)).pack(side="left")
        self.translate_live_var = tk.BooleanVar(value=live_options["translate"])
        tk.Checkbutton(
            live_frame, text="translate them too", variable=self.translate_live_var, fg=FG, bg=BG,
            selectcolor="#333333", activebackground=BG, activeforeground=FG, font=("Segoe UI", 9),
            command=self._on_translate_live_toggle,
        ).pack(side="left", padx=(10, 0))

        # What the speech model is actually being given, names included -
        # refreshed by _poll_status as people join, leave or speak.
        self.context_shown = tk.Label(root, text="", fg="#808080", bg=BG, font=("Segoe UI", 8), anchor="w",
                                      justify="left", wraplength=490)
        self.context_shown.pack(fill="x", padx=12)
        self._context_refresh_ticks = 0

        # --- Overlay position -------------------------------------------------
        section("Overlay position")
        self.position_var = tk.StringVar(value=controller.initial_position or "auto")
        pos_frame = tk.Frame(root, bg=BG)
        pos_frame.pack(fill="x", padx=12)
        for label, value in (("Auto (en bottom, others top)", "auto"), ("Top", "top"), ("Bottom", "bottom")):
            tk.Radiobutton(
                pos_frame, text=label, variable=self.position_var, value=value,
                fg=FG, bg=BG, selectcolor="#333333", activebackground=BG, activeforeground=FG,
                font=("Segoe UI", 9), command=self._on_position_changed,
            ).pack(anchor="w")

        # --- Mic + Discord bridge port ----------------------------------------
        section("Audio")
        self.mic_var = tk.BooleanVar(value=controller.mic_capture.enabled)
        tk.Checkbutton(
            root, text="Capture microphone", variable=self.mic_var, fg=FG, bg=BG,
            selectcolor="#333333", activebackground=BG, activeforeground=FG, font=("Segoe UI", 9),
            command=self._on_mic_toggle,
        ).pack(anchor="w", padx=12)

        port_frame = tk.Frame(root, bg=BG)
        port_frame.pack(fill="x", padx=12, pady=(4, 0))
        tk.Label(port_frame, text="Discord bridge port:", fg=FG, bg=BG, font=("Segoe UI", 9)).pack(side="left")
        self.port_var = tk.StringVar(value=str(controller.initial_bridge_port))
        tk.Entry(port_frame, textvariable=self.port_var, width=8, bg="#333333", fg=FG, insertbackground=FG).pack(
            side="left", padx=6
        )
        tk.Button(
            port_frame, text="Apply", command=self._on_apply_port, bg="#333333", fg=FG,
            activebackground="#444444", activeforeground=FG, relief="flat", padx=8,
        ).pack(side="left")

        # --- Status -----------------------------------------------------------
        section("Status")
        status_panel = tk.Frame(root, bg=BG)
        status_panel.pack(fill="x", padx=12)
        self._status_value_labels: "dict[str, tk.Label]" = {}
        for key, label in STATUS_FIELDS:
            row = tk.Frame(status_panel, bg=BG)
            row.pack(fill="x")
            tk.Label(row, text=label + ":", fg="#808080", bg=BG, font=("Segoe UI", 9), width=17, anchor="w").pack(
                side="left"
            )
            value_lbl = tk.Label(row, text="...", fg=FG, bg=BG, font=("Segoe UI", 9), anchor="w")
            value_lbl.pack(side="left", fill="x", expand=True)
            self._status_value_labels[key] = value_lbl

        self.status_label = tk.Label(
            root, text="", fg="#808080", bg=BG, font=("Segoe UI", 9, "italic"),
            anchor="w", justify="left", wraplength=490,
        )
        self.status_label.pack(fill="x", padx=12, pady=(6, 12))

        self.root.after(150, self._poll_status)

    def _on_lang_toggle(self, code):
        if self._lang_vars[code].get():
            title = f"Live Captions ({LANGUAGE_NAMES.get(code, code.upper())})"
            self.controller.add_window(code, title)
        else:
            self.controller.remove_window(code)

    def mark_target_unchecked(self, code):
        """Called by run_translator when an overlay window closes via its
        own ✕ button, so the checkbox here stays in sync. Setting the
        BooleanVar directly doesn't re-invoke the checkbox's command, so
        this can't loop back into remove_window."""
        var = self._lang_vars.get(code)
        if var is not None:
            var.set(False)

    def _on_model_selected(self, event=None):
        new_size = self.model_var.get()
        self.model_combo.config(state="disabled")
        self.controller.model_holder.request_reload(new_size)

    def _on_position_changed(self):
        pos = self.position_var.get()
        self.controller.set_position(None if pos == "auto" else pos)

    def _on_language_selected(self, event=None):
        code = self._language_by_label[self.language_var.get()]
        speech_options["language"] = code
        settings.update(language=code)

    def _on_apply_context(self):
        # e.g. "Caleb, marina, HSK, bún bò" - the speech model leans toward
        # these spellings. Takes effect from the next segment.
        context = self.context_var.get().strip()
        speech_options["context"] = context
        settings.update(context=context)
        self.status_label.config(text="Names & slang applied." if context else "Names & slang cleared.")

    @staticmethod
    def _format_interval(seconds):
        return "Off" if seconds is None else f"{seconds:g}"

    def _on_live_interval_changed(self, event=None):
        text = self.live_interval_var.get().strip().lower().removesuffix("s").strip()
        if text in ("off", "0", ""):
            seconds = None
        else:
            try:
                seconds = float(text)
            except ValueError:
                self.live_interval_var.set(self._format_interval(live_options["interval"]))
                self.status_label.config(text="Live caption interval must be a number of seconds, or Off.")
                return
            lo, hi = settings.LIVE_INTERVAL_RANGE
            seconds = min(max(seconds, lo), hi)
        self.live_interval_var.set(self._format_interval(seconds))
        if seconds != live_options["interval"]:
            live_options["interval"] = seconds
            settings.update(live_interval=seconds)
            self.status_label.config(text="Live captions off - captions appear when each speaker pauses."
                                     if seconds is None else f"Live captions update every {seconds:g}s.")

    def _on_translate_live_toggle(self):
        live_options["translate"] = self.translate_live_var.get()
        settings.update(translate_live=live_options["translate"])

    def _on_auto_names_toggle(self):
        speech_options["auto_names"] = self.auto_names_var.get()
        settings.update(auto_names=speech_options["auto_names"])
        self._context_refresh_ticks = 0  # show the change right away

    def _on_mic_toggle(self):
        self.controller.mic_capture.set_enabled(self.mic_var.get())
        settings.update(mic=self.mic_var.get())

    def _on_apply_port(self):
        try:
            port = int(self.port_var.get())
        except ValueError:
            self.status_label.config(text="Discord bridge port must be a number.")
            return
        self.controller.set_bridge_port(port)

    def _poll_status(self):
        self._context_refresh_ticks -= 1
        if self._context_refresh_ticks <= 0:
            self._context_refresh_ticks = 13  # ~every 2s at the 150ms poll
            context = speech_context()
            self.context_shown.config(text=f"Speech model hears: {context}" if context else "")
        latest = None
        while True:
            try:
                latest = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            self.status_label.config(text=str(latest))

        latest_fields = None
        while True:
            try:
                latest_fields = self._status_fields_queue.get_nowait()
            except queue.Empty:
                break
        if latest_fields is not None:
            for key, value_lbl in self._status_value_labels.items():
                value_lbl.config(text=latest_fields.get(key, "..."))
            # Drive the combo's enabled state off the structured status
            # rather than parsing broadcast text - covers both the initial
            # load and any later reload, and can't drift out of sync with
            # what actually happened.
            wm_status = latest_fields.get("whisper_model", "")
            if wm_status.startswith("Loading") or wm_status.startswith("Reloading"):
                self.model_combo.config(state="disabled")
            elif wm_status:
                self.model_combo.config(state="readonly")

        # Polled directly rather than through SystemStatus - it changes
        # continuously rather than on discrete subsystem events.
        gpu_lbl = self._status_value_labels.get("gpu")
        if gpu_lbl is not None:
            if _NVML_HANDLE is not None:
                # Whole-GPU reading via the driver (NVML) - sees Whisper's
                # ctranslate2 allocations and the translator's llama-server process alike.
                info = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                used_gb = info.used / (1024 ** 3)
                total_gb = info.total / (1024 ** 3)
                gpu_lbl.config(text=f"{used_gb:.1f} / {total_gb:.1f} GB")
            elif self._gpu_available:
                # Fallback: PyTorch's own allocator only - misses Whisper
                # entirely (see the pynvml import comment above), but better
                # than nothing if nvidia-ml-py isn't installed.
                allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_lbl.config(text=f"{allocated_gb:.1f} / {self._gpu_total_gb:.1f} GB (PyTorch only, not Whisper - install nvidia-ml-py)")
            else:
                gpu_lbl.config(text="No CUDA GPU")

        self.root.after(150, self._poll_status)

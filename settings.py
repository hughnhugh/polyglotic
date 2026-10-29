"""
Settings remembered between launches, in settings.json next to the app, so
a plain launch comes back the way it was last left: target languages,
speech model, mic capture, the speech model's conversation language and
context words, live caption timing, and the
subtitle overlays' look, positions and lock. Command-line options still
override them for that run.
"""

import json
import os
import threading

ROOT = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(ROOT, "settings.json")


def _default_model():
    # Qwen3-ASR when setup.ps1 -QwenASR installed it - stronger than
    # Whisper on Chinese/Cantonese and with far fewer invented captions.
    installed = (os.path.exists(os.path.join(ROOT, "venv-qwen-asr", "Scripts", "python.exe"))
                 and os.path.isdir(os.path.join(ROOT, "models", "Qwen3-ASR-1.7B")))
    return "qwen3-asr-1.7b" if installed else "large-v3"


DEFAULTS = {
    "targets": ["en"],
    # target -> a second language its overlay translates lines already in
    # the target's language into (shown in yellow, like every translation)
    "second_languages": {},
    "model": None,          # filled in by load() - depends on what's installed
    "mic": True,
    "mic_device": None,     # microphone by name; None = Windows' default
    "caption_mic": True,    # caption your own speech (the mic is still captured for nothing else)
    "language": None,       # None = auto-detect; else a code from CONVERSATION_LANGUAGES
    "context": "",          # names/slang passed to the speech model to bias its spelling
    "live_interval": 1.5,   # seconds between live caption updates; None = off (see overlay_core.live_options)
    "translate_live": True, # translate live updates too, not just each pause's final caption
    # Subtitle overlays (ui/overlay.py)
    "overlay": {
        "text_scale": 1.0,      # x the default text size
        "opacity": 0.72,        # background panel opacity
        "show_original": True,  # the original speech under its translation
        "show_speakers": True,
        "show_language": True,  # the spoken language beside the speaker's name
        "fade_after": 8,        # seconds of silence before captions fade out; None = never
    },
    "overlay_geometry": {},     # target -> [x, y, width, height], once moved/resized
    "locked": False,            # overlays click-through
}

# Live update interval: the dropdown's suggestions; any value in this range
# can be typed in too.
LIVE_INTERVAL_CHOICES = [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]
LIVE_INTERVAL_RANGE = (0.5, 20.0)

# Settings' "Conversation language" choices: code -> label. Forcing one
# skips the speech model's language detection - no more misreads of short
# or mumbled speech as Thai/Malay - for calls in a single language.
CONVERSATION_LANGUAGES = {
    None: "Auto-detect",
    "en": "English only",
    "zh": "Mandarin only",
    "yue": "Cantonese only",
    "vi": "Vietnamese only",
}

_lock = threading.Lock()
_values = None


def load():
    """The saved settings over DEFAULTS. A missing or unreadable file just
    gives the defaults."""
    global _values
    with _lock:
        values = json.loads(json.dumps(DEFAULTS))  # deep copy - nested groups get edited in place
        values["model"] = _default_model()
        try:
            with open(PATH, encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k not in DEFAULTS:
                    continue
                # Nested groups merge, so options added since the file was
                # written still get their defaults.
                values[k] = dict(DEFAULTS[k], **v) if isinstance(DEFAULTS[k], dict) and isinstance(v, dict) else v
        except (OSError, ValueError):
            pass
        _values = values
        return dict(values)


def update(**changes):
    """Merges changes in and writes the file - called from the Settings
    window and pipeline as each option changes."""
    global _values
    with _lock:
        if _values is None:
            _values = json.loads(json.dumps(DEFAULTS))
            _values["model"] = _default_model()
        _values.update(changes)
        tmp = PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_values, f, ensure_ascii=False, indent=2)
            os.replace(tmp, PATH)
        except OSError as e:
            print(f"[settings] couldn't save: {e}")

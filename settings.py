"""
Settings remembered between launches, in settings.json next to the app, so
a plain launch comes back the way it was last left: target languages,
speech model, overlay position, mic capture, the Discord bridge port, and
the speech model's conversation language and context words (and whether
to add the Discord channel members' names to them). Command-line
options still override them for that run.
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
    "model": None,          # filled in by load() - depends on what's installed
    "position": None,       # None = auto (bottom for en, top otherwise)
    "mic": True,
    "bridge_port": 8765,
    "language": None,       # None = auto-detect; else a code from CONVERSATION_LANGUAGES
    "context": "",          # names/slang passed to the speech model to bias its spelling
    "auto_names": True,     # also pass the Discord channel members' names (see overlay_core.speech_context)
}

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
        values = dict(DEFAULTS, model=_default_model())
        try:
            with open(PATH, encoding="utf-8") as f:
                saved = json.load(f)
            values.update({k: v for k, v in saved.items() if k in DEFAULTS})
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
            _values = dict(DEFAULTS, model=_default_model())
        _values.update(changes)
        tmp = PATH + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_values, f, ensure_ascii=False, indent=2)
            os.replace(tmp, PATH)
        except OSError as e:
            print(f"[settings] couldn't save: {e}")

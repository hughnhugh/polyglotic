"""
Which model files the app uses, where it looks for them, and which are
there - in one place, so a downloaded model dropped into models/ is all it
takes to wire it in (models/README.md says what goes where). setup.ps1
downloads the same files, and skips any that are already there.

Sized for an 8GB GPU (an RTX 4060 Ti): by default Qwen3-ASR-0.6B for speech
(2.0GB) beside Hy-MT2-1.8B Q8_0 for translation (2.2GB) - 4.2GB in all, room
to spare for Windows, the browser and Discord. The larger Qwen3-ASR-1.7B
(4.7GB, better on Cantonese) fits beside the more compressed Hy-MT2-1.8B
Q4_K_M (1.5GB) - 6.2GB, tight. (GPU memory measured 2026-11-04, while
transcribing and translating.) The 7B translators are still used if
they're dropped in, for bigger GPUs.

No imports beyond the standard library: settings.py reads it at startup.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")

# The translator, run by llama-server (translator.py): choice -> (file in models/, label, size in GB)
TRANSLATORS = {
    "1.8B-Q8_0": ("Hy-MT2-1.8B-Q8_0.gguf", "Hy-MT2-1.8B Q8_0", 1.9),
    "1.8B-Q4_K_M": ("Hy-MT2-1.8B-Q4_K_M.gguf", "Hy-MT2-1.8B Q4_K_M", 1.1),
    "7B-Q4_K_M": ("Hy-MT2-7B-Q4_K_M.gguf", "Hy-MT2-7B Q4_K_M", 4.6),
    "7B-Q8_0": ("HY-MT2-7B-Q8_0.gguf", "Hy-MT2-7B Q8_0", 8.0),
}
# Smaller than this, a .gguf is a broken copy, not the model.
MIN_GGUF_BYTES = 100_000_000

# Qwen3-ASR speech models (qwen_asr_client.py): choice -> folder in models/
SPEECH = {
    "qwen3-asr-0.6b": "Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen3-ASR-1.7B",
}


def translator_path(name):
    return os.path.join(MODELS_DIR, TRANSLATORS[name][0])


def translator_installed(name):
    path = translator_path(name)
    return os.path.isfile(path) and os.path.getsize(path) >= MIN_GGUF_BYTES


def speech_dir(name):
    return os.path.join(MODELS_DIR, SPEECH[name])


def speech_installed(name):
    """The folder's there with its config and its weights."""
    folder = speech_dir(name)
    try:
        files = os.listdir(folder)
    except OSError:
        return False
    return "config.json" in files and any(f.endswith(".safetensors") for f in files)


def default_speech_model():
    """Qwen3-ASR-1.7B if it's been added (setup.bat -LargerSpeechModel, or
    dropped in), else the 8GB default, 0.6B - else Whisper."""
    for name in ("qwen3-asr-1.7b", "qwen3-asr-0.6b"):
        if speech_installed(name):
            return name
    return "large-v3"


def default_translator(speech_model):
    """The translator that fits beside `speech_model` on an 8GB GPU: Q8_0,
    or Q4_K_M beside the larger speech model - whichever is there."""
    order = ["1.8B-Q8_0", "1.8B-Q4_K_M"]
    if speech_model == "qwen3-asr-1.7b":
        order.reverse()
    for name in order + ["7B-Q4_K_M", "7B-Q8_0"]:
        if translator_installed(name):
            return name
    return order[0]

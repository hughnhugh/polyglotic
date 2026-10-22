"""
Qwen3-ASR as a drop-in alternative to faster-whisper's WhisperModel,
selectable from Settings' speech-model dropdown ("qwen3-asr-1.7b" /
"qwen3-asr-0.6b"). Much stronger than Whisper on Cantonese and Chinese
dialects per Qwen's own benchmarks, and it detects all of its 30 languages
plus Cantonese itself (Whisper reports Cantonese speech as "zh").

The model runs in a child process (qwen_asr_server.py) inside its own venv,
venv-qwen-asr/ - its qwen-asr package pins dependency versions that clash
with faster-whisper's, so it can't share the app's venv.

Known gap vs Whisper: Qwen3-ASR exposes no per-segment confidence
(no_speech_prob / avg_logprob / language probability), so the app's
hallucination filter (overlay_core.is_reliable_transcription) has nothing
to check beyond the VAD gate upstream and an empty-text check here.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from types import SimpleNamespace

import numpy as np

from child_process import free_port, kill_with_parent
from overlay_core import LOG_DIR

ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(ROOT, "venv-qwen-asr", "Scripts", "python.exe")
SERVER_SCRIPT = os.path.join(ROOT, "qwen_asr_server.py")

MODEL_DIRS = {
    "qwen3-asr-1.7b": os.path.join(ROOT, "models", "Qwen3-ASR-1.7B"),
    "qwen3-asr-0.6b": os.path.join(ROOT, "models", "Qwen3-ASR-0.6B"),
}

LOAD_TIMEOUT_SEC = 300
REQUEST_TIMEOUT_SEC = 60

# Qwen3-ASR reports languages by English name; the app works in Whisper's
# codes (overlay meta tags, same-language checks, the translator's
# Chinese-prompt choice). "Filipino" maps to "tl", matching lang_codes.
LANGUAGE_CODES = {
    "chinese": "zh", "english": "en", "cantonese": "yue", "arabic": "ar", "german": "de", "french": "fr",
    "spanish": "es", "portuguese": "pt", "indonesian": "id", "italian": "it", "korean": "ko", "russian": "ru",
    "thai": "th", "vietnamese": "vi", "japanese": "ja", "turkish": "tr", "hindi": "hi", "malay": "ms",
    "dutch": "nl", "swedish": "sv", "danish": "da", "finnish": "fi", "polish": "pl", "czech": "cs",
    "filipino": "tl", "persian": "fa", "greek": "el", "hungarian": "hu", "macedonian": "mk", "romanian": "ro",
}


# The other way, for a forced conversation language (overlay_core.speech_options).
LANGUAGE_NAMES = {code: name.capitalize() for name, code in LANGUAGE_CODES.items()}


def is_qwen_model(size):
    return size in MODEL_DIRS


class QwenASRModel:
    """Starts the server on construction (blocking until loaded, like
    WhisperModel's constructor) and exposes the slice of WhisperModel's
    transcribe() interface the app uses."""

    def __init__(self, size):
        model_dir = MODEL_DIRS[size]
        missing = [p for p in (PYTHON, model_dir) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"missing {', '.join(os.path.relpath(p, ROOT) for p in missing)} - "
                                    "run setup.ps1 with -QwenASR")
        port = free_port()
        self._url = f"http://127.0.0.1:{port}"
        os.makedirs(LOG_DIR, exist_ok=True)
        self._log = open(os.path.join(LOG_DIR, "qwen-asr-server.log"), "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [PYTHON, SERVER_SCRIPT, model_dir, str(port)],
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        kill_with_parent(self._proc)

        deadline = time.time() + LOAD_TIMEOUT_SEC
        while time.time() < deadline:
            if self._proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{self._url}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        self.close()
        raise RuntimeError("Qwen3-ASR server failed to start - see logs/qwen-asr-server.log")

    def transcribe(self, audio, language=None, initial_prompt=None, **_whisper_only_options):
        """Returns (segments, info) shaped like faster-whisper's: segments
        have .text/.no_speech_prob/.avg_logprob, info has .language/
        .language_probability. Qwen gives no confidences, so those are
        neutral values that pass the reliability filter - an empty
        transcription, or one Qwen flags as silent, comes back as no
        segments instead, which fails it.
        language: a code to force instead of detecting (raises ValueError
        for one Qwen doesn't support, like WhisperModel does);
        initial_prompt: Whisper's name for it, passed as Qwen's context -
        names and slang it should spell the way they're written there."""
        params = {}
        if language:
            if language not in LANGUAGE_NAMES:
                raise ValueError(f"Qwen3-ASR doesn't support language {language!r}")
            params["language"] = LANGUAGE_NAMES[language]
        if initial_prompt:
            params["context"] = initial_prompt
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        req = urllib.request.Request(f"{self._url}/transcribe{query}",
                                     data=np.asarray(audio, dtype="<f4").tobytes(),
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as r:
            result = json.load(r)

        # Mixed-language speech is reported as e.g. "Chinese,English" - the
        # first is the one the app treats as the source; all of them go in
        # info.languages, so a line mixing in a window's own language still
        # gets translated. An empty language is Qwen's own "silent / no
        # speech" signal.
        names = [n.strip() for n in result["language"].split(",") if n.strip()]
        name = names[0] if names else ""
        code = LANGUAGE_CODES.get(name.lower(), name.lower())
        languages = list(dict.fromkeys(LANGUAGE_CODES.get(n.lower(), n.lower()) for n in names))
        info = SimpleNamespace(language=code or "unknown", language_probability=1.0, languages=languages)
        text = result["text"].strip()
        if not text or not name:
            return [], info
        return [SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=0.0)], info

    def close(self):
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._log.close()

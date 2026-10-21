"""
Text translation via Tencent's Hy-MT2-7B (a translation-tuned LLM), served
by a local llama.cpp `llama-server` child process - both fetched into bin/
and models/ by setup.ps1.

Replaced NLLB-200 after a side-by-side on real captions from logs/ (see
eval/compare_translators.py): far better on short, casual voice-chat lines
(fillers, fragments, slang), which NLLB rendered stiffly or decoded into
repetition loops, and better than Whisper's own X->English translate task,
which occasionally invented content outright.

The prompt tells the model it's translating a casual, speech-recognized
voice-chat line, so it keeps the spoken register instead of formalizing
it. Previous captions are deliberately NOT passed as context: in the same
evaluation that helped some ambiguous fragments but more often led the
model to over-interpret a line into something fluent but different.
"""

import atexit
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

from child_process import free_port, kill_with_parent
from lang_codes import CHINESE_CODES, TARGET_LANGUAGES
from overlay_core import LOG_DIR, broadcast, status

ROOT = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.path.join(ROOT, "bin", "llama-server.exe")
MODEL_NAME = "Hy-MT2-7B"
MODEL_PATH = os.path.join(ROOT, "models", "Hy-MT2-7B-Q4_K_M.gguf")

CONTEXT_TOKENS = 2048        # one caption line plus prompt fits easily; keeps the KV cache (VRAM) small
MAX_OUTPUT_TOKENS = 256
LOAD_TIMEOUT_SEC = 180
READY_WAIT_SEC = 60          # how long a translate() call made during startup waits for the model
REQUEST_TIMEOUT_SEC = 30

_proc = None
_base_url = None
_ready = threading.Event()
_failed = threading.Event()


def start():
    """Launch the server in the background - returns immediately; the
    Settings window's "Translation" status row tracks the load."""
    global _proc, _base_url

    missing = [p for p in (LLAMA_SERVER, MODEL_PATH) if not os.path.exists(p)]
    if missing:
        status.set("translator", "Not installed - run setup.bat")
        broadcast(f"[Translation unavailable - missing {', '.join(os.path.relpath(p, ROOT) for p in missing)}; "
                  "run setup.bat. Showing transcriptions only.]")
        _failed.set()
        return

    status.set("translator", f"Loading {MODEL_NAME}...")
    port = free_port()
    _base_url = f"http://127.0.0.1:{port}"
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "llama-server.log")
    log_file = open(log_path, "w", encoding="utf-8")
    _proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", MODEL_PATH, "--host", "127.0.0.1", "--port", str(port),
         "-ngl", "999", "-c", str(CONTEXT_TOKENS), "-np", "1", "-fa", "on", "--jinja", "--no-webui"],
        stdout=log_file, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kill_with_parent(_proc)  # can never outlive the app and keep ~5GB of VRAM pinned
    atexit.register(stop)

    def _wait_until_ready():
        deadline = time.time() + LOAD_TIMEOUT_SEC
        while time.time() < deadline:
            if _proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{_base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        status.set("translator", f"Ready ({MODEL_NAME})")
                        _ready.set()
                        return
            except (urllib.error.URLError, OSError):
                pass  # still loading (503) or not listening yet
            time.sleep(0.5)
        status.set("translator", f"Failed - see {os.path.relpath(log_path, ROOT)}")
        broadcast("[Translation model failed to load - showing transcriptions only]")
        _failed.set()

    threading.Thread(target=_wait_until_ready, daemon=True).start()


def stop():
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _proc.kill()


def _prompt(text, src_lang, target):
    # Tencent's recommendation: prompt in Chinese whenever Chinese is on
    # either side of the translation, English otherwise. Both follow the
    # model card's "personalization" template.
    _, name_en, name_zh = TARGET_LANGUAGES[target]
    if src_lang in CHINESE_CODES or target in CHINESE_CODES:
        return (f"【待翻译文本】\n{text}\n\n【翻译任务】\n"
                "1、这是一段轻松随意的实时语音聊天中的一句话，由语音识别转写而来，可能有识别错误或不完整的句子。\n"
                "2、保持口语化、随意的语气，俚语、语气词和感叹词按意思自然地翻译，不要变得书面或正式。\n"
                "3、只输出【待翻译文本】的译文，不要额外解释。\n"
                f"4、将【待翻译文本】翻译为{name_zh}。")
    return (f"[Source Text]\n{text}\n\n[Translation Tasks]\n"
            "1. The text is one line from a casual live voice chat, transcribed by speech recognition. It may "
            "contain recognition errors or be an incomplete sentence.\n"
            "2. Keep the casual, spoken tone. Translate slang, filler words and interjections naturally by "
            "meaning; do not make it more formal.\n"
            "3. Only output the translation of the [Source Text], without any additional explanation.\n"
            f"4. Translate the [Source Text] into {name_en}.")


def translate(text, src_lang, target):
    """src_lang: Whisper's detected language code (any); target: a key of
    TARGET_LANGUAGES. Returns "" if the translator isn't available or the
    request fails - callers show the native text alone in that case."""
    if not _ready.wait(READY_WAIT_SEC if not _failed.is_set() else 0):
        return ""
    body = json.dumps({
        "messages": [{"role": "user", "content": _prompt(text, src_lang, target)}],
        "max_tokens": MAX_OUTPUT_TOKENS,
        # Greedy decoding plus the model card's light repetition penalty -
        # the configuration evaluated in eval/compare_translators.py.
        "temperature": 0,
        "repeat_penalty": 1.05,
    }).encode("utf-8")
    req = urllib.request.Request(f"{_base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as r:
            reply = json.load(r)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[translator] request failed: {e}")
        return ""
    return (reply["choices"][0]["message"].get("content") or "").strip()

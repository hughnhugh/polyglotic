"""
The local assistant: a small general-purpose LLM - Qwen3.5-4B, on its own
llama.cpp `llama-server` (the translator's, in bin/) - behind two features:

- "Explain" in a word lookup (ui/lookup.py): what a word or phrase means in
  the sentence it was heard in, how it's built and used - or, for a phrase
  in a translation, which words in the original say it;
- the side chat (ui/assistant_chat.py), opened from an overlay: chat with
  the conversation as context, summarise it, explain the last line, help
  replying.

Optional, and off until switched on (Models page): setup.bat -Assistant
downloads the model (2.7GB), and it holds ~3GB of VRAM only while on.
It's a separate server from the translator's on purpose: that one serves a
single request at a time, so an explanation streaming there would hold up
live captions - and Hy-MT2 is tuned to output translations and nothing else.

Pinyin and definitions come from the dictionary (dictionary.py), not from
here - a 4B model gets tones wrong often enough that it isn't asked for them.
"""

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

import dictionary
from child_process import free_port, kill_with_parent
from overlay_core import LOG_DIR, status

ROOT = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER = os.path.join(ROOT, "bin", "llama-server.exe")
MODEL_NAME = "Qwen3.5-4B"
MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-4B-Q4_K_M.gguf")

# Room for a summary of the last stretch of conversation (CONVERSATION_CHARS)
# plus the chat. Cheap: most of Qwen3.5's layers keep no per-token cache.
CONTEXT_TOKENS = 16384
CONVERSATION_CHARS = 9000    # of captions given as context - roughly the last 15-30 minutes of a lively call
EXPLAIN_MAX_TOKENS = 450
CHAT_MAX_TOKENS = 900
LOAD_TIMEOUT_SEC = 120
READY_WAIT_SEC = 90          # how long a request made while it's loading waits
REQUEST_TIMEOUT_SEC = 60

_lock = threading.Lock()
_proc = None
_base_url = None
_ready = threading.Event()
_failed = threading.Event()


def installed():
    return os.path.exists(LLAMA_SERVER) and os.path.exists(MODEL_PATH)


def running():
    return _proc is not None and _proc.poll() is None


def ready():
    return running() and _ready.is_set()


def start():
    """Launch the server in the background - returns immediately; the
    "assistant" status tracks the load."""
    global _proc, _base_url, _ready, _failed
    with _lock:
        if running():
            return
        if not installed():
            status.set("assistant", "Not installed - run setup.bat -Assistant")
            return
        _ready, _failed = threading.Event(), threading.Event()
        ready, failed = _ready, _failed
        status.set("assistant", f"Loading {MODEL_NAME}...")
        port = free_port()
        _base_url = f"http://127.0.0.1:{port}"
        os.makedirs(LOG_DIR, exist_ok=True)
        log_path = os.path.join(LOG_DIR, "llama-server-assistant.log")
        log_file = open(log_path, "w", encoding="utf-8")
        _proc = proc = subprocess.Popen(
            [LLAMA_SERVER, "-m", MODEL_PATH, "--host", "127.0.0.1", "--port", str(port),
             "-ngl", "999", "-c", str(CONTEXT_TOKENS), "-np", "1", "-fa", "on", "--jinja", "--no-webui",
             "--cache-ram", "0", "--load-mode", "none"],  # as the translator's - see translator.start
            stdout=log_file, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        kill_with_parent(proc)

    def _wait_until_ready():
        deadline = time.time() + LOAD_TIMEOUT_SEC
        while time.time() < deadline and proc is _proc:
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(f"{_base_url}/health", timeout=2) as r:
                    if r.status == 200:
                        status.set("assistant", f"Ready ({MODEL_NAME})")
                        ready.set()
                        return
            except (urllib.error.URLError, OSError):
                pass  # still loading (503) or not listening yet
            time.sleep(0.5)
        if proc is _proc:  # not just stopped on purpose
            status.set("assistant", f"Failed - see {os.path.relpath(log_path, ROOT)}")
        failed.set()

    threading.Thread(target=_wait_until_ready, daemon=True).start()


def stop():
    """Shuts the server down, freeing its VRAM."""
    global _proc
    with _lock:
        proc, _proc = _proc, None
        _failed.set()  # wakes any request still waiting for it
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    status.set("assistant", "Off")


def chat(messages, on_text, cancelled=lambda: False, max_tokens=CHAT_MAX_TOKENS, temperature=0.7):
    """Streams the model's reply to `messages` (OpenAI-style [{role,
    content}]) - on_text(text so far) as it grows. Returns the whole text;
    raises RuntimeError when the model isn't available or the request
    fails. Call from a worker thread."""
    deadline = time.time() + READY_WAIT_SEC
    while not _ready.is_set():
        if _failed.is_set() or not running() or time.time() > deadline:
            raise RuntimeError("The local assistant model isn't running - switch it on in Models.")
        time.sleep(0.2)
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        # Qwen3.5 thinks before answering by default - minutes of hidden
        # tokens for a popup. Its model card's settings for answering directly:
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": temperature, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
    }).encode("utf-8")
    req = urllib.request.Request(f"{_base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SEC) as r:
            for raw in r:
                if cancelled():
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})
                if delta.get("content"):
                    text += delta["content"]
                    on_text(text)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise RuntimeError(f"The request to the local assistant failed: {e}") from e
    return text.strip()


# -- Explain ----------------------------------------------------------------------

def _explain_prompt(phrase, sentence, translation, meanings):
    lines = [f'Sentence: {sentence}']
    if translation:
        lines.append(f"Translation: {translation}")
    lines.append(f'Phrase: {phrase}')
    if meanings:
        lines.append(f"Dictionary: {meanings}")
    lines.append(
        "\nThe sentence is one line from a casual voice chat, transcribed by speech recognition, so it may "
        "contain mistakes. I'm an English speaker learning Chinese. Explain the phrase as it's used in this "
        "sentence, briefly (under 120 words), in plain English:\n"
        "- what it means here, and its literal meaning if that's different\n"
        "- for more than one character, what each character contributes\n"
        "- how it's used - casual or formal, slang, tone\n"
        "- if it's a chengyu or set phrase, the image or story behind it in one sentence\n"
        "Only if the phrase doesn't fit the sentence: it may be a speech-recognition mistake - say what was "
        "probably said. Otherwise don't mention speech recognition. Don't give pinyin. Use **bold** for "
        "Chinese words you mention, no headings.")
    return "\n".join(lines)


def _match_prompt(phrase, sentence, other):
    return (f"Line: {sentence}\nThe same line in another language: {other}\nPhrase (from the first line): {phrase}\n\n"
            "These are one line from a casual voice chat, transcribed by speech recognition and machine-translated, "
            "so they may contain mistakes. I'm an English speaker learning Chinese. Which words in the other line "
            "say what the phrase says? Quote them in **bold**, then explain them briefly (under 120 words), in plain "
            "English: what they mean, what each part contributes, and how they're used - casual or formal, slang, "
            "tone. If nothing there matches the phrase, say so. Don't give pinyin, no headings.")


def explain(phrase, sentence, translation, meanings, on_text, cancelled=lambda: False):
    """Streams an explanation of `phrase` in `sentence` (see chat()).
    `translation` - the line in another language - and `meanings` (the
    dictionary's, a short string) help a small model stay on track. For a
    phrase that isn't Chinese (a translation's), it finds the words that
    say it in the other line, and explains those."""
    if not dictionary.has_han(phrase) and translation:
        prompt = _match_prompt(phrase, sentence, translation)
    else:
        prompt = _explain_prompt(phrase, sentence, translation, meanings)
    return chat([{"role": "user", "content": prompt}], on_text, cancelled, EXPLAIN_MAX_TOKENS)


# -- the side chat's context --------------------------------------------------------

# Not "the user": the model then calls the people in the call users too, and
# mixes them up with whoever it's helping.
ME = "Me"


def conversation(captions, labels, limit=CONVERSATION_CHARS):
    """The captions as a transcript for the model: who's talking, then one
    line per turn - a person's back-to-back captions joined - oldest first,
    the most recent turns that fit in `limit` characters. The user's own
    lines are ME, and lines not in English get an English translation
    alongside, which steadies a small model on misrecognized speech. Live
    (unfinished) captions are left out.

    Speakers with Chinese in their names are "Speaker A", "Speaker B"... -
    a model can't help reading such a name as part of the conversation
    (我在东北玩泥巴, "I'm in the Northeast playing in mud", becomes something
    they said), quoted or not - and so is that name where someone says it.
    `labels` ({name: label}) is added to as speakers turn up, and kept by the
    caller so a person's label lasts the session; restore_names() puts the
    real names back into the model's answer."""
    finished = [c for c in captions if not c.live]
    for c in finished:
        if not c.is_you and c.speaker not in labels and dictionary.has_han(c.speaker):
            n = len(labels)
            labels[c.speaker] = f"Speaker {chr(ord('A') + n)}" if n < 26 else f"Speaker {n + 1}"
    spoken = {_simplified(max(_han_runs(name), key=len)): label for name, label in labels.items()}
    spoken = {form: label for form, label in spoken.items() if len(form) > 1}  # 雪 alone is just a word
    turns = []  # [who, time, [texts]]
    for c in finished:
        who = ME if c.is_you else labels.get(c.speaker) or c.speaker or "Someone"
        text, named = _label_names(c.native, spoken)
        english = c.translations.get("en")
        if english and english[1] and english[0] and not named:  # a name in it - its translation garbles that
            text += f" (English: {english[0]})"
        if turns and turns[-1][0] == who:
            turns[-1][2].append(text)
        else:
            turns.append([who, c.time, [text]])
    kept, used = [], 0
    for who, when, texts in reversed(turns):
        line = f"[{when:%H:%M}] {who}: {' '.join(texts)}"
        if used + len(line) > limit:
            break
        kept.append((who, line))
        used += len(line) + 1
    if not kept:
        return ""
    kept.reverse()
    people = ", ".join(dict.fromkeys(who for who, _ in kept))
    return f"People talking: {people}\n\n" + "\n".join(line for _, line in kept)


def restore_names(answer, labels):
    """The model's answer with the real names back in place of the labels
    conversation() gave them - and "You" for ME, which it sometimes calls
    the user by ("Me confirmed...")."""
    for name, label in sorted(labels.items(), key=lambda item: -len(item[1])):  # "Speaker AB" before "Speaker A"
        answer = re.sub(rf"\b{re.escape(label)}\b", lambda _: name, answer, flags=re.IGNORECASE)
    return re.sub(rf"\b{ME}\b", "You", answer)  # capitalized - as a name; "me" is left alone


def _han_runs(text):
    return re.findall(r"[㐀-鿿豈-﫿]+", text) or [text]


def _simplified(text):
    """`text` with each Traditional character in Simplified, one for one -
    so positions still line up (speech is captioned in Simplified, while a
    name may be Traditional: 從火來暖 said is 从火来暖)."""
    out = []
    for ch in text:
        entries = dictionary.lookup(ch) if dictionary.is_han(ch) else []
        out.append(next((e.simplified for e in entries if e.traditional == ch and len(e.simplified) == 1), ch))
    return "".join(out)


def _label_names(text, spoken):
    """(text, whether it named anyone): `text` with the labelled names said
    in it ({simplified name: label}) replaced by their labels."""
    if not spoken or not dictionary.has_han(text):
        return text, False
    simple, named = _simplified(text), False
    for name in sorted(spoken, key=len, reverse=True):
        while name in simple:
            at = simple.index(name)
            text = text[:at] + spoken[name] + text[at + len(name):]
            simple = simple[:at] + spoken[name] + simple[at + len(name):]
            named = True
    return text, named

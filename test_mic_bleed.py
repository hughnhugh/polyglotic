"""Verifies the app keeps your speech apart from the PC's audio leaking
into your mic (Discord through headphones): overlay_core.is_bleed on the
audio - including with your own clicks and taps on the mic, and talking
over someone - using synthetic speech-like audio (bursts of
syllable-length noise); and the text backup in process_segment, which
drops or takes back a "You" caption repeating someone else's. Also mixed-
language lines being translated into their own main language. No GPU."""

import queue
import types
from collections import deque

import numpy as np

import overlay_core
import translate_vc
import translator

SR = 16000
T0 = 1000.0
SECONDS = 6


def _speech(seed):
    """Syllable-like bursts: 120-320ms of shaped noise with short gaps."""
    rng = np.random.default_rng(seed)
    audio = np.zeros(SR * SECONDS)
    t = 0.0
    while t < SECONDS - 0.4:
        length = rng.uniform(0.12, 0.32)
        start, n = int(t * SR), int(length * SR)
        audio[start:start + n] += rng.normal(0, 0.25, n) * np.hanning(n)
        t += length + rng.uniform(0.04, 0.25)
    return audio


def _buffer(audio):
    buffer, chunk = deque(), 1024
    for i in range(0, len(audio) - chunk + 1, chunk):
        buffer.append((T0 + i / SR, audio[i:i + chunk].astype(np.float32)))
    return buffer


def _late(audio, ms):
    return np.concatenate([np.zeros(int(SR * ms / 1000)), audio])[:len(audio)]


def main():
    rng = np.random.default_rng(7)
    them, me = _speech(1), _speech(2)
    noise = rng.normal(0, 0.002, SR * SECONDS)
    taps = np.zeros(SR * SECONDS)
    for _ in range(8):
        start = int(rng.uniform(0, SECONDS - 0.2) * SR)
        taps[start:start + 400] += rng.normal(0, 0.25, 400)
    hum = 0.2 * np.sin(np.arange(SR * SECONDS) * 2 * np.pi * 220 / SR) * (
        1 + np.sin(np.arange(SR * SECONDS) * 2 * np.pi * 0.5 / SR))

    cases = [
        ("their voice leaking in, quiet and 120ms late", them, 0.04 * _late(them, 120) + noise, True),
        ("leaking in, with my clicks and taps on the mic", them, 0.04 * _late(them, 150) + 0.6 * taps + noise, True),
        ("leaking in loud", them, 0.15 * _late(them, 80) + noise, True),
        ("me talking over them", them, 0.5 * me + 0.04 * _late(them, 120) + noise, False),
        ("me talking, the PC silent", np.zeros(SR * SECONDS), 0.5 * me + noise, False),
        ("me talking over music", hum, 0.5 * me + 0.03 * _late(hum, 100) + noise, False),
    ]
    for name, played, mic, expected in cases:
        got = overlay_core.is_bleed(_buffer(mic), T0 + 0.5, T0 + 5.5, loopback_buffer=_buffer(played))
        assert got == expected, f"{name}: is_bleed={got}, expected {expected}"
        print(f"OK: {name} -> {'bleed, dropped' if got else 'you, kept'}")

    # -- text backup and mixed languages, through process_segment ----------
    translator.translate = lambda text, src, tgt: f"<{tgt}:{text}>"
    q = queue.Queue()
    process = translate_vc.make_process_segment(["en", "zh"], q)

    def model(text, lang, languages=None):
        segment = types.SimpleNamespace(text=text, no_speech_prob=0.0, avg_logprob=-0.2)
        info = types.SimpleNamespace(language=lang, language_probability=0.99)
        if languages:
            info.languages = languages
        return types.SimpleNamespace(transcribe=lambda *a, **k: (iter([segment]), info))

    def drain():
        out = []
        while not q.empty():
            out.append(q.get())
        return out

    process(model("So where's China's chip industry at right now?", "en"), "AUDIO", "", ("1", "marina"),
            caption_group="d1", final=True)
    process(model("where's China's chip industry at", "en"), "AUDIO", "mic", None, caption_group="m1", final=True)
    out = drain()
    assert [r.get("speaker") for r in out if r.get("native")] == ["marina"] and out[-1]["native"] is None, out
    print("OK: a \"You\" caption repeating marina's line is dropped")

    process(model("we should go eat hot pot tonight", "en"), "AUDIO", "mic", None, caption_group="m2", final=True)
    process(model("We should go eat hot pot tonight!", "en"), "AUDIO", "", ("2", "Caleb"), caption_group="d2", final=True)
    out = drain()
    assert any(r.get("group") == "m2" and r.get("native") is None for r in out), out
    print("OK: a \"You\" caption is taken back when Caleb's caption turns out to be the same speech")

    process(model("I'm hungry, what are we doing", "en"), "AUDIO", "mic", None, caption_group="m3", final=True)
    assert [r["speaker"] for r in drain()] == ["You"]
    print("OK: your own speech is kept")

    process(model("反正就是一个非常specific的这种culture concept嘛", "zh"), "AUDIO", "", ("3", "kaektus"))
    record = drain()[0]
    assert record["langs"] == ["zh", "en"] and record["translations"]["zh"][1] is True, record
    process(model("我觉得这个地方很好", "zh"), "AUDIO", "", ("3", "kaektus"))
    record = drain()[0]
    assert record["langs"] == ["zh"] and record["translations"]["zh"] == ("我觉得这个地方很好", False), record
    print("OK: a mixed Chinese/English line is translated into Chinese too; a plain one isn't")
    print("ALL OK")


if __name__ == "__main__":
    main()
